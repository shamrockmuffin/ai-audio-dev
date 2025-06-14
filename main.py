import streamlit as st
import asyncio
from pathlib import Path
import tempfile
import os
import logging
from datetime import datetime
import json
import shutil
import numpy as np
from typing import Dict, Any

# Handle potential import issues
try:
    from config.settings import settings
    from core.professional_audio_processor import ProfessionalAudioProcessor, AudioMetrics
    from core.transcription_handler import TranscriptionHandler
    from core.security.audio_security import AudioFileValidator
    from services.claude_service import ClaudeService
except ImportError as e:
    st.error(f"""
    **Import Error Detected**: {str(e)}
    
    This is likely due to a PyTorch/torchvision dependency conflict. 
    
    **To fix this issue:**
    1. Run the diagnostic script: `python fix_pytorch_deps.py`
    2. Or manually reinstall PyTorch ecosystem:
       ```
       .conda\\Scripts\\pip.exe uninstall torch torchvision torchaudio -y
       .conda\\Scripts\\pip.exe install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
       .conda\\Scripts\\pip.exe install --upgrade transformers
       ```
    3. Restart the application
    
    **Current Error**: {str(e)}
    """)
    st.stop()
from ui.components import (
    render_audio_player,
    render_waveform,
    render_spectrogram,
    render_enhancement_controls,
    render_transcription_view
)
from ui.session_state import SessionStateManager
from utils.audio_helpers import save_audio_file, get_audio_metadata
from utils.cache_manager import cache_manager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)

# Page configuration
st.set_page_config(
    page_title="Audio Analysis & Transcription",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
        color: #1f2937;
    }
    .status-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d1fae5;
        color: #065f46;
    }
    .error-box {
        background-color: #fee2e2;
        color: #991b1b;
    }
    .info-box {
        background-color: #dbeafe;
        color: #1e40af;
    }
</style>
""", unsafe_allow_html=True)

class AudioAnalyzerApp:
    def __init__(self):
        self.session_manager = SessionStateManager()
        self.audio_processor = ProfessionalAudioProcessor(
            target_sample_rate=48000,
            max_true_peak=-1.0,    # EBU R128 standard
            target_lufs=-16.0      # Streaming standard
        )
        self.transcription_handler = TranscriptionHandler()
        self.claude_service = ClaudeService()
        
    async def run(self):
        # Initialize session state
        state = self.session_manager.get_state()
        
        # Header
        st.markdown('<h1 class="main-header">🎵 Audio Analysis & Transcription</h1>', 
                   unsafe_allow_html=True)
        
        # Sidebar
        with st.sidebar:
            st.header("📁 File Management")
            
            # File uploader
            uploaded_file = st.file_uploader(
                "Upload Audio File",
                type=["wav", "mp3", "m4a", "flac", "ogg"],
                help="Supported formats: WAV, MP3, M4A, FLAC, OGG"
            )
            
            if uploaded_file:
                # Save uploaded file
                file_path = await self._handle_file_upload(uploaded_file)
                if file_path:
                    state.add_file(file_path, uploaded_file.name)
            
            # File history
            if state.file_history:
                st.divider()
                st.subheader("📋 Recent Files")
                for idx, file_info in enumerate(state.file_history[-5:]):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        if st.button(f"📄 {file_info['name']}", key=f"file_{idx}"):
                            state.current_file = file_info['path']
                            st.rerun()
                    with col2:
                        if st.button("🗑️", key=f"del_{idx}"):
                            state.remove_file(idx)
                            st.rerun()
            
            # Settings
            st.divider()
            st.subheader("⚙️ Settings")
            
            state.settings['noise_reduction'] = st.checkbox(
                "Noise Reduction", 
                value=state.settings.get('noise_reduction', True)
            )
            
            state.settings['normalize'] = st.checkbox(
                "Audio Normalization", 
                value=state.settings.get('normalize', True)
            )
            
            state.settings['enhance_transcription'] = st.checkbox(
                "AI Transcription Enhancement", 
                value=state.settings.get('enhance_transcription', True)
            )
            
            state.settings['speaker_diarization'] = st.checkbox(
                "Speaker Diarization (PyAnnote)", 
                value=state.settings.get('speaker_diarization', False),
                help="Identify and separate different speakers using PyAnnote.audio"
            )
            
            # Cache management
            st.divider()
            st.subheader("💾 Cache Management")
            
            cache_stats = cache_manager.get_cache_stats()
            st.metric("Cache Size", f"{cache_stats['total_size_mb']:.1f} MB")
            st.metric("Cached Files", cache_stats['total_files'])
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🗑️ Clear Cache"):
                    cache_manager.clear_cache()
                    st.success("Cache cleared!")
                    st.rerun()
            
            with col2:
                if st.button("📊 Cache Stats"):
                    with st.expander("Cache Details", expanded=True):
                        for cache_type, stats in cache_stats['by_type'].items():
                            st.write(f"**{cache_type.title()}**: {stats['files']} files, {stats['size_mb']:.1f} MB")
        
        # Main content area
        if state.current_file:
            await self._render_main_content(state)
        else:
            self._render_welcome_screen()
    
    async def _handle_file_upload(self, uploaded_file) -> str:
        """Handle file upload with security validation"""
        try:
            # Initialize security components
            validator = AudioFileValidator(
                max_file_size_mb=settings.MAX_FILE_SIZE_MB,
                allowed_formats=['wav', 'mp3', 'flac', 'ogg', 'm4a'],
                enable_deep_scan=True
            )
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp.write(uploaded_file.getbuffer())
                tmp_path = tmp.name
            
            # Validate file
            validation = await validator.validate_file(
                tmp_path,
                original_filename=uploaded_file.name
            )
            
            if not validation.is_safe:
                os.unlink(tmp_path)  # Clean up
                st.error(f"Security validation failed: {', '.join(validation.issues)}")
                return None
            
            # Move to secure storage
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_filename = f"{timestamp}_{validation.hash_sha256[:8]}.{validation.file_type}"
            final_path = settings.TEMP_DIR / safe_filename
            
            # Ensure temp directory exists
            settings.TEMP_DIR.mkdir(parents=True, exist_ok=True)
            
            shutil.move(tmp_path, final_path)
            os.chmod(final_path, 0o644)  # Set appropriate permissions
            
            # Display validation info
            st.success(f"✅ File validated successfully")
            with st.expander("📋 File Validation Details"):
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("File Size", f"{validation.file_size / 1024 / 1024:.1f} MB")
                    st.metric("File Type", validation.file_type.upper())
                with col2:
                    if validation.duration:
                        st.metric("Duration", f"{validation.duration:.1f}s")
                    if validation.sample_rate:
                        st.metric("Sample Rate", f"{validation.sample_rate} Hz")
            
            return str(final_path)
            
        except Exception as e:
            st.error(f"Error uploading file: {str(e)}")
            logging.error(f"File upload error: {e}")
            return None
    
    async def _render_main_content(self, state):
        """Render main content area with tabs"""
        tab1, tab2, tab3, tab4 = st.tabs([
            "🎵 Audio Analysis", 
            "✨ Enhancement", 
            "📝 Transcription",
            "💬 AI Assistant"
        ])
        
        with tab1:
            await self._render_analysis_tab(state)
        
        with tab2:
            await self._render_enhancement_tab(state)
        
        with tab3:
            await self._render_transcription_tab(state)
            
        with tab4:
            await self._render_assistant_tab(state)
    
    async def _render_analysis_tab(self, state):
        """Render audio analysis tab"""
        st.header("Audio Analysis")
        
        # Get audio metadata
        metadata = get_audio_metadata(state.current_file)
        
        # Display metadata
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Duration", f"{metadata['duration']:.2f}s")
        with col2:
            st.metric("Sample Rate", f"{metadata['sample_rate']} Hz")
        with col3:
            st.metric("Bit Depth", metadata.get('bit_depth', 'N/A'))
        with col4:
            st.metric("Channels", metadata.get('channels', 1))
        
        # Audio player
        st.subheader("🎧 Audio Player")
        render_audio_player(state.current_file)
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Waveform")
            render_waveform(state.current_file)
        
        with col2:
            st.subheader("🌈 Spectrogram")
            render_spectrogram(state.current_file)
    
    async def _render_enhancement_tab(self, state):
        """Render audio enhancement tab"""
        st.header("Audio Enhancement")
        
        # Enhancement controls
        render_enhancement_controls(state)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🚀 Enhance Audio", type="primary", use_container_width=True):
                await self._enhance_audio(state)
        
        with col2:
            if state.enhanced_file and st.button("💾 Download Enhanced", use_container_width=True):
                with open(state.enhanced_file, 'rb') as f:
                    st.download_button(
                        label="Download",
                        data=f.read(),
                        file_name=f"enhanced_{Path(state.current_file).name}",
                        mime="audio/wav"
                    )
        
        # Show comparison if enhanced
        if state.enhanced_file:
            st.divider()
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Audio")
                render_audio_player(state.current_file)
                render_waveform(state.current_file)
            
            with col2:
                st.subheader("Enhanced Audio")
                render_audio_player(state.enhanced_file)
                render_waveform(state.enhanced_file)
    
    async def _render_transcription_tab(self, state):
        """Render transcription tab"""
        st.header("Audio Transcription")
        
        # Transcription options
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📝 Transcribe Audio", type="primary", use_container_width=True):
                await self._transcribe_audio(state)
        
        with col2:
            if st.button("👥 Transcribe with Speakers", type="secondary", use_container_width=True):
                await self._transcribe_with_speakers(state)
        
        if state.transcription:
            # Show speaker analysis if available
            if 'speaker_analysis' in state.transcription:
                self._render_speaker_analysis(state.transcription['speaker_analysis'])
                st.divider()
            
            render_transcription_view(state.transcription)
            
            # Export options
            st.divider()
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📄 Export as TXT"):
                    self._export_transcription(state, 'txt')
            
            with col2:
                if st.button("📊 Export as JSON"):
                    self._export_transcription(state, 'json')
            
            with col3:
                if st.button("📑 Export as SRT"):
                    self._export_transcription(state, 'srt')
    
    async def _render_assistant_tab(self, state):
        """Render AI assistant chat tab"""
        st.header("AI Assistant")
        
        # Initialize chat history
        if 'chat_history' not in state.__dict__:
            state.chat_history = []
        
        # Display chat history
        for message in state.chat_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask about your audio..."):
            # Add user message
            state.chat_history.append({"role": "user", "content": prompt})
            
            # Get AI response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = await self._get_ai_response(state, prompt)
                    st.write(response)
            
            # Add assistant message
            state.chat_history.append({"role": "assistant", "content": response})
    
    async def _enhance_audio(self, state):
        """Process audio enhancement with streaming and caching"""
        progress_bar = st.progress(0)
        status = st.empty()
        
        try:
            status.text("🔄 Processing audio...")
            
            # Check cache first
            cache_key = cache_manager.get_cache_key(state.current_file, state.settings)
            if cache_key:
                cached_result = cache_manager.get_cached_result(cache_key, "audio")
                if cached_result:
                    state.enhanced_file = cached_result['enhanced_file']
                    state.audio_metrics = cached_result['metrics']
                    progress_bar.progress(1.0)
                    status.success("✅ Audio loaded from cache!")
                    self._display_professional_metrics(state.audio_metrics)
                    return
            
            # Check file size for streaming decision
            file_size = os.path.getsize(state.current_file)
            use_streaming = file_size > 50 * 1024 * 1024  # 50MB
            
            if use_streaming:
                status.text("🔄 Processing large file with streaming...")
                # Stream process large files
                processed_chunks = []
                chunk_count = 0
                
                async for chunk in self.audio_processor.process_audio_file_streaming(
                    state.current_file,
                    chunk_size=1024*1024,  # 1MB chunks
                    settings=state.settings,
                    overlap=2048
                ):
                    processed_chunks.append(chunk)
                    chunk_count += 1
                    progress_bar.progress(min(chunk_count * 0.1, 0.9))
                
                # Combine chunks
                enhanced_audio = np.concatenate(processed_chunks, axis=1)
                sample_rate = self.audio_processor.target_sample_rate
                
            else:
                # Process normally for smaller files
                result = await self.audio_processor.process_audio_file(
                    state.current_file,
                    settings=state.settings,
                    progress_callback=lambda p: progress_bar.progress(p * 0.9)
                )
                enhanced_audio = result.enhanced_audio
                sample_rate = result.sample_rate
            
            # Calculate and display professional metrics
            metrics = self.audio_processor.calculate_professional_metrics(
                enhanced_audio,
                sample_rate
            )
            
            # Save enhanced audio
            enhanced_path = save_audio_file(
                enhanced_audio,
                sample_rate,
                f"enhanced_{Path(state.current_file).name}"
            )
            
            state.enhanced_file = enhanced_path
            state.audio_metrics = metrics
            
            # Cache result
            if cache_key:
                cache_result = {
                    'enhanced_file': enhanced_path,
                    'metrics': metrics
                }
                cache_manager.cache_result(cache_key, cache_result, "audio")
            
            # Display metrics
            self._display_professional_metrics(metrics)
            
            progress_bar.progress(1.0)
            status.success("✅ Audio enhancement complete!")
            
        except Exception as e:
            status.error(f"❌ Error: {str(e)}")
            logging.error(f"Enhancement error: {e}")
    
    def _display_professional_metrics(self, metrics: AudioMetrics):
        """Display professional audio metrics"""
        st.subheader("📊 Professional Audio Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "LUFS Integrated", 
                f"{metrics.lufs_integrated:.1f}",
                help="Integrated loudness (EBU R128)"
            )
        
        with col2:
            st.metric(
                "True Peak", 
                f"{metrics.true_peak:.1f} dBTP",
                delta=f"{metrics.true_peak - self.audio_processor.max_true_peak:.1f}",
                help="True peak level with oversampling"
            )
        
        with col3:
            st.metric(
                "Dynamic Range", 
                f"{metrics.dynamic_range:.1f} dB",
                help="Difference between loud and quiet parts"
            )
        
        with col4:
            st.metric(
                "Phase Coherence", 
                f"{metrics.phase_coherence:.2%}",
                help="Phase stability (higher is better)"
            )
        
        # Additional metrics in expandable section
        with st.expander("🔍 Detailed Metrics"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Peak Amplitude", f"{metrics.peak_amplitude:.3f}")
                st.metric("RMS Level", f"{metrics.rms_level:.3f}")
            
            with col2:
                st.metric("Crest Factor", f"{metrics.crest_factor:.1f} dB")
                st.metric("Stereo Width", f"{metrics.stereo_width:.2f}")
            
            with col3:
                st.metric("Clip Count", f"{metrics.clip_count}")
                st.metric("THD+N", f"{metrics.thd_plus_n:.3%}")
        
        if metrics.clip_count > 0:
            st.warning(f"⚠️ {metrics.clip_count} clipped samples detected")
    
    async def _transcribe_audio(self, state):
        """Transcribe audio with progress tracking"""
        progress_bar = st.progress(0)
        status = st.empty()
        
        try:
            # Use enhanced audio if available
            audio_file = state.enhanced_file or state.current_file
            
            status.text("🎤 Transcribing audio...")
            progress_bar.progress(0.3)
            
            # Get transcription
            result = await self.transcription_handler.transcribe(
                audio_file,
                enhance=state.settings.get('enhance_transcription', True)
            )
            
            progress_bar.progress(0.7)
            
            # Enhance with Claude if enabled
            if state.settings.get('enhance_transcription', True):
                status.text("✨ Enhancing transcription with AI...")
                result['enhanced_text'] = await self.claude_service.enhance_transcription(
                    result['text']
                )
            
            state.transcription = result
            progress_bar.progress(1.0)
            status.success("✅ Transcription complete!")
            
        except Exception as e:
            status.error(f"❌ Error: {str(e)}")
            logging.error(f"Transcription error: {e}")
    
    async def _transcribe_with_speakers(self, state):
        """Transcribe audio with speaker diarization"""
        progress_bar = st.progress(0)
        status = st.empty()
        
        try:
            # Use enhanced audio if available
            audio_file = state.enhanced_file or state.current_file
            
            status.text("🎤 Transcribing with speaker diarization...")
            progress_bar.progress(0.2)
            
            status.text("👥 Analyzing speakers with PyAnnote...")
            progress_bar.progress(0.5)
            
            # Get diarized transcription
            result = await self.transcription_handler.transcribe_with_speaker_diarization(
                audio_file
            )
            
            progress_bar.progress(0.8)
            
            # Enhance with Claude if enabled
            if state.settings.get('enhance_transcription', True) and 'enhanced_text' not in result:
                status.text("✨ Enhancing transcription with AI...")
                result['enhanced_text'] = await self.claude_service.enhance_transcription(
                    result['text']
                )
            
            state.transcription = result
            progress_bar.progress(1.0)
            
            # Show success with speaker count
            speaker_count = result.get('speaker_analysis', {}).get('total_speakers', 0)
            status.success(f"✅ Transcription complete! Detected {speaker_count} speaker(s)")
            
        except Exception as e:
            status.error(f"❌ Error: {str(e)}")
            logging.error(f"Speaker diarization error: {e}")
    
    def _render_speaker_analysis(self, speaker_analysis: Dict):
        """Render speaker analysis results"""
        st.subheader("👥 Speaker Analysis")
        
        # Overview metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Speakers", 
                speaker_analysis.get('total_speakers', 0)
            )
        
        with col2:
            gender_dist = speaker_analysis.get('gender_distribution', {})
            male_count = gender_dist.get('male', 0)
            female_count = gender_dist.get('female', 0)
            st.metric("Male Speakers", male_count)
        
        with col3:
            st.metric("Female Speakers", female_count)
        
        with col4:
            confidence = speaker_analysis.get('diarization_confidence', 0)
            st.metric("Confidence", f"{confidence:.2%}")
        
        # Speaker details
        speakers = speaker_analysis.get('speakers', {})
        if speakers:
            st.subheader("🎯 Speaker Details")
            
            for speaker_id, speaker_data in speakers.items():
                with st.expander(f"🎤 {speaker_id} ({speaker_data.get('gender', 'unknown').title()})"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Total Duration", f"{speaker_data.get('total_duration', 0):.1f}s")
                        st.metric("Segments", len(speaker_data.get('segments', [])))
                        st.metric("Gender Confidence", f"{speaker_data.get('confidence', 0):.2%}")
                    
                    with col2:
                        # Voice characteristics
                        voice_chars = speaker_data.get('voice_characteristics', {})
                        pitch_stats = voice_chars.get('pitch_statistics', {})
                        
                        if pitch_stats:
                            st.metric("Mean Pitch", f"{pitch_stats.get('mean', 0):.1f} Hz")
                            st.metric("Pitch Range", f"{pitch_stats.get('min', 0):.0f}-{pitch_stats.get('max', 0):.0f} Hz")
                    
                    # Show segments timeline
                    segments = speaker_data.get('segments', [])
                    if segments:
                        st.write("**Speaking Segments:**")
                        for i, segment in enumerate(segments[:5]):  # Show first 5 segments
                            st.write(f"• {segment['start']:.1f}s - {segment['end']:.1f}s ({segment['duration']:.1f}s)")
                        
                        if len(segments) > 5:
                            st.write(f"... and {len(segments) - 5} more segments")
        
        # Method and device info
        st.info(f"Analysis method: {speaker_analysis.get('method', 'unknown')} on {speaker_analysis.get('device_used', 'unknown')}")
    
    async def _get_ai_response(self, state, prompt: str) -> str:
        """Get AI response based on context"""
        try:
            # Build context
            context = self._build_context(state)
            
            # Get response from Claude
            response = await self.claude_service.get_contextual_response(
                prompt=prompt,
                context=context
            )
            
            return response
            
        except Exception as e:
            logging.error(f"AI response error: {e}")
            return f"Sorry, I encountered an error: {str(e)}"
    
    def _build_context(self, state) -> str:
        """Build context for AI assistant"""
        context_parts = []
        
        # Add audio metadata
        if state.current_file:
            metadata = get_audio_metadata(state.current_file)
            context_parts.append(f"Audio file: {Path(state.current_file).name}")
            context_parts.append(f"Duration: {metadata['duration']:.2f} seconds")
            context_parts.append(f"Sample rate: {metadata['sample_rate']} Hz")
        
        # Add transcription if available
        if state.transcription:
            text = state.transcription.get('enhanced_text', state.transcription.get('text', ''))
            context_parts.append(f"Transcription: {text[:500]}...")
        
        # Add enhancement info
        if state.enhanced_file:
            context_parts.append("Audio has been enhanced")
        
        return "\n".join(context_parts)
    
    def _export_transcription(self, state, format: str):
        """Export transcription in various formats"""
        try:
            transcription = state.transcription
            filename = f"transcription_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            if format == 'txt':
                content = transcription.get('enhanced_text', transcription['text'])
                st.download_button(
                    label="Download TXT",
                    data=content,
                    file_name=f"{filename}.txt",
                    mime="text/plain"
                )
            
            elif format == 'json':
                content = json.dumps(transcription, indent=2)
                st.download_button(
                    label="Download JSON",
                    data=content,
                    file_name=f"{filename}.json",
                    mime="application/json"
                )
            
            elif format == 'srt':
                # Convert to SRT format (simplified)
                srt_content = self._convert_to_srt(transcription)
                st.download_button(
                    label="Download SRT",
                    data=srt_content,
                    file_name=f"{filename}.srt",
                    mime="text/plain"
                )
                
        except Exception as e:
            st.error(f"Export error: {str(e)}")
    
    def _convert_to_srt(self, transcription) -> str:
        """Convert transcription to SRT format"""
        # Simplified SRT conversion
        lines = []
        segments = transcription.get('segments', [])
        
        for i, segment in enumerate(segments):
            lines.append(str(i + 1))
            start = self._format_timestamp(segment.get('start', 0))
            end = self._format_timestamp(segment.get('end', 0))
            lines.append(f"{start} --> {end}")
            lines.append(segment.get('text', '').strip())
            lines.append("")
        
        return "\n".join(lines)
    
    def _format_timestamp(self, seconds: float) -> str:
        """Format seconds to SRT timestamp"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
    
    def _render_welcome_screen(self):
        """Render welcome screen when no file is loaded"""
        st.markdown("""
        <div style="text-align: center; padding: 3rem;">
            <h2>Welcome to Audio Analyzer! 👋</h2>
            <p style="font-size: 1.2rem; color: #666; margin: 2rem 0;">
                Upload an audio file to get started with analysis, enhancement, and transcription.
            </p>
            <div style="display: flex; justify-content: center; gap: 2rem; margin-top: 3rem;">
                <div style="text-align: center;">
                    <h3>🎵</h3>
                    <p><strong>Analyze</strong><br/>Visualize waveforms and spectrograms</p>
                </div>
                <div style="text-align: center;">
                    <h3>✨</h3>
                    <p><strong>Enhance</strong><br/>Reduce noise and improve quality</p>
                </div>
                <div style="text-align: center;">
                    <h3>📝</h3>
                    <p><strong>Transcribe</strong><br/>Convert speech to text with AI</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# Main execution
if __name__ == "__main__":
    app = AudioAnalyzerApp()
    asyncio.run(app.run()) 