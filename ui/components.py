import streamlit as st
import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
from typing import Optional, Dict, List
import logging

logger = logging.getLogger(__name__)

def render_audio_player(file_path: str, key: Optional[str] = None):
    """Render audio player with controls"""
    try:
        # Add custom styling for audio player
        st.markdown("""
        <style>
            audio {
                width: 100%;
                height: 40px;
            }
        </style>
        """, unsafe_allow_html=True)
        
        # Display audio player
        with open(file_path, 'rb') as audio_file:
            audio_bytes = audio_file.read()
            st.audio(audio_bytes, format='audio/wav', start_time=0)
            
    except Exception as e:
        st.error(f"Error loading audio: {str(e)}")
        logger.error(f"Audio player error: {e}")

def render_waveform(file_path: str, figsize: tuple = (12, 4), max_points: int = 10000):
    """Render waveform visualization with data sampling to prevent large transfers"""
    try:
        # Load audio
        y, sr = librosa.load(file_path, sr=None)
        duration = len(y) / sr
        
        # Sample data if too large to prevent Streamlit message size limit
        if len(y) > max_points:
            # Use decimation to reduce data points while preserving shape
            step = len(y) // max_points
            y_sampled = y[::step]
            time_sampled = np.linspace(0, duration, len(y_sampled))
            st.info(f"📊 Displaying sampled waveform ({len(y_sampled):,} of {len(y):,} points) to optimize performance")
        else:
            y_sampled = y
            time_sampled = np.linspace(0, duration, len(y))
        
        # Create plotly figure
        fig = go.Figure()
        
        # Add waveform trace
        fig.add_trace(go.Scatter(
            x=time_sampled,
            y=y_sampled,
            mode='lines',
            name='Waveform',
            line=dict(color='#3b82f6', width=1),
            hovertemplate='Time: %{x:.2f}s<br>Amplitude: %{y:.3f}<extra></extra>'
        ))
        
        # Update layout
        fig.update_layout(
            title=dict(text="Audio Waveform", font=dict(size=16)),
            xaxis_title="Time (seconds)",
            yaxis_title="Amplitude",
            height=300,
            margin=dict(l=0, r=0, t=40, b=40),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            hovermode='x unified',
            showlegend=False
        )
        
        # Add grid
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
        
        sanitized_path = file_path.replace('/', '_').replace('\\', '_').replace('\\', '_')
        st.plotly_chart(fig, use_container_width=True, key=f"waveform_{sanitized_path}")
        
    except Exception as e:
        st.error(f"Error rendering waveform: {str(e)}")
        logger.error(f"Waveform error: {e}")

def render_spectrogram(file_path: str, figsize: tuple = (12, 4), max_duration: float = 60.0):
    """Render spectrogram visualization with duration limiting to prevent large transfers"""
    try:
        # Load audio with duration limit to prevent memory issues
        y, sr = librosa.load(file_path, sr=None, duration=max_duration)
        
        if len(y) / sr > max_duration:
            st.info(f"📊 Displaying first {max_duration} seconds of audio for spectrogram to optimize performance")
        
        # Compute spectrogram with reduced resolution for large files
        hop_length = 512 if len(y) < sr * 30 else 1024  # Reduce resolution for long files
        D = librosa.stft(y, hop_length=hop_length)
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Display spectrogram
        img = librosa.display.specshow(
            S_db,
            x_axis='time',
            y_axis='hz',
            sr=sr,
            ax=ax,
            cmap='viridis'
        )
        
        # Add colorbar
        fig.colorbar(img, ax=ax, format='%+2.0f dB')
        
        # Set title and labels
        ax.set_title('Spectrogram', fontsize=16)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency (Hz)')
        
        # Adjust layout
        plt.tight_layout()
        
        # Display in Streamlit
        st.pyplot(fig)
        plt.close()
        
    except Exception as e:
        st.error(f"Error rendering spectrogram: {str(e)}")
        logger.error(f"Spectrogram error: {e}")

def render_enhancement_controls(state):
    """Render audio enhancement controls"""
    st.subheader("🎛️ Enhancement Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Noise Reduction**")
        state.settings['noise_reduction'] = st.checkbox(
            "Enable noise reduction",
            value=state.settings.get('noise_reduction', True),
            help="Remove background noise from audio"
        )
        
        if state.settings['noise_reduction']:
            state.settings['noise_reduction_strength'] = st.slider(
                "Noise reduction strength",
                min_value=0.0,
                max_value=2.0,
                value=1.0,
                step=0.1,
                help="Higher values remove more noise but may affect audio quality"
            )
    
    with col2:
        st.markdown("**Audio Normalization**")
        state.settings['normalize'] = st.checkbox(
            "Enable normalization",
            value=state.settings.get('normalize', True),
            help="Adjust audio levels for consistent volume"
        )
        
        if state.settings['normalize']:
            state.settings['target_db'] = st.slider(
                "Target level (dB)",
                min_value=-30.0,
                max_value=-10.0,
                value=-20.0,
                step=1.0,
                help="Target loudness level in decibels"
            )
    
    # Advanced options in expander
    with st.expander("🔧 Advanced Options"):
        col1, col2 = st.columns(2)
        
        with col1:
            state.settings['band_pass'] = st.checkbox(
                "Band-pass filter",
                value=state.settings.get('band_pass', True),
                help="Focus on speech frequencies (300-3400 Hz)"
            )
            
            state.settings['compression'] = st.checkbox(
                "Dynamic range compression",
                value=state.settings.get('compression', False),
                help="Reduce volume differences"
            )
        
        with col2:
            if state.settings['compression']:
                state.settings['compression_ratio'] = st.slider(
                    "Compression ratio",
                    min_value=2.0,
                    max_value=10.0,
                    value=4.0,
                    step=0.5
                )

def render_transcription_view(transcription: Dict):
    """Render transcription results with speaker diarization support"""
    # Check if this is a diarized transcription
    has_speakers = 'diarized_segments' in transcription and transcription['diarized_segments']
    
    # Show enhanced transcription if available
    if 'enhanced_text' in transcription:
        st.subheader("✨ Enhanced Transcription")
        
        # Display in a nice container
        st.markdown(
            f"""
            <div style="
                background-color: #f0f9ff;
                border-left: 4px solid #3b82f6;
                padding: 1.5rem;
                border-radius: 0.5rem;
                margin: 1rem 0;
            ">
                {transcription['enhanced_text']}
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Show original in expander
        with st.expander("📄 Show Original Transcription"):
            st.text(transcription.get('text', ''))
    else:
        # Show raw transcription
        st.subheader("📝 Transcription")
        st.text_area(
            label="",
            value=transcription.get('text', ''),
            height=300,
            disabled=True
        )
    
    # Show metadata if available
    if 'confidence' in transcription:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Confidence", f"{transcription['confidence']:.2%}")
        with col2:
            if 'duration' in transcription:
                st.metric("Duration", f"{transcription['duration']:.1f}s")
        with col3:
            if 'word_count' in transcription:
                st.metric("Word Count", transcription['word_count'])
    
    # Show diarized segments if available
    if has_speakers:
        st.subheader("👥 Speaker-Separated Transcription")
        
        # Create speaker color mapping
        speakers = set(seg.get('speaker_id', 'UNKNOWN') for seg in transcription['diarized_segments'])
        speaker_colors = {
            speaker: color for speaker, color in zip(
                speakers, 
                ['#3b82f6', '#ef4444', '#10b981', '#f59e0b', '#8b5cf6', '#06b6d4']
            )
        }
        
        # Display segments with speaker colors
        for i, segment in enumerate(transcription['diarized_segments']):
            speaker_id = segment.get('speaker_id', 'UNKNOWN')
            speaker_gender = segment.get('speaker_gender', 'unknown')
            speaker_confidence = segment.get('speaker_confidence', 0)
            text = segment.get('text', '')
            start_time = segment.get('start', 0)
            end_time = segment.get('end', 0)
            
            # Get speaker color
            color = speaker_colors.get(speaker_id, '#6b7280')
            
            # Create speaker badge
            gender_emoji = "👨" if speaker_gender == 'male' else "👩" if speaker_gender == 'female' else "👤"
            
            st.markdown(
                f"""
                <div style="
                    background-color: {color}15;
                    border-left: 4px solid {color};
                    padding: 1rem;
                    border-radius: 0.5rem;
                    margin: 0.5rem 0;
                ">
                    <div style="
                        display: flex;
                        justify-content: space-between;
                        align-items: center;
                        margin-bottom: 0.5rem;
                    ">
                        <span style="
                            background-color: {color};
                            color: white;
                            padding: 0.25rem 0.75rem;
                            border-radius: 1rem;
                            font-size: 0.875rem;
                            font-weight: 600;
                        ">
                            {gender_emoji} {speaker_id}
                        </span>
                        <span style="
                            color: #6b7280;
                            font-size: 0.875rem;
                        ">
                            {start_time:.1f}s - {end_time:.1f}s
                            {f" (confidence: {speaker_confidence:.1%})" if speaker_confidence > 0 else ""}
                        </span>
                    </div>
                    <div style="font-size: 1rem; line-height: 1.5;">
                        {text}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        # Show speaker legend
        with st.expander("🎨 Speaker Legend"):
            for speaker_id in speakers:
                color = speaker_colors.get(speaker_id, '#6b7280')
                st.markdown(
                    f"""
                    <div style="display: flex; align-items: center; margin: 0.25rem 0;">
                        <div style="
                            width: 20px;
                            height: 20px;
                            background-color: {color};
                            border-radius: 50%;
                            margin-right: 0.5rem;
                        "></div>
                        <span>{speaker_id}</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
    
    # Show regular segments if no diarization
    elif 'segments' in transcription and transcription['segments']:
        with st.expander("🎯 Show Timestamped Segments"):
            for i, segment in enumerate(transcription['segments']):
                st.markdown(
                    f"**[{segment['start']:.1f}s - {segment['end']:.1f}s]** "
                    f"{segment['text']}"
                )
                if i < len(transcription['segments']) - 1:
                    st.divider()

def render_progress_bar(progress: float, text: str = ""):
    """Render a custom progress bar"""
    progress_percentage = int(progress * 100)
    
    st.markdown(
        f"""
        <div style="margin: 1rem 0;">
            <p style="margin-bottom: 0.5rem;">{text}</p>
            <div style="
                background-color: #e5e7eb;
                border-radius: 0.5rem;
                height: 0.5rem;
                overflow: hidden;
            ">
                <div style="
                    background-color: #3b82f6;
                    height: 100%;
                    width: {progress_percentage}%;
                    transition: width 0.3s ease;
                "></div>
            </div>
            <p style="text-align: right; margin-top: 0.25rem; font-size: 0.875rem;">
                {progress_percentage}%
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

def render_error_message(error: str, details: Optional[str] = None):
    """Render a styled error message"""
    st.markdown(
        f"""
        <div class="error-box">
            <strong>❌ Error:</strong> {error}
            {f'<br><small>{details}</small>' if details else ''}
        </div>
        """,
        unsafe_allow_html=True
    )

def render_success_message(message: str):
    """Render a styled success message"""
    st.markdown(
        f"""
        <div class="success-box">
            <strong>✅ Success:</strong> {message}
        </div>
        """,
        unsafe_allow_html=True
    )

def render_info_message(message: str):
    """Render a styled info message"""
    st.markdown(
        f"""
        <div class="info-box">
            <strong>ℹ️ Info:</strong> {message}
        </div>
        """,
        unsafe_allow_html=True
    ) 