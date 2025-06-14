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

def render_waveform(file_path: str, figsize: tuple = (12, 4)):
    """Render waveform visualization"""
    try:
        # Load audio
        y, sr = librosa.load(file_path, sr=None)
        duration = len(y) / sr
        
        # Create plotly figure
        time = np.linspace(0, duration, len(y))
        
        fig = go.Figure()
        
        # Add waveform trace
        fig.add_trace(go.Scatter(
            x=time,
            y=y,
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
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error rendering waveform: {str(e)}")
        logger.error(f"Waveform error: {e}")

def render_spectrogram(file_path: str, figsize: tuple = (12, 4)):
    """Render spectrogram visualization"""
    try:
        # Load audio
        y, sr = librosa.load(file_path, sr=None)
        
        # Compute spectrogram
        D = librosa.stft(y)
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
    """Render transcription results"""
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
    
    # Show segments if available
    if 'segments' in transcription and transcription['segments']:
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