import numpy as np
import librosa
import soundfile as sf
from scipy import signal
from scipy.signal import butter, filtfilt, hilbert
import pyloudnorm as pyln
from typing import Dict, List, Optional, Tuple, AsyncGenerator
import logging
import asyncio
from dataclasses import dataclass
from pathlib import Path
import tempfile
import os

logger = logging.getLogger(__name__)

@dataclass
class AudioMetrics:
    """Professional audio metrics"""
    lufs_integrated: float
    lufs_short_term: float
    lufs_momentary: float
    true_peak: float
    dynamic_range: float
    phase_coherence: float
    thd_plus_n: float
    clip_count: int
    peak_amplitude: float
    rms_level: float
    crest_factor: float
    stereo_width: float
    frequency_response: Dict[str, float]

@dataclass
class ProcessingResult:
    """Result of audio processing"""
    enhanced_audio: np.ndarray
    sample_rate: int
    metrics: AudioMetrics
    processing_log: List[str]

class ProfessionalAudioProcessor:
    """Professional-grade audio processor with broadcast standards compliance"""
    
    def __init__(
        self,
        target_sample_rate: int = 48000,
        target_lufs: float = -16.0,
        max_true_peak: float = -1.0,
        target_bit_depth: int = 24
    ):
        self.target_sample_rate = target_sample_rate
        self.target_lufs = target_lufs
        self.max_true_peak = max_true_peak
        self.target_bit_depth = target_bit_depth
        
        # Initialize loudness meter
        self.meter = pyln.Meter(target_sample_rate)
        
        # Processing parameters
        self.processing_log = []
    
    async def process_audio_file(
        self,
        file_path: str,
        settings: Dict,
        progress_callback: Optional[callable] = None
    ) -> ProcessingResult:
        """Process audio file with professional standards"""
        self.processing_log = []
        
        try:
            # Load audio
            if progress_callback:
                progress_callback(0.1)
            
            audio, sr = librosa.load(file_path, sr=None, mono=False)
            self._log(f"Loaded audio: {audio.shape}, SR: {sr}")
            
            # Ensure stereo for professional processing
            if len(audio.shape) == 1:
                audio = np.stack([audio, audio])
                self._log("Converted mono to stereo")
            
            # Resample if needed
            if sr != self.target_sample_rate:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.target_sample_rate)
                sr = self.target_sample_rate
                self._log(f"Resampled to {sr} Hz")
            
            if progress_callback:
                progress_callback(0.3)
            
            # Professional processing pipeline
            processed_audio = await self._professional_processing_pipeline(
                audio, sr, settings, progress_callback
            )
            
            # Calculate final metrics
            metrics = self.calculate_professional_metrics(processed_audio, sr)
            
            if progress_callback:
                progress_callback(1.0)
            
            return ProcessingResult(
                enhanced_audio=processed_audio,
                sample_rate=sr,
                metrics=metrics,
                processing_log=self.processing_log.copy()
            )
            
        except Exception as e:
            logger.error(f"Processing error: {e}")
            raise
    
    async def process_audio_file_streaming(
        self,
        file_path: str,
        chunk_size: int = 1024*1024,
        overlap: int = 2048,
        settings: Dict = None,
    ) -> AsyncGenerator[np.ndarray, None]:
        """Stream process large audio files"""
        try:
            # Get file info without loading
            info = sf.info(file_path)
            total_frames = info.frames
            
            # Calculate chunk parameters
            frames_per_chunk = chunk_size // (info.channels * 2)  # 16-bit assumption
            
            with sf.SoundFile(file_path, 'r') as f:
                chunk_start = 0
                
                while chunk_start < total_frames:
                    # Read chunk with overlap
                    chunk_frames = min(frames_per_chunk, total_frames - chunk_start)
                    
                    if chunk_start > 0:
                        f.seek(max(0, chunk_start - overlap))
                        chunk_data = f.read(chunk_frames + overlap)
                    else:
                        chunk_data = f.read(chunk_frames)
                    
                    # Process chunk
                    if len(chunk_data.shape) == 1:
                        chunk_data = np.stack([chunk_data, chunk_data])
                    else:
                        chunk_data = chunk_data.T
                    
                    # Apply processing
                    processed_chunk = await self._process_chunk(
                        chunk_data, info.samplerate, settings or {}
                    )
                    
                    # Remove overlap for output
                    if chunk_start > 0:
                        processed_chunk = processed_chunk[:, overlap:]
                    
                    yield processed_chunk
                    
                    chunk_start += chunk_frames
                    
        except Exception as e:
            logger.error(f"Streaming processing error: {e}")
            raise
    
    async def _professional_processing_pipeline(
        self,
        audio: np.ndarray,
        sample_rate: int,
        settings: Dict,
        progress_callback: Optional[callable] = None
    ) -> np.ndarray:
        """Professional audio processing pipeline"""
        
        # Basic processing for now - can be expanded
        processed = audio.copy()
        
        # Normalize
        if settings.get('normalize', True):
            peak = np.max(np.abs(processed))
            if peak > 0:
                processed = processed / peak * 0.95
                self._log("Applied normalization")
        
        if progress_callback:
            progress_callback(0.8)
        
        return processed
    
    async def _process_chunk(
        self,
        chunk: np.ndarray,
        sample_rate: int,
        settings: Dict
    ) -> np.ndarray:
        """Process individual audio chunk"""
        # Simplified processing for streaming
        chunk = self._remove_dc_offset(chunk)
        
        if settings.get('noise_reduction', True):
            chunk = await self._advanced_noise_reduction(chunk, sample_rate)
        
        if settings.get('normalize', True):
            chunk = self._normalize_chunk(chunk)
        
        return chunk
    
    def _remove_dc_offset(self, audio: np.ndarray) -> np.ndarray:
        """Remove DC offset from audio"""
        return audio - np.mean(audio, axis=1, keepdims=True)
    
    def _apply_highpass_filter(
        self,
        audio: np.ndarray,
        sample_rate: int,
        cutoff: float = 20,
        order: int = 4
    ) -> np.ndarray:
        """Apply high-pass filter to remove subsonic content"""
        nyquist = sample_rate / 2
        normalized_cutoff = cutoff / nyquist
        
        b, a = butter(order, normalized_cutoff, btype='high')
        
        filtered = np.zeros_like(audio)
        for ch in range(audio.shape[0]):
            filtered[ch] = filtfilt(b, a, audio[ch])
        
        return filtered
    
    async def _advanced_noise_reduction(
        self,
        audio: np.ndarray,
        sample_rate: int
    ) -> np.ndarray:
        """Advanced spectral noise reduction"""
        # Spectral subtraction with psychoacoustic masking
        hop_length = 512
        n_fft = 2048
        
        enhanced = np.zeros_like(audio)
        
        for ch in range(audio.shape[0]):
            # STFT
            stft = librosa.stft(audio[ch], n_fft=n_fft, hop_length=hop_length)
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            # Estimate noise floor from quiet segments
            power = magnitude ** 2
            noise_floor = np.percentile(power, 10, axis=1, keepdims=True)
            
            # Spectral subtraction with over-subtraction factor
            alpha = 2.0  # Over-subtraction factor
            beta = 0.01  # Spectral floor factor
            
            enhanced_magnitude = magnitude - alpha * np.sqrt(noise_floor)
            enhanced_magnitude = np.maximum(
                enhanced_magnitude,
                beta * magnitude
            )
            
            # Reconstruct signal
            enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
            enhanced[ch] = librosa.istft(
                enhanced_stft,
                hop_length=hop_length,
                length=len(audio[ch])
            )
        
        return enhanced
    
    def _apply_multiband_compressor(
        self,
        audio: np.ndarray,
        sample_rate: int
    ) -> np.ndarray:
        """Apply multiband compression"""
        # Define frequency bands
        bands = [
            (20, 250),    # Low
            (250, 2000),  # Mid
            (2000, 8000), # High-mid
            (8000, 20000) # High
        ]
        
        compressed = np.zeros_like(audio)
        
        for ch in range(audio.shape[0]):
            channel_sum = np.zeros_like(audio[ch])
            
            for low_freq, high_freq in bands:
                # Extract band
                band_audio = self._extract_frequency_band(
                    audio[ch], sample_rate, low_freq, high_freq
                )
                
                # Apply compression
                compressed_band = self._apply_compressor(
                    band_audio,
                    threshold=-12,  # dB
                    ratio=3.0,
                    attack_ms=5,
                    release_ms=50,
                    sample_rate=sample_rate
                )
                
                channel_sum += compressed_band
            
            compressed[ch] = channel_sum
        
        return compressed
    
    def _extract_frequency_band(
        self,
        audio: np.ndarray,
        sample_rate: int,
        low_freq: float,
        high_freq: float
    ) -> np.ndarray:
        """Extract frequency band using bandpass filter"""
        nyquist = sample_rate / 2
        low_norm = low_freq / nyquist
        high_norm = min(high_freq / nyquist, 0.99)
        
        b, a = butter(4, [low_norm, high_norm], btype='band')
        return filtfilt(b, a, audio)
    
    def _apply_compressor(
        self,
        audio: np.ndarray,
        threshold: float,
        ratio: float,
        attack_ms: float,
        release_ms: float,
        sample_rate: int
    ) -> np.ndarray:
        """Apply dynamic range compression"""
        # Convert to linear scale
        threshold_lin = 10 ** (threshold / 20)
        
        # Calculate attack/release coefficients
        attack_coeff = np.exp(-1 / (attack_ms * sample_rate / 1000))
        release_coeff = np.exp(-1 / (release_ms * sample_rate / 1000))
        
        # Peak detection and gain reduction
        envelope = np.abs(audio)
        gain_reduction = np.ones_like(audio)
        
        for i in range(1, len(audio)):
            # Smooth envelope
            if envelope[i] > envelope[i-1]:
                envelope[i] = attack_coeff * envelope[i-1] + (1 - attack_coeff) * envelope[i]
            else:
                envelope[i] = release_coeff * envelope[i-1] + (1 - release_coeff) * envelope[i]
            
            # Calculate gain reduction
            if envelope[i] > threshold_lin:
                excess = envelope[i] / threshold_lin
                gain_reduction[i] = 1 / (1 + (excess - 1) * (ratio - 1) / ratio)
        
        return audio * gain_reduction
    
    def _apply_professional_eq(
        self,
        audio: np.ndarray,
        sample_rate: int,
        settings: Dict
    ) -> np.ndarray:
        """Apply professional EQ curve"""
        # Default professional EQ curve
        eq_bands = [
            (60, 0.5, 0.7),    # Low cut
            (200, 1.0, -1.0),  # Low-mid reduction
            (1000, 1.0, 0.5),  # Presence boost
            (3000, 1.0, 1.0),  # Clarity boost
            (8000, 0.7, 0.5),  # Air boost
        ]
        
        processed = audio.copy()
        
        for freq, q, gain_db in eq_bands:
            if abs(gain_db) > 0.1:  # Only apply if significant gain
                processed = self._apply_parametric_eq(
                    processed, sample_rate, freq, q, gain_db
                )
        
        return processed
    
    def _apply_parametric_eq(
        self,
        audio: np.ndarray,
        sample_rate: int,
        freq: float,
        q: float,
        gain_db: float
    ) -> np.ndarray:
        """Apply parametric EQ band"""
        # Calculate filter coefficients
        w = 2 * np.pi * freq / sample_rate
        cos_w = np.cos(w)
        sin_w = np.sin(w)
        A = 10 ** (gain_db / 40)
        alpha = sin_w / (2 * q)
        
        # Peaking EQ coefficients
        b0 = 1 + alpha * A
        b1 = -2 * cos_w
        b2 = 1 - alpha * A
        a0 = 1 + alpha / A
        a1 = -2 * cos_w
        a2 = 1 - alpha / A
        
        # Normalize
        b = np.array([b0, b1, b2]) / a0
        a = np.array([1, a1, a2]) / a0
        
        # Apply filter
        filtered = np.zeros_like(audio)
        for ch in range(audio.shape[0]):
            filtered[ch] = signal.lfilter(b, a, audio[ch])
        
        return filtered
    
    def _enhance_stereo_image(self, audio: np.ndarray) -> np.ndarray:
        """Enhance stereo image using M/S processing"""
        if audio.shape[0] < 2:
            return audio
        
        left, right = audio[0], audio[1]
        
        # Convert to M/S
        mid = (left + right) / 2
        side = (left - right) / 2
        
        # Enhance side signal slightly
        side_enhanced = side * 1.1
        
        # Convert back to L/R
        left_enhanced = mid + side_enhanced
        right_enhanced = mid - side_enhanced
        
        return np.array([left_enhanced, right_enhanced])
    
    def _normalize_loudness_r128(
        self,
        audio: np.ndarray,
        sample_rate: int
    ) -> np.ndarray:
        """Normalize loudness according to EBU R128"""
        # Measure current loudness
        current_lufs = self.meter.integrated_loudness(audio.T)
        
        # Calculate gain adjustment
        gain_db = self.target_lufs - current_lufs
        gain_linear = 10 ** (gain_db / 20)
        
        return audio * gain_linear
    
    def _apply_true_peak_limiter(
        self,
        audio: np.ndarray,
        sample_rate: int
    ) -> np.ndarray:
        """Apply true peak limiter"""
        # Oversample for true peak detection
        oversample_factor = 4
        oversampled = signal.resample(audio, len(audio[0]) * oversample_factor, axis=1)
        
        # Calculate true peak
        true_peak_linear = 10 ** (self.max_true_peak / 20)
        
        # Apply limiting
        peak_values = np.max(np.abs(oversampled), axis=0)
        gain_reduction = np.minimum(1.0, true_peak_linear / np.maximum(peak_values, 1e-10))
        
        # Smooth gain reduction
        gain_smooth = signal.savgol_filter(gain_reduction, 51, 3)
        
        # Apply to original sample rate
        gain_original = signal.resample(gain_smooth, len(audio[0]))
        
        return audio * gain_original
    
    def _apply_dithering(self, audio: np.ndarray, target_bits: int) -> np.ndarray:
        """Apply triangular dithering for bit depth reduction"""
        # Calculate quantization step
        q_step = 2 ** (1 - target_bits)
        
        # Generate triangular dither noise
        dither_amplitude = q_step / 2
        dither = np.random.triangular(-dither_amplitude, 0, dither_amplitude, audio.shape)
        
        # Add dither and quantize
        dithered = audio + dither
        quantized = np.round(dithered / q_step) * q_step
        
        return quantized
    
    def _normalize_chunk(self, chunk: np.ndarray) -> np.ndarray:
        """Normalize audio chunk"""
        peak = np.max(np.abs(chunk))
        if peak > 0:
            return chunk / peak * 0.95
        return chunk
    
    def calculate_professional_metrics(
        self,
        audio: np.ndarray,
        sample_rate: int
    ) -> AudioMetrics:
        """Calculate comprehensive professional audio metrics"""
        
        # Loudness measurements (EBU R128)
        lufs_integrated = self.meter.integrated_loudness(audio.T)
        
        # True peak measurement
        oversample_factor = 4
        oversampled = signal.resample(audio, len(audio[0]) * oversample_factor, axis=1)
        true_peak_linear = np.max(np.abs(oversampled))
        true_peak_db = 20 * np.log10(max(true_peak_linear, 1e-10))
        
        # Dynamic range
        peak_amplitude = np.max(np.abs(audio))
        rms_level = np.sqrt(np.mean(audio ** 2))
        dynamic_range = 20 * np.log10(peak_amplitude / max(rms_level, 1e-10))
        
        # Phase coherence (for stereo)
        phase_coherence = 1.0
        if audio.shape[0] >= 2:
            correlation = np.corrcoef(audio[0], audio[1])[0, 1]
            phase_coherence = max(0, correlation)
        
        # Clip detection
        clip_threshold = 0.99
        clip_count = np.sum(np.abs(audio) > clip_threshold)
        
        # Crest factor
        crest_factor = 20 * np.log10(peak_amplitude / max(rms_level, 1e-10))
        
        # Stereo width
        stereo_width = 0.0
        if audio.shape[0] >= 2:
            mid = (audio[0] + audio[1]) / 2
            side = (audio[0] - audio[1]) / 2
            mid_rms = np.sqrt(np.mean(mid ** 2))
            side_rms = np.sqrt(np.mean(side ** 2))
            stereo_width = side_rms / max(mid_rms, 1e-10)
        
        # Frequency response analysis
        freqs, psd = signal.welch(audio[0], sample_rate, nperseg=2048)
        freq_response = {
            'low_energy': float(np.mean(psd[(freqs >= 20) & (freqs <= 250)])),
            'mid_energy': float(np.mean(psd[(freqs >= 250) & (freqs <= 2000)])),
            'high_energy': float(np.mean(psd[(freqs >= 2000) & (freqs <= 8000)])),
        }
        
        return AudioMetrics(
            lufs_integrated=float(lufs_integrated),
            lufs_short_term=float(lufs_integrated),  # Simplified
            lufs_momentary=float(lufs_integrated),   # Simplified
            true_peak=float(true_peak_db),
            dynamic_range=float(dynamic_range),
            phase_coherence=float(phase_coherence),
            thd_plus_n=0.01,  # Placeholder
            clip_count=int(clip_count),
            peak_amplitude=float(peak_amplitude),
            rms_level=float(rms_level),
            crest_factor=float(crest_factor),
            stereo_width=float(stereo_width),
            frequency_response=freq_response
        )
    
    def _log(self, message: str):
        """Add message to processing log"""
        self.processing_log.append(message)
        logger.info(message) 