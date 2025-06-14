import numpy as np
import librosa
import noisereduce as nr
from scipy import signal
from typing import List, Optional, Dict, Callable
import logging
from concurrent.futures import ThreadPoolExecutor
import asyncio
from dataclasses import dataclass

@dataclass
class EnhancementResult:
    """Result of audio enhancement processing"""
    enhanced_audio: np.ndarray
    original_audio: np.ndarray
    sample_rate: int
    enhancement_params: Dict
    processing_time: float

class AudioProcessor:
    """Handles audio processing and enhancement"""
    
    def __init__(self, max_workers: int = 4):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.logger = logging.getLogger(__name__)
        
    async def process_audio_file(
        self, 
        file_path: str,
        settings: Dict,
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> EnhancementResult:
        """
        Process audio file with enhancement
        
        Args:
            file_path: Path to audio file
            settings: Enhancement settings dictionary
            progress_callback: Optional callback for progress updates
            
        Returns:
            EnhancementResult object
        """
        try:
            import time
            start_time = time.time()
            
            # Load audio
            if progress_callback:
                progress_callback(0.1)
                
            audio_data, sr = await self._load_audio_async(file_path)
            
            if progress_callback:
                progress_callback(0.2)
            
            # Apply enhancements based on settings
            enhanced_audio = audio_data.copy()
            
            if settings.get('noise_reduction', True):
                if progress_callback:
                    progress_callback(0.4)
                enhanced_audio = await self._apply_noise_reduction_async(enhanced_audio, sr)
            
            if settings.get('normalize', True):
                if progress_callback:
                    progress_callback(0.6)
                enhanced_audio = self._normalize_audio(enhanced_audio)
            
            if settings.get('band_pass', True):
                if progress_callback:
                    progress_callback(0.8)
                enhanced_audio = self._apply_band_pass_filter(enhanced_audio, sr)
            
            # Apply compression if needed
            if settings.get('compression', False):
                enhanced_audio = self._apply_compression(enhanced_audio)
            
            if progress_callback:
                progress_callback(1.0)
            
            processing_time = time.time() - start_time
            
            return EnhancementResult(
                enhanced_audio=enhanced_audio,
                original_audio=audio_data,
                sample_rate=sr,
                enhancement_params=self._get_enhancement_params(settings),
                processing_time=processing_time
            )
            
        except Exception as e:
            self.logger.error(f"Error processing audio: {e}")
            raise
    
    async def _load_audio_async(self, file_path: str) -> tuple:
        """Load audio file asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self._load_audio,
            file_path
        )
    
    def _load_audio(self, file_path: str) -> tuple:
        """Load audio file"""
        try:
            # Load with librosa
            audio_data, sr = librosa.load(file_path, sr=None, mono=True)
            
            # Validate audio data
            if not np.isfinite(audio_data).all():
                raise ValueError("Audio contains non-finite values")
            
            return audio_data, sr
            
        except Exception as e:
            self.logger.error(f"Error loading audio file: {e}")
            raise
    
    async def _apply_noise_reduction_async(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Apply noise reduction asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self._apply_noise_reduction,
            audio,
            sr
        )
    
    def _apply_noise_reduction(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Apply noise reduction to audio"""
        try:
            # Use noisereduce library
            reduced_noise = nr.reduce_noise(
                y=audio, 
                sr=sr,
                stationary=True,
                prop_decrease=1.0
            )
            return reduced_noise
            
        except Exception as e:
            self.logger.warning(f"Noise reduction failed: {e}")
            return audio
    
    def _normalize_audio(self, audio: np.ndarray, target_db: float = -20.0) -> np.ndarray:
        """Normalize audio to target dB level"""
        try:
            # Calculate RMS
            rms = np.sqrt(np.mean(audio**2))
            
            # Avoid division by zero
            if rms == 0:
                return audio
            
            # Calculate scaling factor
            target_rms = 10**(target_db / 20)
            scaling_factor = target_rms / rms
            
            # Apply scaling with clipping prevention
            normalized = audio * scaling_factor
            
            # Soft clipping if needed
            if np.abs(normalized).max() > 0.95:
                normalized = np.tanh(normalized * 0.7) / 0.7
            
            return normalized
            
        except Exception as e:
            self.logger.warning(f"Normalization failed: {e}")
            return audio
    
    def _apply_band_pass_filter(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Apply band-pass filter for speech frequencies"""
        try:
            # Design butterworth bandpass filter
            # Focus on speech frequency range (300-3400 Hz)
            nyquist = sr / 2
            low_freq = 300 / nyquist
            high_freq = min(3400 / nyquist, 0.95)  # Ensure < 1
            
            # Create filter
            b, a = signal.butter(
                4,  # Order
                [low_freq, high_freq],
                btype='band',
                output='ba'
            )
            
            # Apply filter
            filtered = signal.filtfilt(b, a, audio)
            
            return filtered
            
        except Exception as e:
            self.logger.warning(f"Band-pass filter failed: {e}")
            return audio
    
    def _apply_compression(self, audio: np.ndarray, 
                          threshold: float = -20.0,
                          ratio: float = 4.0) -> np.ndarray:
        """Apply dynamic range compression"""
        try:
            # Convert to dB
            audio_db = 20 * np.log10(np.abs(audio) + 1e-10)
            
            # Apply compression
            compressed_db = np.where(
                audio_db > threshold,
                threshold + (audio_db - threshold) / ratio,
                audio_db
            )
            
            # Convert back to linear
            compressed = np.sign(audio) * (10 ** (compressed_db / 20))
            
            return compressed
            
        except Exception as e:
            self.logger.warning(f"Compression failed: {e}")
            return audio
    
    def _get_enhancement_params(self, settings: Dict) -> Dict:
        """Get enhancement parameters for result"""
        return {
            "noise_reduction": settings.get('noise_reduction', True),
            "normalization": settings.get('normalize', True),
            "band_pass_filter": settings.get('band_pass', True),
            "compression": settings.get('compression', False),
            "target_db": -20.0,
            "filter_range": "300-3400 Hz"
        }
    
    def analyze_audio(self, audio_data: np.ndarray, sr: int) -> Dict:
        """Analyze audio and return metrics"""
        try:
            # Basic metrics
            duration = len(audio_data) / sr
            rms = np.sqrt(np.mean(audio_data**2))
            peak = np.abs(audio_data).max()
            
            # Dynamic range
            db_range = 20 * np.log10(peak / (rms + 1e-10))
            
            # Tempo detection (if applicable)
            tempo = None
            try:
                tempo, _ = librosa.beat.beat_track(y=audio_data, sr=sr)
            except:
                pass
            
            # Zero crossing rate
            zcr = np.mean(librosa.feature.zero_crossing_rate(audio_data))
            
            return {
                "duration": duration,
                "sample_rate": sr,
                "rms": float(rms),
                "peak": float(peak),
                "dynamic_range_db": float(db_range),
                "tempo": float(tempo) if tempo else None,
                "zero_crossing_rate": float(zcr)
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing audio: {e}")
            return {}
    
    def __del__(self):
        """Cleanup executor on deletion"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False) 