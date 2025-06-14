import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
import asyncio
from pathlib import Path
from config.settings import settings

# Import torchaudio and transformers with error handling
try:
    import torchaudio
except ImportError as e:
    logging.error(f"Failed to import torchaudio: {e}")
    torchaudio = None

try:
    # Import transformers components separately to avoid circular imports
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
except ImportError as e:
    logging.error(f"Failed to import transformers: {e}")
    WhisperProcessor = None
    WhisperForConditionalGeneration = None

class WhisperService:
    """Service for audio transcription using Whisper"""
    
    def __init__(self, model_name: str = None):
        self.logger = logging.getLogger(__name__)
        self.model_name = model_name or settings.WHISPER_MODEL
        self.device = self._get_best_device()
        
        # Check if required dependencies are available
        if WhisperProcessor is None or WhisperForConditionalGeneration is None:
            raise ImportError("Transformers library not properly installed. Please reinstall with: pip install transformers")
        if torchaudio is None:
            raise ImportError("Torchaudio not properly installed. Please reinstall with: pip install torchaudio")
            
        self._load_model()
    
    def _get_best_device(self):
        """Get the best available device for processing"""
        if torch.cuda.is_available() and settings.ENABLE_GPU:
            # Use the specified GPU device
            device_id = settings.GPU_DEVICE_ID
            if device_id < torch.cuda.device_count():
                device = f"cuda:{device_id}"
                gpu_name = torch.cuda.get_device_name(device_id)
                self.logger.info(f"CUDA available. Using GPU {device_id}: {gpu_name}")
                return device
            else:
                self.logger.warning(f"GPU device {device_id} not available. Using CPU.")
                return "cpu"
        else:
            if not torch.cuda.is_available():
                self.logger.warning("CUDA not available. Using CPU.")
            else:
                self.logger.info("GPU disabled in settings. Using CPU.")
            return "cpu"
    
    def _load_model(self):
        """Load Whisper model and processor"""
        try:
            self.logger.info(f"Loading Whisper model: {self.model_name}")
            self.processor = WhisperProcessor.from_pretrained(self.model_name)
            # Determine dtype based on device and settings
            use_fp16 = self.device.startswith("cuda") and settings.USE_FP16
            dtype = torch.float16 if use_fp16 else torch.float32
            
            self.model = WhisperForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=dtype,
                device_map="auto" if self.device.startswith("cuda") else None
            )
            
            # Enable GPU optimizations if using CUDA
            if self.device.startswith("cuda"):
                if settings.USE_FP16:
                    self.model.half()  # Use FP16 for faster inference
                torch.cuda.empty_cache()  # Clear GPU cache
                self.logger.info(f"Model loaded on {self.device} with {'FP16' if settings.USE_FP16 else 'FP32'} precision")
            else:
                self.logger.info(f"Model loaded on {self.device}")
                
        except Exception as e:
            self.logger.error(f"Error loading Whisper model: {e}")
            raise
    
    async def transcribe_audio(
        self,
        audio_path: str,
        language: Optional[str] = None,
        return_timestamps: bool = True
    ) -> Dict:
        """
        Transcribe audio file
        
        Args:
            audio_path: Path to audio file
            language: Optional language code
            return_timestamps: Whether to return word timestamps
            
        Returns:
            Dictionary with transcription results
        """
        try:
            # Load audio
            audio_array, sampling_rate = await self._load_audio_async(audio_path)
            
            # Resample if needed
            if sampling_rate != 16000:
                audio_array = await self._resample_audio_async(
                    audio_array, 
                    sampling_rate, 
                    16000
                )
                sampling_rate = 16000
            
            # Split into chunks if needed
            chunks = self._split_audio(audio_array, sampling_rate)
            
            # Transcribe chunks
            all_segments = []
            full_text = []
            
            for i, chunk in enumerate(chunks):
                self.logger.info(f"Transcribing chunk {i+1}/{len(chunks)}")
                segments, text = await self._transcribe_chunk(
                    chunk,
                    sampling_rate,
                    language,
                    return_timestamps
                )
                all_segments.extend(segments)
                full_text.append(text)
            
            # Combine results
            result = {
                'text': ' '.join(full_text),
                'segments': all_segments,
                'language': language or 'en',
                'duration': len(audio_array) / sampling_rate,
                'confidence': self._calculate_confidence(all_segments),
                'word_count': len(' '.join(full_text).split())
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Transcription error: {e}")
            raise
    
    async def _load_audio_async(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """Load audio file asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._load_audio, audio_path)
    
    def _load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """Load audio file"""
        waveform, sample_rate = torchaudio.load(audio_path)
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        # Convert to numpy
        return waveform.squeeze().numpy(), sample_rate
    
    async def _resample_audio_async(
        self, 
        audio: np.ndarray, 
        orig_sr: int, 
        target_sr: int
    ) -> np.ndarray:
        """Resample audio asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, 
            self._resample_audio, 
            audio, 
            orig_sr, 
            target_sr
        )
    
    def _resample_audio(
        self, 
        audio: np.ndarray, 
        orig_sr: int, 
        target_sr: int
    ) -> np.ndarray:
        """Resample audio to target sample rate"""
        resampler = torchaudio.transforms.Resample(orig_sr, target_sr)
        audio_tensor = torch.from_numpy(audio).float()
        resampled = resampler(audio_tensor)
        return resampled.numpy()
    
    def _split_audio(
        self, 
        audio: np.ndarray, 
        sample_rate: int, 
        chunk_length: int = 30
    ) -> List[np.ndarray]:
        """Split audio into chunks"""
        chunk_samples = chunk_length * sample_rate
        chunks = []
        
        for i in range(0, len(audio), chunk_samples):
            chunk = audio[i:i + chunk_samples]
            if len(chunk) > sample_rate:  # At least 1 second
                chunks.append(chunk)
        
        return chunks
    
    async def _transcribe_chunk(
        self,
        audio_chunk: np.ndarray,
        sample_rate: int,
        language: Optional[str],
        return_timestamps: bool
    ) -> Tuple[List[Dict], str]:
        """Transcribe a single audio chunk"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._transcribe_chunk_sync,
            audio_chunk,
            sample_rate,
            language,
            return_timestamps
        )
    
    def _transcribe_chunk_sync(
        self,
        audio_chunk: np.ndarray,
        sample_rate: int,
        language: Optional[str],
        return_timestamps: bool
    ) -> Tuple[List[Dict], str]:
        """Synchronous chunk transcription"""
        # Prepare input
        input_features = self.processor(
            audio_chunk,
            sampling_rate=sample_rate,
            return_tensors="pt"
        ).input_features.to(self.device)
        
        # Ensure input features match model precision
        if self.device.startswith("cuda") and settings.USE_FP16:
            input_features = input_features.half()
        
        # Generate transcription
        with torch.no_grad():
            predicted_ids = self.model.generate(
                input_features,
                language=language,
                return_timestamps=return_timestamps
            )
        
        # Decode
        transcription = self.processor.batch_decode(
            predicted_ids,
            skip_special_tokens=True
        )[0]
        
        # Parse segments if timestamps requested
        segments = []
        if return_timestamps:
            # Simple segment parsing (would need more sophisticated parsing for production)
            words = transcription.split()
            segment_duration = len(audio_chunk) / sample_rate / len(words) if words else 0
            
            for i, word in enumerate(words):
                segments.append({
                    'text': word,
                    'start': i * segment_duration,
                    'end': (i + 1) * segment_duration
                })
        
        return segments, transcription
    
    def _calculate_confidence(self, segments: List[Dict]) -> float:
        """Calculate overall confidence score"""
        # Simplified confidence calculation
        # In production, would use actual model confidence scores
        if not segments:
            return 0.0
        
        # Base confidence on segment count and text length
        avg_segment_length = np.mean([len(s.get('text', '')) for s in segments])
        confidence = min(0.95, 0.7 + (avg_segment_length / 20) * 0.25)
        
        return confidence 