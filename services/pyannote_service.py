#!/usr/bin/env python3
"""
PyAnnote Audio Service for Professional Speaker Diarization
Integrates pyannote.audio for state-of-the-art speaker analysis
"""

import os
import logging
import numpy as np
import librosa
import warnings
from typing import Optional, Dict, List, Any, Tuple
import torch
import torchaudio
from pathlib import Path
import asyncio
from datetime import datetime

# PyAnnote imports with fallback
try:
    from pyannote.audio import Pipeline, Model
    from pyannote.core import Segment, Annotation
    from pyannote.audio.pipelines.utils.hook import ProgressHook
    from pyannote.audio.pipelines import VoiceActivityDetection, OverlappedSpeechDetection
    PYANNOTE_AVAILABLE = True
except ImportError:
    PYANNOTE_AVAILABLE = False
    # Create dummy classes for type hints when PyAnnote is not available
    class Annotation:
        pass

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class PyAnnoteService:
    """Professional speaker diarization service using PyAnnote.audio"""
    
    def __init__(self, 
                 use_auth_token: Optional[str] = None,
                 use_gpu: bool = True,
                 show_progress: bool = True,
                 device: Optional[str] = None):
        """
        Initialize PyAnnote Service
        
        Args:
            use_auth_token: HuggingFace token for accessing gated models
            use_gpu: Whether to use GPU acceleration
            show_progress: Whether to show progress bars
            device: Specific device to use ('cuda', 'cpu', etc.)
        """
        self.sample_rate = 16000
        # Use provided token, environment variable, or hardcoded token as fallback
        self.use_auth_token = (
            use_auth_token or 
            os.environ.get('HUGGING_FACE_TOKEN') or 
            "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"
        )
        
        # Try to use HF Hub authentication if available
        try:
            from huggingface_hub import HfFolder
            if not self.use_auth_token:
                stored_token = HfFolder.get_token()
                if stored_token:
                    self.use_auth_token = stored_token
                    logger.info("✓ Using stored Hugging Face token")
        except ImportError:
            pass
        self.show_progress = show_progress
        self.available = PYANNOTE_AVAILABLE
        
        # Determine device
        if device:
            self.device = torch.device(device)
        elif use_gpu and torch.cuda.is_available():
            self.device = torch.device('cuda')
            logger.info("✓ Using GPU acceleration (CUDA) for PyAnnote")
        else:
            self.device = torch.device('cpu')
            logger.info("→ Using CPU processing for PyAnnote")
        
        # Initialize pipelines
        self.diarization_pipeline = None
        self.segmentation_model = None
        self.vad_pipeline = None
        self.overlap_detection_pipeline = None
        
        if self.available:
            self._initialize_pipelines()
        else:
            logger.warning("PyAnnote.audio not available. Install with: pip install pyannote.audio")
    
    def _initialize_pipelines(self):
        """Initialize pyannote pipelines"""
        try:
            logger.info("Initializing pyannote.audio pipelines...")
            
            # 1. Speaker diarization pipeline - Use 3.0 model
            try:
                logger.info(f"Loading speaker diarization with token: {self.use_auth_token[:10]}...")
                self.diarization_pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.0",
                    use_auth_token=self.use_auth_token
                )
                logger.info("✓ Loaded speaker-diarization-3.0")
                
                # Move to device and set precision
                if self.device.type == 'cuda':
                    self.diarization_pipeline.to(self.device)
                    # Set to float32 to avoid precision mismatch issues
                    for model in self.diarization_pipeline._models.values():
                        if hasattr(model, 'float'):
                            model.float()
                    logger.info(f"✓ Moved diarization pipeline to {self.device} with float32 precision")
                
            except Exception as e:
                error_msg = str(e)
                if "401" in error_msg and "Unauthorized" in error_msg:
                    logger.error("❌ AUTHENTICATION ERROR: Please accept the user conditions for PyAnnote models:")
                    logger.error("   1. Visit https://huggingface.co/pyannote/speaker-diarization-3.0")
                    logger.error("   2. Visit https://huggingface.co/pyannote/segmentation-3.0")
                    logger.error("   3. Accept the conditions for both models")
                    logger.error("   4. Make sure you're using a valid Hugging Face token")
                else:
                    logger.warning(f"Failed to load diarization-3.0: {e}")
                
                try:
                    # Fallback to 3.1 if 3.0 fails
                    self.diarization_pipeline = Pipeline.from_pretrained(
                        "pyannote/speaker-diarization-3.1",
                        use_auth_token=self.use_auth_token
                    )
                    logger.info("✓ Loaded speaker-diarization-3.1 as fallback")
                    
                    if self.device.type == 'cuda':
                        self.diarization_pipeline.to(self.device)
                        # Set to float32 to avoid precision mismatch issues
                        for model in self.diarization_pipeline._models.values():
                            if hasattr(model, 'float'):
                                model.float()
                        logger.info(f"✓ Moved diarization pipeline to {self.device} with float32 precision")
                        
                except Exception as e2:
                    error_msg2 = str(e2)
                    if "401" in error_msg2 and "Unauthorized" in error_msg2:
                        logger.error("❌ AUTHENTICATION ERROR: Please accept the user conditions for PyAnnote models:")
                        logger.error("   1. Visit https://huggingface.co/pyannote/speaker-diarization-3.1")
                        logger.error("   2. Accept the conditions for the model")
                        logger.error("   3. Make sure you're using a valid Hugging Face token")
                    else:
                        logger.error(f"Failed to load any diarization model: {e2}")
                    self.diarization_pipeline = None
            
            # 2. Segmentation model for advanced VAD and overlap detection
            try:
                logger.info("Loading segmentation model...")
                self.segmentation_model = Model.from_pretrained(
                    "pyannote/segmentation-3.0",
                    use_auth_token=self.use_auth_token
                )
                logger.info("✓ Loaded segmentation-3.0 model")
                
                # Set to float32 to avoid precision issues
                if hasattr(self.segmentation_model, 'float'):
                    self.segmentation_model.float()
                    
                # Keep segmentation model on CPU to avoid device conflicts
                logger.info("✓ Keeping segmentation model on CPU for compatibility")
                
                # Initialize VAD pipeline with segmentation model
                try:
                    self.vad_pipeline = VoiceActivityDetection(segmentation=self.segmentation_model)
                    # Use default hyperparameters for PyAnnote 3.0
                    self.vad_pipeline.instantiate()
                    logger.info("✓ Initialized VAD pipeline with segmentation model")
                except Exception as vad_error:
                    logger.warning(f"VAD pipeline initialization failed: {vad_error}")
                    self.vad_pipeline = None
                
                # Initialize overlapped speech detection
                try:
                    self.overlap_detection_pipeline = OverlappedSpeechDetection(segmentation=self.segmentation_model)
                    # Use default hyperparameters for PyAnnote 3.0
                    self.overlap_detection_pipeline.instantiate()
                    logger.info("✓ Initialized overlap detection pipeline")
                except Exception as overlap_error:
                    logger.warning(f"Overlap detection pipeline initialization failed: {overlap_error}")
                    self.overlap_detection_pipeline = None
                
            except Exception as e:
                error_msg = str(e)
                if "401" in error_msg and "Unauthorized" in error_msg:
                    logger.error("❌ AUTHENTICATION ERROR: Please accept the user conditions for:")
                    logger.error("   Visit https://huggingface.co/pyannote/segmentation-3.0 and accept conditions")
                else:
                    logger.warning(f"Failed to load segmentation model: {e}")
                self.segmentation_model = None
                self.vad_pipeline = None
                self.overlap_detection_pipeline = None
            
            logger.info("✓ PyAnnote pipelines initialization complete")
            
        except Exception as e:
            logger.error(f"Error initializing pyannote pipelines: {e}")
            self.diarization_pipeline = None
            self.segmentation_model = None
            self.vad_pipeline = None
            self.overlap_detection_pipeline = None
    
    async def analyze_speakers(self, audio_path: str) -> Dict[str, Any]:
        """
        Perform comprehensive speaker analysis
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with speaker analysis results
        """
        if not self.available or not self.diarization_pipeline:
            return await self._fallback_speaker_analysis(audio_path)
        
        try:
            logger.info(f"Starting PyAnnote speaker analysis for: {audio_path}")
            
            # Run diarization
            if self.show_progress:
                with ProgressHook() as hook:
                    diarization = self.diarization_pipeline(audio_path, hook=hook)
            else:
                diarization = self.diarization_pipeline(audio_path)
            
            # Load audio for additional analysis
            waveform, _ = librosa.load(audio_path, sr=self.sample_rate)
            
            # Process diarization results
            speakers = {}
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                if speaker not in speakers:
                    speakers[speaker] = {
                        'segments': [],
                        'total_duration': 0.0,
                        'gender': 'unknown',
                        'confidence': 0.0,
                        'voice_characteristics': {}
                    }
                
                segment = {
                    'start': float(turn.start),
                    'end': float(turn.end),
                    'duration': float(turn.end - turn.start)
                }
                speakers[speaker]['segments'].append(segment)
                speakers[speaker]['total_duration'] += segment['duration']
            
            # Analyze voice characteristics for each speaker
            for speaker_id, speaker_data in speakers.items():
                characteristics = await self._analyze_voice_characteristics(
                    waveform, speaker_data['segments']
                )
                speaker_data['voice_characteristics'] = characteristics
                speaker_data['gender'] = characteristics.get('gender', 'unknown')
                speaker_data['confidence'] = characteristics.get('confidence', 0.0)
            
            # Calculate gender distribution
            gender_count = {'male': 0, 'female': 0, 'unknown': 0}
            for speaker_data in speakers.values():
                gender_count[speaker_data['gender']] += 1
            
            result = {
                'speakers': speakers,
                'total_speakers': len(speakers),
                'gender_distribution': gender_count,
                'diarization_confidence': self._calculate_diarization_confidence(diarization),
                'method': 'pyannote.audio',
                'device_used': str(self.device)
            }
            
            logger.info(f"✓ Speaker analysis complete: {len(speakers)} speakers detected")
            return result
            
        except Exception as e:
            logger.error(f"PyAnnote speaker analysis failed: {e}")
            return await self._fallback_speaker_analysis(audio_path)
    
    async def analyze_voice_activity(self, audio_path: str) -> Dict[str, Any]:
        """Advanced voice activity detection using pyannote segmentation"""
        if not self.vad_pipeline:
            return await self._fallback_vad_analysis(audio_path)
        
        try:
            logger.info("Running advanced voice activity detection...")
            
            # Use CPU for segmentation models to avoid CUDA device issues
            with torch.no_grad():
                vad_result = self.vad_pipeline(audio_path)
            
            # Convert to segments
            vad_segments = []
            total_speech_duration = 0.0
            
            for segment in vad_result.itersegments():
                seg_data = {
                    'start': float(segment.start),
                    'end': float(segment.end),
                    'duration': float(segment.duration)
                }
                vad_segments.append(seg_data)
                total_speech_duration += segment.duration
            
            # Calculate speech ratio
            waveform, _ = librosa.load(audio_path, sr=self.sample_rate)
            total_duration = len(waveform) / self.sample_rate
            speech_ratio = total_speech_duration / total_duration if total_duration > 0 else 0.0
            
            return {
                'vad_segments': vad_segments,
                'total_speech_duration': total_speech_duration,
                'speech_ratio': speech_ratio,
                'method': 'pyannote_segmentation'
            }
            
        except Exception as e:
            logger.error(f"Advanced VAD failed: {e}")
            return await self._fallback_vad_analysis(audio_path)
    
    async def detect_overlapped_speech(self, audio_path: str) -> Dict[str, Any]:
        """Detect overlapped speech using pyannote segmentation"""
        if not self.overlap_detection_pipeline:
            return {'overlap_segments': [], 'total_overlap_duration': 0.0, 'overlap_ratio': 0.0}
        
        try:
            logger.info("Detecting overlapped speech...")
            
            # Use CPU for segmentation models to avoid CUDA device issues
            with torch.no_grad():
                overlap_result = self.overlap_detection_pipeline(audio_path)
            
            # Convert to segments
            overlap_segments = []
            total_overlap_duration = 0.0
            
            for segment in overlap_result.itersegments():
                seg_data = {
                    'start': float(segment.start),
                    'end': float(segment.end),
                    'duration': float(segment.duration)
                }
                overlap_segments.append(seg_data)
                total_overlap_duration += segment.duration
            
            # Calculate overlap ratio
            waveform, _ = librosa.load(audio_path, sr=self.sample_rate)
            total_duration = len(waveform) / self.sample_rate
            overlap_ratio = total_overlap_duration / total_duration if total_duration > 0 else 0.0
            
            return {
                'overlap_segments': overlap_segments,
                'total_overlap_duration': total_overlap_duration,
                'overlap_ratio': overlap_ratio,
                'method': 'pyannote_segmentation'
            }
            
        except Exception as e:
            logger.error(f"Overlap detection failed: {e}")
            return {'overlap_segments': [], 'total_overlap_duration': 0.0, 'overlap_ratio': 0.0}
    
    async def _analyze_voice_characteristics(self, waveform: np.ndarray, segments: List[Dict]) -> Dict[str, Any]:
        """Analyze detailed voice characteristics for a speaker"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._analyze_voice_characteristics_sync, waveform, segments)
    
    def _analyze_voice_characteristics_sync(self, waveform: np.ndarray, segments: List[Dict]) -> Dict[str, Any]:
        """Synchronous voice characteristics analysis"""
        all_features = []
        pitch_values = []
        
        for segment in segments:
            start_sample = int(segment['start'] * self.sample_rate)
            end_sample = int(segment['end'] * self.sample_rate)
            
            if end_sample > len(waveform):
                end_sample = len(waveform)
            
            segment_audio = waveform[start_sample:end_sample]
            
            if len(segment_audio) < self.sample_rate * 0.3:  # Skip very short segments
                continue
            
            # Pitch analysis
            try:
                pitches, magnitudes = librosa.piptrack(
                    y=segment_audio, sr=self.sample_rate, threshold=0.1
                )
                # Get strong pitch values
                strong_pitches = pitches[magnitudes > np.percentile(magnitudes[magnitudes > 0], 75)]
                if len(strong_pitches) > 0:
                    pitch_values.extend(strong_pitches[strong_pitches > 50])  # Filter out very low values
            except:
                pass
            
            # Spectral features
            try:
                spectral_centroids = librosa.feature.spectral_centroid(y=segment_audio, sr=self.sample_rate)[0]
                spectral_bandwidth = librosa.feature.spectral_bandwidth(y=segment_audio, sr=self.sample_rate)[0]
                spectral_rolloff = librosa.feature.spectral_rolloff(y=segment_audio, sr=self.sample_rate)[0]
                
                # MFCC features (voice timbre)
                mfccs = librosa.feature.mfcc(y=segment_audio, sr=self.sample_rate, n_mfcc=13)
                
                features = {
                    'spectral_centroid_mean': float(np.mean(spectral_centroids)),
                    'spectral_centroid_std': float(np.std(spectral_centroids)),
                    'spectral_bandwidth_mean': float(np.mean(spectral_bandwidth)),
                    'spectral_rolloff_mean': float(np.mean(spectral_rolloff)),
                    'mfcc_features': [float(x) for x in np.mean(mfccs, axis=1)[:8]]
                }
                all_features.append(features)
            except:
                pass
        
        # Aggregate characteristics
        characteristics = {
            'pitch_statistics': {},
            'spectral_statistics': {},
            'gender': 'unknown',
            'confidence': 0.0
        }
        
        # Pitch analysis
        if pitch_values:
            pitch_values = np.array(pitch_values)
            characteristics['pitch_statistics'] = {
                'mean': float(np.mean(pitch_values)),
                'std': float(np.std(pitch_values)),
                'median': float(np.median(pitch_values)),
                'min': float(np.min(pitch_values)),
                'max': float(np.max(pitch_values)),
                'q25': float(np.percentile(pitch_values, 25)),
                'q75': float(np.percentile(pitch_values, 75))
            }
            
            # Gender classification based on pitch
            mean_pitch = np.mean(pitch_values)
            if mean_pitch < 130:
                characteristics['gender'] = 'male'
                characteristics['confidence'] = 0.8
            elif mean_pitch > 200:
                characteristics['gender'] = 'female'
                characteristics['confidence'] = 0.8
            elif 130 <= mean_pitch <= 160:
                characteristics['gender'] = 'male'
                characteristics['confidence'] = 0.6
            elif 160 <= mean_pitch <= 200:
                characteristics['gender'] = 'female'
                characteristics['confidence'] = 0.6
            else:
                characteristics['confidence'] = 0.4
        
        # Spectral analysis
        if all_features:
            spectral_centroids = [f['spectral_centroid_mean'] for f in all_features]
            characteristics['spectral_statistics'] = {
                'centroid_mean': float(np.mean(spectral_centroids)),
                'centroid_std': float(np.std(spectral_centroids)),
                'feature_consistency': float(1.0 / (1.0 + np.std(spectral_centroids) / np.mean(spectral_centroids)))
            }
        
        return characteristics
    
    def _calculate_diarization_confidence(self, diarization: Annotation) -> float:
        """Calculate confidence score for diarization results"""
        total_duration = sum(segment.duration for segment in diarization.itersegments())
        if total_duration == 0:
            return 0.0
        
        # Factors affecting confidence:
        # 1. Number of speakers (more speakers = lower confidence)
        # 2. Segment duration consistency
        # 3. Speaker change frequency
        
        num_speakers = len(diarization.labels())
        segments = list(diarization.itersegments())
        
        if not segments:
            return 0.0
        
        # Duration consistency
        durations = [seg.duration for seg in segments]
        duration_consistency = 1.0 / (1.0 + np.std(durations) / np.mean(durations))
        
        # Speaker change frequency (lower is better for confidence)
        speaker_changes = len(segments) / total_duration
        change_factor = 1.0 / (1.0 + speaker_changes)
        
        # Number of speakers factor
        speaker_factor = 1.0 / (1.0 + 0.2 * num_speakers)
        
        confidence = duration_consistency * change_factor * speaker_factor
        return min(confidence, 1.0)
    
    async def _fallback_speaker_analysis(self, audio_path: str) -> Dict[str, Any]:
        """Fallback analysis when pyannote is not available"""
        try:
            logger.info("Using fallback speaker analysis...")
            waveform, _ = librosa.load(audio_path, sr=self.sample_rate)
            
            # Simple VAD
            rms = librosa.feature.rms(y=waveform, frame_length=1024, hop_length=512)[0]
            threshold = np.percentile(rms, 30)
            voice_frames = rms > threshold
            
            # Convert to segments
            segments = []
            in_voice = False
            start_time = 0
            
            frame_times = librosa.frames_to_time(np.arange(len(voice_frames)), sr=self.sample_rate, hop_length=512)
            
            for i, is_voice in enumerate(voice_frames):
                if is_voice and not in_voice:
                    start_time = frame_times[i]
                    in_voice = True
                elif not is_voice and in_voice:
                    if frame_times[i] - start_time > 0.5:
                        segments.append({
                            'start': start_time,
                            'end': frame_times[i],
                            'duration': frame_times[i] - start_time
                        })
                    in_voice = False
            
            if in_voice:
                segments.append({
                    'start': start_time,
                    'end': frame_times[-1],
                    'duration': frame_times[-1] - start_time
                })
            
            speakers = {}
            if segments:
                speakers['SPEAKER_00'] = {
                    'segments': segments,
                    'total_duration': sum(seg['duration'] for seg in segments),
                    'gender': 'unknown',
                    'confidence': 0.5,
                    'voice_characteristics': {}
                }
                
                # Try to determine gender
                characteristics = await self._analyze_voice_characteristics(waveform, segments)
                speakers['SPEAKER_00']['voice_characteristics'] = characteristics
                speakers['SPEAKER_00']['gender'] = characteristics.get('gender', 'unknown')
                speakers['SPEAKER_00']['confidence'] = characteristics.get('confidence', 0.5)
            
            gender_count = {'male': 0, 'female': 0, 'unknown': 0}
            for speaker_data in speakers.values():
                gender_count[speaker_data['gender']] += 1
            
            return {
                'speakers': speakers,
                'total_speakers': len(speakers),
                'gender_distribution': gender_count,
                'diarization_confidence': 0.5,
                'method': 'fallback',
                'device_used': str(self.device)
            }
            
        except Exception as e:
            logger.error(f"Fallback analysis failed: {e}")
            return {
                'speakers': {},
                'total_speakers': 0,
                'gender_distribution': {'male': 0, 'female': 0, 'unknown': 0},
                'diarization_confidence': 0.0,
                'method': 'failed',
                'device_used': str(self.device)
            }
    
    async def _fallback_vad_analysis(self, audio_path: str) -> Dict[str, Any]:
        """Fallback VAD analysis"""
        try:
            waveform, _ = librosa.load(audio_path, sr=self.sample_rate)
            
            # Simple energy-based VAD
            rms = librosa.feature.rms(y=waveform, frame_length=1024, hop_length=512)[0]
            threshold = np.percentile(rms, 30)
            voice_frames = rms > threshold
            
            # Convert to segments
            segments = []
            in_voice = False
            start_time = 0
            total_speech_duration = 0.0
            
            frame_times = librosa.frames_to_time(np.arange(len(voice_frames)), sr=self.sample_rate, hop_length=512)
            
            for i, is_voice in enumerate(voice_frames):
                if is_voice and not in_voice:
                    start_time = frame_times[i]
                    in_voice = True
                elif not is_voice and in_voice:
                    if frame_times[i] - start_time > 0.1:  # Minimum 100ms
                        duration = frame_times[i] - start_time
                        segments.append({
                            'start': start_time,
                            'end': frame_times[i],
                            'duration': duration
                        })
                        total_speech_duration += duration
                    in_voice = False
            
            if in_voice:
                duration = frame_times[-1] - start_time
                segments.append({
                    'start': start_time,
                    'end': frame_times[-1],
                    'duration': duration
                })
                total_speech_duration += duration
            
            total_duration = len(waveform) / self.sample_rate
            speech_ratio = total_speech_duration / total_duration if total_duration > 0 else 0.0
            
            return {
                'vad_segments': segments,
                'total_speech_duration': total_speech_duration,
                'speech_ratio': speech_ratio,
                'method': 'fallback_energy'
            }
            
        except Exception as e:
            logger.error(f"Fallback VAD failed: {e}")
            return {
                'vad_segments': [],
                'total_speech_duration': 0.0,
                'speech_ratio': 0.0,
                'method': 'failed'
            }