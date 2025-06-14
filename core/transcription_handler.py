from typing import Dict, Optional, List
import logging
from pathlib import Path
from services.whisper_service import WhisperService
from services.claude_service import ClaudeService
from services.pyannote_service import PyAnnoteService
from config.settings import settings
import asyncio

logger = logging.getLogger(__name__)

class TranscriptionHandler:
    """Handles the complete transcription pipeline"""
    
    def __init__(self):
        self.whisper_service = WhisperService()
        self.claude_service = ClaudeService()
        self.pyannote_service = PyAnnoteService(
            use_auth_token=settings.HUGGING_FACE_TOKEN,
            use_gpu=settings.ENABLE_GPU
        )
        
    async def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        enhance: bool = True,
        return_segments: bool = True
    ) -> Dict:
        """
        Complete transcription pipeline
        
        Args:
            audio_path: Path to audio file
            language: Optional language code
            enhance: Whether to enhance with Claude
            return_segments: Whether to return timestamped segments
            
        Returns:
            Dictionary with transcription results
        """
        try:
            logger.info(f"Starting transcription for: {audio_path}")
            
            # Get raw transcription from Whisper
            whisper_result = await self.whisper_service.transcribe_audio(
                audio_path,
                language=language,
                return_timestamps=return_segments
            )
            
            result = {
                'text': whisper_result['text'],
                'segments': whisper_result.get('segments', []),
                'language': whisper_result.get('language', 'en'),
                'duration': whisper_result.get('duration', 0),
                'confidence': whisper_result.get('confidence', 0),
                'word_count': whisper_result.get('word_count', 0)
            }
            
            # Enhance if requested
            if enhance and result['text']:
                logger.info("Enhancing transcription with Claude")
                
                enhanced_text = await self.claude_service.enhance_transcription(
                    result['text']
                )
                
                result['enhanced_text'] = enhanced_text
                result['enhanced_word_count'] = len(enhanced_text.split())
            
            logger.info("Transcription completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            raise
    
    async def transcribe_with_speaker_diarization(
        self,
        audio_path: str,
        num_speakers: Optional[int] = None
    ) -> Dict:
        """
        Transcribe with speaker diarization using PyAnnote.audio
        
        Args:
            audio_path: Path to audio file
            num_speakers: Optional number of speakers (not used with PyAnnote)
            
        Returns:
            Dictionary with diarized transcription
        """
        try:
            logger.info(f"Starting diarized transcription for: {audio_path}")
            
            # Get basic transcription
            transcription_result = await self.transcribe(audio_path, enhance=True)
            
            # Get speaker diarization
            speaker_analysis = await self.pyannote_service.analyze_speakers(audio_path)
            
            # Get voice activity detection
            vad_analysis = await self.pyannote_service.analyze_voice_activity(audio_path)
            
            # Get overlap detection
            overlap_analysis = await self.pyannote_service.detect_overlapped_speech(audio_path)
            
            # Combine results
            result = {
                **transcription_result,
                'speaker_analysis': speaker_analysis,
                'voice_activity': vad_analysis,
                'overlap_analysis': overlap_analysis,
                'diarized_segments': self._align_transcription_with_speakers(
                    transcription_result.get('segments', []),
                    speaker_analysis.get('speakers', {})
                )
            }
            
            logger.info("✓ Diarized transcription completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Diarized transcription error: {e}")
            # Fallback to regular transcription
            return await self.transcribe(audio_path, enhance=True)
    
    def _align_transcription_with_speakers(
        self, 
        transcription_segments: List[Dict],
        speakers: Dict[str, Dict]
    ) -> List[Dict]:
        """
        Align transcription segments with speaker segments
        
        Args:
            transcription_segments: Whisper transcription segments
            speakers: PyAnnote speaker analysis results
            
        Returns:
            List of aligned segments with speaker information
        """
        if not transcription_segments or not speakers:
            return transcription_segments
        
        # Create a flat list of all speaker segments with speaker IDs
        speaker_segments = []
        for speaker_id, speaker_data in speakers.items():
            for segment in speaker_data.get('segments', []):
                speaker_segments.append({
                    'start': segment['start'],
                    'end': segment['end'],
                    'speaker_id': speaker_id,
                    'gender': speaker_data.get('gender', 'unknown'),
                    'confidence': speaker_data.get('confidence', 0.0)
                })
        
        # Sort by start time
        speaker_segments.sort(key=lambda x: x['start'])
        
        # Align transcription segments with speaker segments
        aligned_segments = []
        
        for trans_seg in transcription_segments:
            trans_start = trans_seg.get('start', 0)
            trans_end = trans_seg.get('end', trans_start + 1)
            trans_mid = (trans_start + trans_end) / 2
            
            # Find the speaker segment that best overlaps with this transcription segment
            best_speaker = None
            best_overlap = 0
            
            for spk_seg in speaker_segments:
                # Calculate overlap
                overlap_start = max(trans_start, spk_seg['start'])
                overlap_end = min(trans_end, spk_seg['end'])
                overlap_duration = max(0, overlap_end - overlap_start)
                
                if overlap_duration > best_overlap:
                    best_overlap = overlap_duration
                    best_speaker = spk_seg
            
            # Create aligned segment
            aligned_segment = {
                **trans_seg,
                'speaker_id': best_speaker['speaker_id'] if best_speaker else 'UNKNOWN',
                'speaker_gender': best_speaker['gender'] if best_speaker else 'unknown',
                'speaker_confidence': best_speaker['confidence'] if best_speaker else 0.0,
                'overlap_duration': best_overlap
            }
            
            aligned_segments.append(aligned_segment)
        
        return aligned_segments 