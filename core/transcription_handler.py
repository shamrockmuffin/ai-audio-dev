from typing import Dict, Optional, List
import logging
from pathlib import Path
from services.whisper_service import WhisperService
from services.claude_service import ClaudeService
from config.settings import settings
import asyncio

logger = logging.getLogger(__name__)

class TranscriptionHandler:
    """Handles the complete transcription pipeline"""
    
    def __init__(self):
        self.whisper_service = WhisperService()
        self.claude_service = ClaudeService()
        
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
        Transcribe with speaker diarization
        
        Args:
            audio_path: Path to audio file
            num_speakers: Optional number of speakers
            
        Returns:
            Dictionary with diarized transcription
        """
        # This is a placeholder for speaker diarization
        # In production, you would integrate a diarization model
        
        result = await self.transcribe(audio_path, enhance=True)
        
        # Simulated diarization result
        result['speakers'] = self._simulate_diarization(
            result['segments'], 
            num_speakers
        )
        
        return result
    
    def _simulate_diarization(
        self, 
        segments: List[Dict], 
        num_speakers: Optional[int] = None
    ) -> List[Dict]:
        """Simulate speaker diarization (placeholder)"""
        # In production, use pyannote or similar
        speakers = []
        current_speaker = 0
        
        for i, segment in enumerate(segments):
            # Simple simulation: switch speakers every few segments
            if i % 5 == 0 and i > 0:
                current_speaker = (current_speaker + 1) % (num_speakers or 2)
            
            speakers.append({
                'segment_id': i,
                'speaker': f"Speaker {current_speaker + 1}",
                'confidence': 0.85
            })
        
        return speakers 