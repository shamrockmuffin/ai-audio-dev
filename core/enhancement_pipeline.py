from typing import Dict, List, Optional, Callable
import numpy as np
from dataclasses import dataclass
import logging
from core.audio_processor import AudioProcessor
from utils.audio_helpers import calculate_audio_statistics

logger = logging.getLogger(__name__)

@dataclass
class PipelineStage:
    """Represents a stage in the enhancement pipeline"""
    name: str
    function: Callable
    enabled: bool = True
    params: Optional[Dict] = None

class EnhancementPipeline:
    """Configurable audio enhancement pipeline"""
    
    def __init__(self):
        self.audio_processor = AudioProcessor()
        self.stages: List[PipelineStage] = []
        self._setup_default_pipeline()
    
    def _setup_default_pipeline(self):
        """Setup default enhancement stages"""
        self.stages = [
            PipelineStage(
                name="Pre-processing",
                function=self._preprocess,
                enabled=True
            ),
            PipelineStage(
                name="Noise Reduction",
                function=self._noise_reduction,
                enabled=True,
                params={'strength': 1.0}
            ),
            PipelineStage(
                name="Normalization",
                function=self._normalize,
                enabled=True,
                params={'target_db': -20.0}
            ),
            PipelineStage(
                name="EQ Enhancement",
                function=self._eq_enhancement,
                enabled=True,
                params={'boost_speech': True}
            ),
            PipelineStage(
                name="Post-processing",
                function=self._postprocess,
                enabled=True
            )
        ]
    
    async def process(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        settings: Optional[Dict] = None,
        progress_callback: Optional[Callable] = None
    ) -> Dict:
        """
        Process audio through enhancement pipeline
        
        Args:
            audio_data: Input audio array
            sample_rate: Sample rate
            settings: Optional settings override
            progress_callback: Progress callback function
            
        Returns:
            Dictionary with enhanced audio and metadata
        """
        try:
            # Apply settings
            if settings:
                self._apply_settings(settings)
            
            # Initial statistics
            initial_stats = calculate_audio_statistics(audio_data, sample_rate)
            
            # Process through stages
            processed_audio = audio_data.copy()
            stage_results = []
            
            enabled_stages = [s for s in self.stages if s.enabled]
            
            for i, stage in enumerate(enabled_stages):
                logger.info(f"Processing stage: {stage.name}")
                
                # Process stage
                processed_audio = await self._process_stage(
                    stage,
                    processed_audio,
                    sample_rate
                )
                
                # Calculate statistics after stage
                stage_stats = calculate_audio_statistics(
                    processed_audio, 
                    sample_rate
                )
                
                stage_results.append({
                    'name': stage.name,
                    'stats': stage_stats
                })
                
                # Update progress
                if progress_callback:
                    progress = (i + 1) / len(enabled_stages)
                    progress_callback(progress)
            
            # Final statistics
            final_stats = calculate_audio_statistics(processed_audio, sample_rate)
            
            return {
                'audio': processed_audio,
                'sample_rate': sample_rate,
                'initial_stats': initial_stats,
                'final_stats': final_stats,
                'stage_results': stage_results,
                'pipeline_config': [
                    {'name': s.name, 'enabled': s.enabled, 'params': s.params}
                    for s in self.stages
                ]
            }
            
        except Exception as e:
            logger.error(f"Pipeline processing error: {e}")
            raise
    
    async def _process_stage(
        self,
        stage: PipelineStage,
        audio: np.ndarray,
        sample_rate: int
    ) -> np.ndarray:
        """Process a single pipeline stage"""
        try:
            if stage.params:
                return await stage.function(audio, sample_rate, **stage.params)
            else:
                return await stage.function(audio, sample_rate)
        except Exception as e:
            logger.error(f"Error in stage {stage.name}: {e}")
            # Return original audio if stage fails
            return audio
    
    async def _preprocess(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Pre-processing stage"""
        # Remove DC offset
        audio = audio - np.mean(audio)
        
        # Gentle fade in/out to prevent clicks
        fade_samples = int(0.01 * sample_rate)  # 10ms fade
        audio[:fade_samples] *= np.linspace(0, 1, fade_samples)
        audio[-fade_samples:] *= np.linspace(1, 0, fade_samples)
        
        return audio
    
    async def _noise_reduction(
        self, 
        audio: np.ndarray, 
        sample_rate: int,
        strength: float = 1.0
    ) -> np.ndarray:
        """Noise reduction stage"""
        # Delegate to audio processor
        return self.audio_processor._apply_noise_reduction(audio, sample_rate)
    
    async def _normalize(
        self,
        audio: np.ndarray,
        sample_rate: int,
        target_db: float = -20.0
    ) -> np.ndarray:
        """Normalization stage"""
        return self.audio_processor._normalize_audio(audio, target_db)
    
    async def _eq_enhancement(
        self,
        audio: np.ndarray,
        sample_rate: int,
        boost_speech: bool = True
    ) -> np.ndarray:
        """EQ enhancement for speech clarity"""
        if not boost_speech:
            return audio
        
        # Simple speech frequency boost (1-4 kHz)
        from scipy import signal
        
        # Design peaking EQ filter
        center_freq = 2500  # Hz
        Q = 2.0
        gain_db = 3.0
        
        w0 = 2 * np.pi * center_freq / sample_rate
        A = 10**(gain_db / 40)
        
        # Peaking EQ coefficients
        alpha = np.sin(w0) / (2 * Q)
        
        b0 = 1 + alpha * A
        b1 = -2 * np.cos(w0)
        b2 = 1 - alpha * A
        a0 = 1 + alpha / A
        a1 = -2 * np.cos(w0)
        a2 = 1 - alpha / A
        
        # Normalize coefficients
        b = np.array([b0, b1, b2]) / a0
        a = np.array([1, a1/a0, a2/a0])
        
        # Apply filter
        enhanced = signal.filtfilt(b, a, audio)
        
        return enhanced
    
    async def _postprocess(
        self,
        audio: np.ndarray,
        sample_rate: int
    ) -> np.ndarray:
        """Post-processing stage"""
        # Final limiting to prevent clipping
        audio = np.clip(audio, -0.99, 0.99)
        
        # Smooth any remaining discontinuities
        from scipy.ndimage import median_filter
        audio = median_filter(audio, size=3)
        
        return audio
    
    def _apply_settings(self, settings: Dict):
        """Apply settings to pipeline stages"""
        # Update stage parameters based on settings
        for stage in self.stages:
            if stage.name == "Noise Reduction":
                stage.enabled = settings.get('noise_reduction', True)
                if 'noise_reduction_strength' in settings:
                    stage.params['strength'] = settings['noise_reduction_strength']
            
            elif stage.name == "Normalization":
                stage.enabled = settings.get('normalize', True)
                if 'target_db' in settings:
                    stage.params['target_db'] = settings['target_db']
            
            elif stage.name == "EQ Enhancement":
                stage.enabled = settings.get('eq_enhancement', True)
    
    def add_stage(self, stage: PipelineStage, position: Optional[int] = None):
        """Add a custom stage to the pipeline"""
        if position is None:
            self.stages.append(stage)
        else:
            self.stages.insert(position, stage)
    
    def remove_stage(self, name: str):
        """Remove a stage from the pipeline"""
        self.stages = [s for s in self.stages if s.name != name]
    
    def get_stage_info(self) -> List[Dict]:
        """Get information about all pipeline stages"""
        return [
            {
                'name': stage.name,
                'enabled': stage.enabled,
                'params': stage.params
            }
            for stage in self.stages
        ] 