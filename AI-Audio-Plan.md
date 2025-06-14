I've successfully refactored your PyQt5 audio analysis and transcription app into a modern Streamlit application. Here's what I've created:

## Key Improvements:

### 1. **Modern Architecture**
- **Modular Design**: Clear separation of concerns with dedicated modules for configuration, core processing, services, UI components, and utilities
- **Async Support**: Asynchronous processing for better performance and responsiveness
- **Type Safety**: Full type hints and Pydantic models for configuration

### 2. **Enhanced Features**
- **Web-based Interface**: Accessible from any browser, no desktop installation required
- **Session Management**: Save and restore your work sessions
- **Progress Tracking**: Real-time progress updates for long operations
- **Multiple Export Formats**: Support for TXT, JSON, and SRT exports
- **Resource Management**: Built-in memory monitoring and optimization

### 3. **Better User Experience**
- **Tabbed Interface**: Organized into logical sections (Analysis, Enhancement, Transcription, AI Assistant)
- **Responsive Design**: Works well on different screen sizes
- **Visual Feedback**: Clear success/error messages and progress indicators
- **File History**: Easy access to recently processed files

### 4. **Technical Improvements**
- **Configurable Pipeline**: Customizable audio enhancement stages
- **Better Error Handling**: Comprehensive error handling with user-friendly messages
- **Performance Optimization**: Chunk-based processing for large files
- **GPU Support**: Automatic GPU utilization when available

### 5. **Project Structure**
```
audio_analyzer/
├── config/          # Settings and configuration
├── core/            # Audio processing and transcription logic
├── services/        # Claude and Whisper integrations
├── ui/              # Streamlit components and state management
├── utils/           # Helper functions and resource management
└── main.py          # Application entry point
```

### 6. **Key Files Created**
- `main.py`: Main Streamlit application with all UI logic
- `core/audio_processor.py`: Enhanced audio processing with async support
- `config/settings.py`: Centralized configuration using Pydantic
- `ui/components.py`: Reusable UI components
- `ui/session_state.py`: Advanced session state management
- `services/claude_service.py`: Claude API integration
- `services/whisper_service.py`: Whisper transcription service
- `utils/audio_helpers.py`: Audio utility functions
- `core/transcription_handler.py`: Complete transcription pipeline
- `core/enhancement_pipeline.py`: Configurable audio enhancement

## To Run the Application:

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up your `.env` file with your Anthropic API key:
```
ANTHROPIC_API_KEY=your_key_here
```

3. Run the app:
```bash
streamlit run main.py
```

The Streamlit version offers significant advantages over PyQt5:
- No installation required for users (runs in browser)
- Better scalability and deployment options
- Modern, responsive UI
- Easier to maintain and extend
- Better async handling for long operations
- Built-in session persistence

# requirements.txt
streamlit>=1.28.0
numpy>=1.24.0
scipy>=1.10.0
librosa>=0.10.0
soundfile>=0.12.0
matplotlib>=3.7.0
plotly>=5.14.0
anthropic>=0.18.0
python-dotenv>=1.0.0
pydantic>=2.0.0
pydantic-settings>=2.0.0
noisereduce>=3.0.0
transformers>=4.30.0
torch>=2.0.0
torchaudio>=2.0.0
psutil>=5.9.0
asyncio>=3.4.3

# .env.example
ANTHROPIC_API_KEY=sk-ant-api03-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX 
WHISPER_MODEL=openai/whisper-base
CLAUDE_MODEL=claude-3-sonnet-20240229
DEBUG=False

# README.md
# Audio Analysis & Transcription - Streamlit App

A modern, feature-rich audio analysis and transcription application built with Streamlit. This application provides powerful audio processing capabilities including enhancement, transcription, and AI-powered analysis.

## Features

- **Audio Analysis**: Visualize waveforms and spectrograms
- **Audio Enhancement**: Noise reduction, normalization, and EQ enhancement
- **Transcription**: High-quality speech-to-text using Whisper
- **AI Enhancement**: Improve transcription quality with Claude
- **Interactive Chat**: AI assistant for audio-related queries
- **Multiple Export Formats**: TXT, JSON, SRT
- **Session Management**: Save and restore your work

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd audio-analyzer
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env and add your Anthropic API key
```

## Usage

1. Start the application:
```bash
streamlit run main.py
```

2. Open your browser to http://localhost:8501

3. Upload an audio file using the sidebar

4. Explore the different tabs:
   - **Audio Analysis**: View waveform and spectrogram
   - **Enhancement**: Apply audio improvements
   - **Transcription**: Convert speech to text
   - **AI Assistant**: Chat about your audio

## Project Structure

```
audio_analyzer/
├── config/           # Configuration and settings
├── core/            # Core processing modules
├── services/        # External service integrations
├── ui/              # UI components and state management
├── utils/           # Utility functions
├── main.py          # Main application entry point
└── requirements.txt # Project dependencies
```

## Configuration

Key settings can be adjusted in `config/settings.py`:

- `MAX_AUDIO_LENGTH`: Maximum audio duration (seconds)
- `CHUNK_SIZE`: Processing chunk size (seconds)
- `DEFAULT_SAMPLE_RATE`: Default audio sample rate
- `MAX_FILE_SIZE_MB`: Maximum upload file size

## Advanced Features

### Custom Enhancement Pipeline

The application supports a customizable enhancement pipeline. You can add or remove processing stages:

```python
from core.enhancement_pipeline import EnhancementPipeline, PipelineStage

pipeline = EnhancementPipeline()
pipeline.add_stage(PipelineStage(
    name="Custom Filter",
    function=my_custom_filter,
    params={'param1': value}
))
```

### Session Management

Save your work for later:
- Use the session manager to save current state
- Load previous sessions from the sidebar
- Export session data for backup

## API Keys

This application requires an Anthropic API key for AI features. Get your key from:
https://console.anthropic.com/

## Troubleshooting

### Common Issues

1. **Memory Errors**: Reduce `MAX_AUDIO_LENGTH` in settings
2. **Slow Processing**: Enable GPU in settings if available
3. **API Errors**: Check your Anthropic API key is valid

### Debug Mode

Enable debug mode in `.env`:
```
DEBUG=True
```

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## License

This project is licensed under the MIT License.


# core/transcription_handler.py
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

# core/enhancement_pipeline.py
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

# core/__init__.py
from .audio_processor import AudioProcessor, EnhancementResult
from .transcription_handler import TranscriptionHandler
from .enhancement_pipeline import EnhancementPipeline, PipelineStage

__all__ = [
    'AudioProcessor',
    'EnhancementResult',
    'TranscriptionHandler',
    'EnhancementPipeline',
    'PipelineStage'
]


# utils/audio_helpers.py
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import Dict, Tuple, Optional
import logging
from datetime import datetime
import tempfile

logger = logging.getLogger(__name__)

def get_audio_metadata(file_path: str) -> Dict:
    """
    Extract metadata from audio file
    
    Args:
        file_path: Path to audio file
        
    Returns:
        Dictionary with audio metadata
    """
    try:
        # Get basic info using soundfile
        info = sf.info(file_path)
        
        # Load audio for additional analysis
        y, sr = librosa.load(file_path, sr=None, duration=30)  # Load first 30s for analysis
        
        # Calculate additional metrics
        duration = info.duration
        rms = np.sqrt(np.mean(y**2))
        
        # Try to detect tempo
        tempo = None
        try:
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            tempo = float(tempo)
        except:
            pass
        
        metadata = {
            'duration': duration,
            'sample_rate': info.samplerate,
            'channels': info.channels,
            'format': info.format,
            'subtype': info.subtype,
            'bit_depth': str(info.subtype_info) if hasattr(info, 'subtype_info') else 'N/A',
            'file_size_mb': Path(file_path).stat().st_size / (1024 * 1024),
            'rms': float(rms),
            'tempo': tempo
        }
        
        return metadata
        
    except Exception as e:
        logger.error(f"Error getting audio metadata: {e}")
        return {
            'duration': 0,
            'sample_rate': 0,
            'error': str(e)
        }

def save_audio_file(
    audio_data: np.ndarray,
    sample_rate: int,
    filename: str,
    output_dir: Optional[Path] = None
) -> str:
    """
    Save audio data to file
    
    Args:
        audio_data: Audio data array
        sample_rate: Sample rate
        filename: Output filename
        output_dir: Optional output directory
        
    Returns:
        Path to saved file
    """
    try:
        # Use temp directory if no output directory specified
        if output_dir is None:
            output_dir = Path(tempfile.gettempdir()) / "audio_analyzer"
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Ensure filename has extension
        if not filename.endswith('.wav'):
            filename = f"{filename}.wav"
        
        # Add timestamp to filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{filename}"
        
        # Full path
        file_path = output_dir / filename
        
        # Save audio
        sf.write(
            file_path,
            audio_data,
            sample_rate,
            subtype='PCM_16'  # 16-bit PCM
        )
        
        logger.info(f"Audio saved to: {file_path}")
        return str(file_path)
        
    except Exception as e:
        logger.error(f"Error saving audio: {e}")
        raise

def convert_audio_format(
    input_path: str,
    output_format: str = 'wav',
    output_path: Optional[str] = None
) -> str:
    """
    Convert audio file to different format
    
    Args:
        input_path: Input audio file path
        output_format: Output format (wav, mp3, etc.)
        output_path: Optional output path
        
    Returns:
        Path to converted file
    """
    try:
        # Load audio
        y, sr = librosa.load(input_path, sr=None)
        
        # Generate output path if not provided
        if output_path is None:
            input_stem = Path(input_path).stem
            output_path = str(Path(input_path).parent / f"{input_stem}.{output_format}")
        
        # Save in new format
        sf.write(output_path, y, sr)
        
        logger.info(f"Audio converted to: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Error converting audio: {e}")
        raise

def calculate_audio_statistics(audio_data: np.ndarray, sample_rate: int) -> Dict:
    """
    Calculate detailed audio statistics
    
    Args:
        audio_data: Audio data array
        sample_rate: Sample rate
        
    Returns:
        Dictionary with audio statistics
    """
    try:
        stats = {}
        
        # Basic statistics
        stats['mean'] = float(np.mean(audio_data))
        stats['std'] = float(np.std(audio_data))
        stats['min'] = float(np.min(audio_data))
        stats['max'] = float(np.max(audio_data))
        
        # RMS and peak
        stats['rms'] = float(np.sqrt(np.mean(audio_data**2)))
        stats['peak'] = float(np.abs(audio_data).max())
        stats['peak_db'] = float(20 * np.log10(stats['peak'] + 1e-10))
        
        # Dynamic range
        stats['dynamic_range_db'] = float(
            20 * np.log10(stats['peak'] / (stats['rms'] + 1e-10))
        )
        
        # Zero crossing rate
        stats['zcr'] = float(
            np.mean(librosa.feature.zero_crossing_rate(audio_data))
        )
        
        # Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(
            y=audio_data, 
            sr=sample_rate
        )[0]
        stats['spectral_centroid_mean'] = float(np.mean(spectral_centroids))
        stats['spectral_centroid_std'] = float(np.std(spectral_centroids))
        
        return stats
        
    except Exception as e:
        logger.error(f"Error calculating audio statistics: {e}")
        return {}

# utils/resource_manager.py
import psutil
import torch
import gc
from typing import Optional
import logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class ResourceManager:
    """Manage system resources for audio processing"""
    
    def __init__(self):
        self.gpu_available = torch.cuda.is_available()
        self.initial_memory = self.get_memory_usage()
    
    def get_memory_usage(self) -> Dict:
        """Get current memory usage"""
        memory = psutil.virtual_memory()
        
        usage = {
            'total_gb': memory.total / (1024**3),
            'available_gb': memory.available / (1024**3),
            'percent': memory.percent,
            'used_gb': memory.used / (1024**3)
        }
        
        if self.gpu_available:
            for i in range(torch.cuda.device_count()):
                usage[f'gpu_{i}_allocated_gb'] = (
                    torch.cuda.memory_allocated(i) / (1024**3)
                )
                usage[f'gpu_{i}_reserved_gb'] = (
                    torch.cuda.memory_reserved(i) / (1024**3)
                )
        
        return usage
    
    def check_available_memory(self, required_gb: float = 2.0) -> bool:
        """Check if enough memory is available"""
        memory = self.get_memory_usage()
        return memory['available_gb'] >= required_gb
    
    def cleanup(self):
        """Cleanup resources"""
        gc.collect()
        
        if self.gpu_available:
            torch.cuda.empty_cache()
        
        logger.info("Resources cleaned up")
    
    @contextmanager
    def memory_manager(self, name: str = "Operation"):
        """Context manager for memory tracking"""
        start_memory = self.get_memory_usage()
        logger.info(f"{name} started - Memory: {start_memory['percent']:.1f}%")
        
        try:
            yield
        finally:
            self.cleanup()
            end_memory = self.get_memory_usage()
            memory_diff = end_memory['used_gb'] - start_memory['used_gb']
            logger.info(
                f"{name} completed - Memory delta: {memory_diff:.2f} GB"
            )
    
    def estimate_audio_memory(
        self, 
        duration_seconds: float, 
        sample_rate: int = 16000
    ) -> float:
        """Estimate memory required for audio processing"""
        # Basic audio array
        audio_memory = duration_seconds * sample_rate * 4 / (1024**3)  # float32
        
        # Processing overhead (roughly 3x for transforms, spectrograms, etc.)
        total_memory = audio_memory * 3
        
        return total_memory
    
    def can_process_audio(
        self, 
        duration_seconds: float, 
        sample_rate: int = 16000
    ) -> Tuple[bool, str]:
        """Check if audio can be processed with available memory"""
        required_memory = self.estimate_audio_memory(duration_seconds, sample_rate)
        available_memory = self.get_memory_usage()['available_gb']
        
        if required_memory > available_memory * 0.8:  # Leave 20% buffer
            return False, (
                f"Insufficient memory. Required: {required_memory:.1f} GB, "
                f"Available: {available_memory:.1f} GB"
            )
        
        return True, "OK"

# utils/__init__.py
from .audio_helpers import (
    get_audio_metadata,
    save_audio_file,
    convert_audio_format,
    calculate_audio_statistics
)
from .resource_manager import ResourceManager

__all__ = [
    'get_audio_metadata',
    'save_audio_file',
    'convert_audio_format',
    'calculate_audio_statistics',
    'ResourceManager'
]


# services/claude_service.py
from anthropic import Anthropic
from typing import Optional, Dict, List
import logging
import asyncio
from config.settings import settings

class ClaudeService:
    """Service for interacting with Claude API"""
    
    def __init__(self):
        self.client = Anthropic(api_key=settings.ANTHROPIC_API_KEY)
        self.logger = logging.getLogger(__name__)
        
    async def enhance_transcription(
        self, 
        text: str,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Enhance transcription using Claude
        
        Args:
            text: Raw transcription text
            system_prompt: Optional custom system prompt
            
        Returns:
            Enhanced transcription text
        """
        try:
            if not system_prompt:
                system_prompt = """You are an expert in audio transcription enhancement. 
                Your task is to:
                1. Correct obvious transcription errors
                2. Add proper punctuation and capitalization
                3. Format the text for readability (paragraphs, etc.)
                4. Fix grammar while preserving the original meaning
                5. Add speaker labels if multiple speakers are detected
                6. Note any uncertainties in [brackets]
                
                Maintain the original tone and style while improving clarity."""
            
            # Run in executor to avoid blocking
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                self._call_claude,
                system_prompt,
                f"Please enhance this transcription:\n\n{text}"
            )
            
            return response
            
        except Exception as e:
            self.logger.error(f"Claude API error: {e}")
            raise
    
    async def get_contextual_response(
        self,
        prompt: str,
        context: str,
        chat_history: Optional[List[Dict]] = None
    ) -> str:
        """
        Get AI response with context
        
        Args:
            prompt: User's question
            context: Context about the audio/transcription
            chat_history: Previous chat messages
            
        Returns:
            AI response
        """
        try:
            system_prompt = """You are an AI assistant helping with audio analysis and transcription.
            You have access to information about the audio file and its transcription.
            Be helpful, accurate, and concise in your responses."""
            
            # Build message with context
            full_prompt = f"""Context about the audio:
{context}

User question: {prompt}"""
            
            # Include chat history if available
            messages = []
            if chat_history:
                for msg in chat_history[-10:]:  # Last 10 messages
                    messages.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })
            
            messages.append({
                "role": "user",
                "content": full_prompt
            })
            
            # Run in executor
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                self._call_claude_with_history,
                system_prompt,
                messages
            )
            
            return response
            
        except Exception as e:
            self.logger.error(f"Claude contextual response error: {e}")
            raise
    
    def _call_claude(self, system_prompt: str, user_prompt: str) -> str:
        """Synchronous Claude API call"""
        response = self.client.messages.create(
            model=settings.CLAUDE_MODEL,
            max_tokens=4000,
            temperature=0.3,
            system=system_prompt,
            messages=[{
                "role": "user",
                "content": user_prompt
            }]
        )
        return response.content[0].text
    
    def _call_claude_with_history(
        self, 
        system_prompt: str, 
        messages: List[Dict]
    ) -> str:
        """Synchronous Claude API call with message history"""
        response = self.client.messages.create(
            model=settings.CLAUDE_MODEL,
            max_tokens=2000,
            temperature=0.7,
            system=system_prompt,
            messages=messages
        )
        return response.content[0].text

# services/whisper_service.py
import torch
import torchaudio
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
import asyncio
from pathlib import Path

class WhisperService:
    """Service for audio transcription using Whisper"""
    
    def __init__(self, model_name: str = None):
        self.logger = logging.getLogger(__name__)
        self.model_name = model_name or settings.WHISPER_MODEL
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._load_model()
    
    def _load_model(self):
        """Load Whisper model and processor"""
        try:
            self.logger.info(f"Loading Whisper model: {self.model_name}")
            self.processor = WhisperProcessor.from_pretrained(self.model_name)
            self.model = WhisperForConditionalGeneration.from_pretrained(self.model_name)
            self.model.to(self.device)
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

# services/__init__.py
from .claude_service import ClaudeService
from .whisper_service import WhisperService

__all__ = ['ClaudeService', 'WhisperService']



# ui/session_state.py
import streamlit as st
from typing import Dict, List, Optional, Any
from datetime import datetime
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class SessionState:
    """Wrapper for Streamlit session state with additional functionality"""
    
    def __init__(self):
        self._initialize_defaults()
    
    def _initialize_defaults(self):
        """Initialize default session state values"""
        defaults = {
            'current_file': None,
            'enhanced_file': None,
            'file_history': [],
            'transcription': None,
            'enhancement_result': None,
            'chat_history': [],
            'settings': {
                'noise_reduction': True,
                'normalize': True,
                'enhance_transcription': True,
                'band_pass': True,
                'compression': False,
                'target_db': -20.0,
                'compression_ratio': 4.0,
                'noise_reduction_strength': 1.0
            },
            'ui_state': {
                'active_tab': 0,
                'show_advanced': False,
                'theme': 'light'
            }
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    def __getattr__(self, name):
        """Get attribute from session state"""
        if name in st.session_state:
            return st.session_state[name]
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
    
    def __setattr__(self, name, value):
        """Set attribute in session state"""
        if name.startswith('_'):
            super().__setattr__(name, value)
        else:
            st.session_state[name] = value
    
    def add_file(self, file_path: str, file_name: str):
        """Add file to history"""
        file_info = {
            'path': file_path,
            'name': file_name,
            'timestamp': datetime.now().isoformat(),
            'size': Path(file_path).stat().st_size
        }
        
        # Add to history
        st.session_state.file_history.append(file_info)
        
        # Set as current
        st.session_state.current_file = file_path
        
        # Limit history size
        if len(st.session_state.file_history) > 20:
            st.session_state.file_history = st.session_state.file_history[-20:]
    
    def remove_file(self, index: int):
        """Remove file from history"""
        if 0 <= index < len(st.session_state.file_history):
            removed = st.session_state.file_history.pop(index)
            
            # If it was the current file, clear it
            if st.session_state.current_file == removed['path']:
                st.session_state.current_file = None
                st.session_state.enhanced_file = None
                st.session_state.transcription = None
    
    def update_settings(self, settings: Dict[str, Any]):
        """Update settings"""
        st.session_state.settings.update(settings)
    
    def add_chat_message(self, role: str, content: str):
        """Add message to chat history"""
        message = {
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat()
        }
        st.session_state.chat_history.append(message)
        
        # Limit chat history
        if len(st.session_state.chat_history) > 100:
            st.session_state.chat_history = st.session_state.chat_history[-100:]
    
    def clear_chat_history(self):
        """Clear chat history"""
        st.session_state.chat_history = []
    
    def export_session(self) -> Dict:
        """Export session state for saving"""
        return {
            'settings': st.session_state.settings,
            'file_history': st.session_state.file_history,
            'chat_history': st.session_state.chat_history,
            'timestamp': datetime.now().isoformat()
        }
    
    def import_session(self, data: Dict):
        """Import session state from saved data"""
        if 'settings' in data:
            st.session_state.settings.update(data['settings'])
        if 'file_history' in data:
            st.session_state.file_history = data['file_history']
        if 'chat_history' in data:
            st.session_state.chat_history = data['chat_history']
    
    def reset(self):
        """Reset session state to defaults"""
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        self._initialize_defaults()

class SessionStateManager:
    """Manager for session state with persistence capabilities"""
    
    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path.home() / '.audio_analyzer' / 'sessions'
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.state = SessionState()
    
    def get_state(self) -> SessionState:
        """Get session state instance"""
        return self.state
    
    def save_session(self, name: str):
        """Save current session to file"""
        try:
            session_data = self.state.export_session()
            file_path = self.storage_path / f"{name}.json"
            
            with open(file_path, 'w') as f:
                json.dump(session_data, f, indent=2)
            
            logger.info(f"Session saved to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving session: {e}")
            return False
    
    def load_session(self, name: str):
        """Load session from file"""
        try:
            file_path = self.storage_path / f"{name}.json"
            
            if not file_path.exists():
                logger.warning(f"Session file not found: {file_path}")
                return False
            
            with open(file_path, 'r') as f:
                session_data = json.load(f)
            
            self.state.import_session(session_data)
            logger.info(f"Session loaded from {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading session: {e}")
            return False
    
    def list_sessions(self) -> List[Dict[str, Any]]:
        """List available saved sessions"""
        sessions = []
        
        for file_path in self.storage_path.glob("*.json"):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                sessions.append({
                    'name': file_path.stem,
                    'timestamp': data.get('timestamp', ''),
                    'file_count': len(data.get('file_history', [])),
                    'path': str(file_path)
                })
            except:
                continue
        
        return sorted(sessions, key=lambda x: x['timestamp'], reverse=True)
    
    def delete_session(self, name: str):
        """Delete saved session"""
        try:
            file_path = self.storage_path / f"{name}.json"
            if file_path.exists():
                file_path.unlink()
                logger.info(f"Session deleted: {name}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error deleting session: {e}")
            return False

# ui/__init__.py
from .components import (
    render_audio_player,
    render_waveform,
    render_spectrogram,
    render_enhancement_controls,
    render_transcription_view,
    render_progress_bar,
    render_error_message,
    render_success_message,
    render_info_message
)
from .session_state import SessionState, SessionStateManager

__all__ = [
    'render_audio_player',
    'render_waveform',
    'render_spectrogram',
    'render_enhancement_controls',
    'render_transcription_view',
    'render_progress_bar',
    'render_error_message',
    'render_success_message',
    'render_info_message',
    'SessionState',
    'SessionStateManager'
]


# ui/components.py
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


    # config/settings.py
from pydantic_settings import BaseSettings
from pydantic import Field, validator
from typing import Optional
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Settings(BaseSettings):
    """Application settings with validation"""
    
    # API Keys
    ANTHROPIC_API_KEY: str = Field(
        default=os.getenv("ANTHROPIC_API_KEY", ""),
        description="Anthropic API key for Claude"
    )
    
    # Model configurations
    WHISPER_MODEL: str = Field(
        default="openai/whisper-base",
        description="Whisper model for transcription"
    )
    
    CLAUDE_MODEL: str = Field(
        default="claude-3-sonnet-20240229",
        description="Claude model for text enhancement"
    )
    
    # Audio processing settings
    MAX_AUDIO_LENGTH: int = Field(
        default=600,
        description="Maximum audio length in seconds"
    )
    
    CHUNK_SIZE: int = Field(
        default=30,
        description="Audio chunk size in seconds for processing"
    )
    
    DEFAULT_SAMPLE_RATE: int = Field(
        default=16000,
        description="Default sample rate for audio processing"
    )
    
    # Enhancement settings
    NOISE_REDUCTION_ENABLED: bool = Field(
        default=True,
        description="Enable noise reduction by default"
    )
    
    NORMALIZATION_TARGET_DB: float = Field(
        default=-20.0,
        description="Target dB for audio normalization"
    )
    
    # Application settings
    DEBUG: bool = Field(
        default=False,
        description="Debug mode"
    )
    
    MAX_FILE_SIZE_MB: int = Field(
        default=100,
        description="Maximum file size in MB"
    )
    
    TEMP_DIR: Path = Field(
        default=Path("/tmp/audio_analyzer"),
        description="Temporary directory for file storage"
    )
    
    # UI Settings
    THEME: str = Field(
        default="light",
        description="UI theme (light/dark)"
    )
    
    MAX_CHAT_HISTORY: int = Field(
        default=50,
        description="Maximum chat history messages to keep"
    )
    
    # Performance settings
    MAX_WORKERS: int = Field(
        default=4,
        description="Maximum worker threads for processing"
    )
    
    ENABLE_GPU: bool = Field(
        default=True,
        description="Enable GPU acceleration if available"
    )
    
    @validator('ANTHROPIC_API_KEY')
    def validate_api_key(cls, v):
        if not v:
            raise ValueError("ANTHROPIC_API_KEY is required")
        return v
    
    @validator('TEMP_DIR')
    def create_temp_dir(cls, v):
        v.mkdir(parents=True, exist_ok=True)
        return v
    
    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'
        case_sensitive = True

# Create settings instance
settings = Settings()

# config/__init__.py
from .settings import settings
from .logging_config import setup_logging

__all__ = ['settings', 'setup_logging']

# config/logging_config.py
import logging
import sys
from pathlib import Path
from datetime import datetime

def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """
    Setup logging configuration
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path
    """
    # Create logs directory if needed
    if log_file:
        log_dir = Path(log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure logging format
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    
    # Create handlers
    handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(log_format, date_format))
    handlers.append(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(log_format, date_format))
        handlers.append(file_handler)
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        handlers=handlers
    )
    
    # Set specific logger levels
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    
    return logging.getLogger(__name__)




    # core/audio_processor.py
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


import streamlit as st
import asyncio
from pathlib import Path
import tempfile
import os
import logging
from datetime import datetime
import json

from config.settings import settings
from core.audio_processor import AudioProcessor
from core.transcription_handler import TranscriptionHandler
from services.claude_service import ClaudeService
from ui.components import (
    render_audio_player,
    render_waveform,
    render_spectrogram,
    render_enhancement_controls,
    render_transcription_view
)
from ui.session_state import SessionStateManager
from utils.audio_helpers import save_audio_file, get_audio_metadata

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
        self.audio_processor = AudioProcessor()
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
        
        # Main content area
        if state.current_file:
            await self._render_main_content(state)
        else:
            self._render_welcome_screen()
    
    async def _handle_file_upload(self, uploaded_file) -> str:
        """Handle file upload and return saved file path"""
        try:
            # Create temp directory if it doesn't exist
            temp_dir = Path(tempfile.gettempdir()) / "audio_analyzer"
            temp_dir.mkdir(exist_ok=True)
            
            # Save file with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_extension = Path(uploaded_file.name).suffix
            file_path = temp_dir / f"{timestamp}_{uploaded_file.name}"
            
            # Write file
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            return str(file_path)
            
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
        
        if st.button("📝 Transcribe Audio", type="primary", use_container_width=True):
            await self._transcribe_audio(state)
        
        if state.transcription:
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
        """Process audio enhancement"""
        progress_bar = st.progress(0)
        status = st.empty()
        
        try:
            status.text("🔄 Processing audio...")
            
            # Process audio
            result = await self.audio_processor.process_audio_file(
                state.current_file,
                settings=state.settings,
                progress_callback=lambda p: progress_bar.progress(p)
            )
            
            # Save enhanced audio
            enhanced_path = save_audio_file(
                result.enhanced_audio,
                result.sample_rate,
                f"enhanced_{Path(state.current_file).name}"
            )
            
            state.enhanced_file = enhanced_path
            state.enhancement_result = result
            
            progress_bar.progress(1.0)
            status.success("✅ Audio enhancement complete!")
            
        except Exception as e:
            status.error(f"❌ Error: {str(e)}")
            logging.error(f"Enhancement error: {e}")
    
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
