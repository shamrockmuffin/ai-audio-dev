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