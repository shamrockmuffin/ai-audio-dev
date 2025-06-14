import pytest
import numpy as np
import tempfile
import os
from pathlib import Path
import asyncio

from core.professional_audio_processor import ProfessionalAudioProcessor, AudioMetrics
from core.security.audio_security import AudioFileValidator
from utils.cache_manager import AudioCacheManager

class TestProfessionalIntegration:
    """Integration tests for professional audio features"""
    
    @pytest.fixture
    def sample_audio(self):
        """Generate sample audio for testing"""
        duration = 2.0  # 2 seconds
        sample_rate = 48000
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Generate stereo sine wave
        frequency = 440  # A4
        left = np.sin(2 * np.pi * frequency * t) * 0.5
        right = np.sin(2 * np.pi * frequency * t * 1.01) * 0.5  # Slightly detuned
        
        return np.array([left, right]), sample_rate
    
    @pytest.fixture
    def temp_audio_file(self, sample_audio):
        """Create temporary audio file"""
        audio, sr = sample_audio
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            import soundfile as sf
            sf.write(f.name, audio.T, sr)
            yield f.name
        
        # Cleanup
        os.unlink(f.name)
    
    def test_professional_processor_initialization(self):
        """Test professional audio processor initialization"""
        processor = ProfessionalAudioProcessor(
            target_sample_rate=48000,
            target_lufs=-16.0,
            max_true_peak=-1.0
        )
        
        assert processor.target_sample_rate == 48000
        assert processor.target_lufs == -16.0
        assert processor.max_true_peak == -1.0
        assert processor.meter is not None
    
    @pytest.mark.asyncio
    async def test_audio_processing_pipeline(self, temp_audio_file):
        """Test complete audio processing pipeline"""
        processor = ProfessionalAudioProcessor()
        
        settings = {
            'normalize': True,
            'noise_reduction': False,  # Skip for speed
        }
        
        result = await processor.process_audio_file(
            temp_audio_file,
            settings=settings
        )
        
        assert result.enhanced_audio is not None
        assert result.sample_rate == processor.target_sample_rate
        assert isinstance(result.metrics, AudioMetrics)
        assert len(result.processing_log) > 0
    
    def test_professional_metrics_calculation(self, sample_audio):
        """Test professional metrics calculation"""
        processor = ProfessionalAudioProcessor()
        audio, sr = sample_audio
        
        metrics = processor.calculate_professional_metrics(audio, sr)
        
        assert isinstance(metrics, AudioMetrics)
        assert isinstance(metrics.lufs_integrated, float)
        assert isinstance(metrics.true_peak, float)
        assert isinstance(metrics.dynamic_range, float)
        assert isinstance(metrics.phase_coherence, float)
        assert isinstance(metrics.clip_count, int)
        assert isinstance(metrics.frequency_response, dict)
        
        # Check reasonable values
        assert -100 < metrics.lufs_integrated < 0  # LUFS should be negative
        assert 0 <= metrics.phase_coherence <= 1   # Coherence 0-1
        assert metrics.clip_count >= 0             # Non-negative
    
    @pytest.mark.asyncio
    async def test_security_validation(self, temp_audio_file):
        """Test security validation"""
        validator = AudioFileValidator(
            max_file_size_mb=100,
            allowed_formats=['wav', 'mp3', 'flac'],
            enable_deep_scan=True
        )
        
        result = await validator.validate_file(
            temp_audio_file,
            original_filename="test.wav"
        )
        
        assert result.is_safe == True
        assert result.file_type == 'wav'
        assert result.file_size > 0
        assert result.duration > 0
        assert result.sample_rate > 0
        assert len(result.hash_sha256) == 64  # SHA256 hex length
        assert len(result.issues) == 0
    
    @pytest.mark.asyncio
    async def test_security_validation_failure(self):
        """Test security validation with invalid file"""
        validator = AudioFileValidator()
        
        # Test with non-existent file
        result = await validator.validate_file("nonexistent.wav")
        
        assert result.is_safe == False
        assert "File does not exist" in result.issues
    
    def test_cache_manager_functionality(self, temp_audio_file):
        """Test cache manager"""
        cache_manager = AudioCacheManager(cache_dir=".test_cache")
        
        # Test cache key generation
        settings = {'normalize': True}
        cache_key = cache_manager.get_cache_key(temp_audio_file, settings)
        assert cache_key is not None
        assert len(cache_key) > 0
        
        # Test caching and retrieval
        test_data = {'result': 'test_value'}
        cache_manager.cache_result(cache_key, test_data, "test")
        
        retrieved = cache_manager.get_cached_result(cache_key, "test")
        assert retrieved == test_data
        
        # Test cache stats
        stats = cache_manager.get_cache_stats()
        assert isinstance(stats, dict)
        assert 'total_files' in stats
        assert 'total_size_mb' in stats
        
        # Cleanup
        cache_manager.clear_cache()
        
        # Remove test cache directory
        import shutil
        if Path(".test_cache").exists():
            shutil.rmtree(".test_cache")
    
    def test_metrics_serialization(self, sample_audio):
        """Test that metrics can be serialized (for caching)"""
        processor = ProfessionalAudioProcessor()
        audio, sr = sample_audio
        
        metrics = processor.calculate_professional_metrics(audio, sr)
        
        # Test JSON serialization
        import json
        from dataclasses import asdict
        
        metrics_dict = asdict(metrics)
        json_str = json.dumps(metrics_dict)
        
        # Should not raise exception
        assert len(json_str) > 0
        
        # Test deserialization
        restored_dict = json.loads(json_str)
        assert restored_dict['lufs_integrated'] == metrics.lufs_integrated
    
    @pytest.mark.asyncio
    async def test_large_file_handling(self):
        """Test handling of large files (simulated)"""
        processor = ProfessionalAudioProcessor()
        
        # Create a larger audio array (simulating large file)
        duration = 10.0  # 10 seconds
        sample_rate = 48000
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Stereo audio
        left = np.sin(2 * np.pi * 440 * t) * 0.5
        right = np.sin(2 * np.pi * 440 * t * 1.01) * 0.5
        audio = np.array([left, right])
        
        # Test metrics calculation on larger audio
        metrics = processor.calculate_professional_metrics(audio, sample_rate)
        
        assert isinstance(metrics, AudioMetrics)
        assert metrics.duration > 5.0  # Should be around 10 seconds
    
    def test_error_handling(self):
        """Test error handling in various components"""
        processor = ProfessionalAudioProcessor()
        
        # Test with invalid audio data
        invalid_audio = np.array([])
        
        try:
            metrics = processor.calculate_professional_metrics(invalid_audio, 48000)
            # Should handle gracefully or raise appropriate exception
        except Exception as e:
            # Should be a meaningful error
            assert len(str(e)) > 0

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 