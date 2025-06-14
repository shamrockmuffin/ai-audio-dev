import psutil
import torch
import gc
from typing import Optional, Dict, Tuple
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