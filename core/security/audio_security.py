import os
import hashlib
import magic
import tempfile
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import librosa
import numpy as np
from pydantic import BaseModel

logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """Result of audio file validation"""
    is_safe: bool
    file_type: str
    file_size: int
    duration: Optional[float]
    sample_rate: Optional[int]
    channels: Optional[int]
    hash_sha256: str
    issues: List[str]
    metadata: Dict[str, Any]

class AudioFileValidator:
    """Comprehensive audio file security validator"""
    
    def __init__(
        self,
        max_file_size_mb: int = 100,
        allowed_formats: List[str] = None,
        enable_deep_scan: bool = True,
        max_duration_seconds: int = 3600
    ):
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024
        self.allowed_formats = allowed_formats or ['wav', 'mp3', 'flac', 'ogg', 'm4a']
        self.enable_deep_scan = enable_deep_scan
        self.max_duration_seconds = max_duration_seconds
        
        # Dangerous file signatures to detect
        self.dangerous_signatures = {
            b'\x4D\x5A': 'PE executable',
            b'\x7F\x45\x4C\x46': 'ELF executable',
            b'\xCA\xFE\xBA\xBE': 'Mach-O executable',
            b'\x50\x4B\x03\x04': 'ZIP archive',
            b'\x52\x61\x72\x21': 'RAR archive',
        }
    
    async def validate_file(
        self, 
        file_path: str, 
        original_filename: str = None
    ) -> ValidationResult:
        """Comprehensive file validation"""
        issues = []
        metadata = {}
        
        try:
            # Basic file checks
            if not os.path.exists(file_path):
                return ValidationResult(
                    is_safe=False,
                    file_type='unknown',
                    file_size=0,
                    duration=None,
                    sample_rate=None,
                    channels=None,
                    hash_sha256='',
                    issues=['File does not exist'],
                    metadata={}
                )
            
            # File size check
            file_size = os.path.getsize(file_path)
            if file_size > self.max_file_size_bytes:
                issues.append(f'File too large: {file_size / 1024 / 1024:.1f}MB > {self.max_file_size_bytes / 1024 / 1024}MB')
            
            if file_size == 0:
                issues.append('Empty file')
            
            # Calculate file hash
            hash_sha256 = self._calculate_file_hash(file_path)
            
            # MIME type detection
            mime_type = magic.from_file(file_path, mime=True)
            file_type = self._extract_file_type(mime_type, original_filename)
            
            if file_type not in self.allowed_formats:
                issues.append(f'Unsupported file type: {file_type}')
            
            # Check for dangerous file signatures
            dangerous_sig = self._check_dangerous_signatures(file_path)
            if dangerous_sig:
                issues.append(f'Dangerous file signature detected: {dangerous_sig}')
            
            # Audio-specific validation
            duration = None
            sample_rate = None
            channels = None
            
            if self.enable_deep_scan and not issues:
                try:
                    # Load audio metadata without loading full audio
                    y, sr = librosa.load(file_path, sr=None, duration=1.0)  # Load only 1 second
                    
                    # Get full duration without loading entire file
                    duration = librosa.get_duration(path=file_path)
                    sample_rate = sr
                    channels = 1 if len(y.shape) == 1 else y.shape[0]
                    
                    # Duration check
                    if duration > self.max_duration_seconds:
                        issues.append(f'Audio too long: {duration:.1f}s > {self.max_duration_seconds}s')
                    
                    # Sample rate validation
                    if sample_rate < 8000 or sample_rate > 192000:
                        issues.append(f'Unusual sample rate: {sample_rate}Hz')
                    
                    # Check for audio anomalies
                    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
                        issues.append('Audio contains invalid values (NaN/Inf)')
                    
                    # Check dynamic range
                    if np.max(np.abs(y)) < 0.001:
                        issues.append('Audio signal too quiet (possible silence)')
                    
                    metadata.update({
                        'peak_amplitude': float(np.max(np.abs(y))),
                        'rms_level': float(np.sqrt(np.mean(y**2))),
                        'zero_crossings': int(np.sum(np.diff(np.signbit(y))))
                    })
                    
                except Exception as e:
                    issues.append(f'Audio validation failed: {str(e)}')
            
            # Filename validation
            if original_filename:
                filename_issues = self._validate_filename(original_filename)
                issues.extend(filename_issues)
            
            return ValidationResult(
                is_safe=len(issues) == 0,
                file_type=file_type,
                file_size=file_size,
                duration=duration,
                sample_rate=sample_rate,
                channels=channels,
                hash_sha256=hash_sha256,
                issues=issues,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Validation error: {e}")
            return ValidationResult(
                is_safe=False,
                file_type='unknown',
                file_size=0,
                duration=None,
                sample_rate=None,
                channels=None,
                hash_sha256='',
                issues=[f'Validation error: {str(e)}'],
                metadata={}
            )
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA256 hash of file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    
    def _extract_file_type(self, mime_type: str, filename: str = None) -> str:
        """Extract file type from MIME type and filename"""
        mime_to_ext = {
            'audio/wav': 'wav',
            'audio/wave': 'wav',
            'audio/x-wav': 'wav',
            'audio/mpeg': 'mp3',
            'audio/mp3': 'mp3',
            'audio/flac': 'flac',
            'audio/ogg': 'ogg',
            'audio/x-m4a': 'm4a',
            'audio/mp4': 'm4a',
        }
        
        file_type = mime_to_ext.get(mime_type, 'unknown')
        
        # Fallback to filename extension
        if file_type == 'unknown' and filename:
            ext = Path(filename).suffix.lower().lstrip('.')
            if ext in self.allowed_formats:
                file_type = ext
        
        return file_type
    
    def _check_dangerous_signatures(self, file_path: str) -> Optional[str]:
        """Check for dangerous file signatures"""
        try:
            with open(file_path, 'rb') as f:
                header = f.read(16)
                
            for signature, description in self.dangerous_signatures.items():
                if header.startswith(signature):
                    return description
                    
        except Exception:
            pass
        
        return None
    
    def _validate_filename(self, filename: str) -> List[str]:
        """Validate filename for security issues"""
        issues = []
        
        # Check for path traversal
        if '..' in filename or '/' in filename or '\\' in filename:
            issues.append('Filename contains path traversal characters')
        
        # Check for dangerous characters
        dangerous_chars = ['<', '>', ':', '"', '|', '?', '*', '\0']
        if any(char in filename for char in dangerous_chars):
            issues.append('Filename contains dangerous characters')
        
        # Check length
        if len(filename) > 255:
            issues.append('Filename too long')
        
        # Check for hidden files
        if filename.startswith('.'):
            issues.append('Hidden file detected')
        
        return issues 