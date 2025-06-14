import hashlib
import pickle
import json
from pathlib import Path
from typing import Optional, Any, Dict
import logging
import os
import time

logger = logging.getLogger(__name__)

class AudioCacheManager:
    """Cache manager for audio processing results"""
    
    def __init__(self, cache_dir: str = ".cache/audio", max_cache_size_mb: int = 1000):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_cache_size_bytes = max_cache_size_mb * 1024 * 1024
        
        # Create subdirectories
        (self.cache_dir / "audio").mkdir(exist_ok=True)
        (self.cache_dir / "transcription").mkdir(exist_ok=True)
        (self.cache_dir / "metrics").mkdir(exist_ok=True)
        
        # Cache metadata file
        self.metadata_file = self.cache_dir / "cache_metadata.json"
        self.metadata = self._load_metadata()
    
    def get_cache_key(self, file_path: str, settings: Dict) -> str:
        """Generate cache key from file and settings"""
        try:
            # Get file hash
            with open(file_path, 'rb') as f:
                # Read in chunks for large files
                file_hash = hashlib.sha256()
                while chunk := f.read(8192):
                    file_hash.update(chunk)
                file_hash_str = file_hash.hexdigest()
            
            # Get settings hash
            settings_str = json.dumps(settings, sort_keys=True)
            settings_hash = hashlib.sha256(settings_str.encode()).hexdigest()
            
            return f"{file_hash_str}_{settings_hash}"
            
        except Exception as e:
            logger.error(f"Error generating cache key: {e}")
            return None
    
    def get_cached_result(self, cache_key: str, cache_type: str = "audio") -> Optional[Any]:
        """Retrieve cached result if exists"""
        try:
            cache_file = self.cache_dir / cache_type / f"{cache_key}.pkl"
            
            if cache_file.exists():
                # Check if cache is still valid (24 hours)
                if time.time() - cache_file.stat().st_mtime < 86400:
                    with open(cache_file, 'rb') as f:
                        result = pickle.load(f)
                    
                    # Update access time in metadata
                    self._update_access_time(cache_key, cache_type)
                    return result
                else:
                    # Remove expired cache
                    cache_file.unlink()
                    self._remove_from_metadata(cache_key, cache_type)
            
        except Exception as e:
            logger.error(f"Error retrieving cached result: {e}")
        
        return None
    
    def cache_result(self, cache_key: str, result: Any, cache_type: str = "audio"):
        """Cache processing result"""
        try:
            cache_file = self.cache_dir / cache_type / f"{cache_key}.pkl"
            
            # Check cache size limits
            self._cleanup_cache_if_needed()
            
            # Save result
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)
            
            # Update metadata
            self._add_to_metadata(cache_key, cache_type, cache_file.stat().st_size)
            
            logger.info(f"Cached result: {cache_key} ({cache_type})")
            
        except Exception as e:
            logger.error(f"Error caching result: {e}")
    
    def clear_cache(self, cache_type: str = None):
        """Clear cache (all or specific type)"""
        try:
            if cache_type:
                cache_dir = self.cache_dir / cache_type
                for file in cache_dir.glob("*.pkl"):
                    file.unlink()
                logger.info(f"Cleared {cache_type} cache")
            else:
                for cache_dir in ["audio", "transcription", "metrics"]:
                    for file in (self.cache_dir / cache_dir).glob("*.pkl"):
                        file.unlink()
                logger.info("Cleared all cache")
            
            # Reset metadata
            self.metadata = {}
            self._save_metadata()
            
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        stats = {
            "total_files": 0,
            "total_size_mb": 0,
            "by_type": {}
        }
        
        try:
            for cache_type in ["audio", "transcription", "metrics"]:
                cache_dir = self.cache_dir / cache_type
                files = list(cache_dir.glob("*.pkl"))
                
                type_size = sum(f.stat().st_size for f in files)
                stats["by_type"][cache_type] = {
                    "files": len(files),
                    "size_mb": type_size / (1024 * 1024)
                }
                
                stats["total_files"] += len(files)
                stats["total_size_mb"] += type_size / (1024 * 1024)
        
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
        
        return stats
    
    def _load_metadata(self) -> Dict:
        """Load cache metadata"""
        try:
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Error loading cache metadata: {e}")
        
        return {}
    
    def _save_metadata(self):
        """Save cache metadata"""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving cache metadata: {e}")
    
    def _add_to_metadata(self, cache_key: str, cache_type: str, file_size: int):
        """Add entry to metadata"""
        if cache_type not in self.metadata:
            self.metadata[cache_type] = {}
        
        self.metadata[cache_type][cache_key] = {
            "size": file_size,
            "created": time.time(),
            "accessed": time.time()
        }
        
        self._save_metadata()
    
    def _update_access_time(self, cache_key: str, cache_type: str):
        """Update access time in metadata"""
        if cache_type in self.metadata and cache_key in self.metadata[cache_type]:
            self.metadata[cache_type][cache_key]["accessed"] = time.time()
            self._save_metadata()
    
    def _remove_from_metadata(self, cache_key: str, cache_type: str):
        """Remove entry from metadata"""
        if cache_type in self.metadata and cache_key in self.metadata[cache_type]:
            del self.metadata[cache_type][cache_key]
            self._save_metadata()
    
    def _cleanup_cache_if_needed(self):
        """Clean up cache if it exceeds size limit"""
        try:
            total_size = 0
            all_files = []
            
            # Collect all cache files with metadata
            for cache_type in ["audio", "transcription", "metrics"]:
                cache_dir = self.cache_dir / cache_type
                for file in cache_dir.glob("*.pkl"):
                    file_size = file.stat().st_size
                    total_size += file_size
                    
                    # Get access time from metadata
                    cache_key = file.stem
                    access_time = 0
                    if (cache_type in self.metadata and 
                        cache_key in self.metadata[cache_type]):
                        access_time = self.metadata[cache_type][cache_key]["accessed"]
                    
                    all_files.append({
                        "file": file,
                        "size": file_size,
                        "access_time": access_time,
                        "cache_type": cache_type,
                        "cache_key": cache_key
                    })
            
            # If over limit, remove least recently used files
            if total_size > self.max_cache_size_bytes:
                # Sort by access time (oldest first)
                all_files.sort(key=lambda x: x["access_time"])
                
                removed_size = 0
                target_removal = total_size - (self.max_cache_size_bytes * 0.8)  # Remove to 80% of limit
                
                for file_info in all_files:
                    if removed_size >= target_removal:
                        break
                    
                    # Remove file
                    file_info["file"].unlink()
                    removed_size += file_info["size"]
                    
                    # Remove from metadata
                    self._remove_from_metadata(
                        file_info["cache_key"], 
                        file_info["cache_type"]
                    )
                
                logger.info(f"Cache cleanup: removed {removed_size / (1024*1024):.1f}MB")
        
        except Exception as e:
            logger.error(f"Error during cache cleanup: {e}")

# Global cache manager instance
cache_manager = AudioCacheManager() 