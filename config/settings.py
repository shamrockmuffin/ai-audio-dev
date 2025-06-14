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
    
    HUGGING_FACE_TOKEN: str = Field(
        default=os.getenv("HUGGING_FACE_TOKEN", ""),
        description="Hugging Face token for accessing gated models"
    )
    
    # Model configurations
    WHISPER_MODEL: str = Field(
        default="openai/whisper-base",
        description="Whisper model for transcription"
    )
    
    CLAUDE_MODEL: str = Field(
        default="claude-3-5-sonnet-latest",
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
    
    GPU_DEVICE_ID: int = Field(
        default=0,
        description="GPU device ID to use (0 for RTX 4070, 1 for RTX 3060)"
    )
    
    USE_FP16: bool = Field(
        default=False,
        description="Use FP16 precision for faster GPU inference (disabled by default to avoid precision issues)"
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