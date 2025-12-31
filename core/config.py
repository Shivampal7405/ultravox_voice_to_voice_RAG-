"""
Configuration Management
Central configuration using Pydantic Settings
"""
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache


class UltravoxSettings(BaseSettings):
    """Ultravox API settings"""
    api_key: str = Field(..., alias="ULTRAVOX_API_KEY")
    model: str = Field("ultravox-v0.7", alias="ULTRAVOX_MODEL")
    voice: str = Field("Mark", alias="ULTRAVOX_VOICE")
    api_base: str = Field("https://api.ultravox.ai/api", alias="ULTRAVOX_API_BASE")


class AudioSettings(BaseSettings):
    """Audio capture and playback settings"""
    sample_rate: int = Field(16000, alias="AUDIO_SAMPLE_RATE")
    chunk_ms: int = Field(20, alias="AUDIO_CHUNK_MS")
    channels: int = Field(1, alias="AUDIO_CHANNELS")
    
    @property
    def chunk_size(self) -> int:
        """Calculate chunk size in samples"""
        return int(self.sample_rate * self.chunk_ms / 1000)
    
    @property
    def bytes_per_chunk(self) -> int:
        """Calculate bytes per chunk (s16le = 2 bytes per sample)"""
        return self.chunk_size * 2 * self.channels


class VADSettings(BaseSettings):
    """Voice Activity Detection settings"""
    threshold_db: float = Field(-35.0, alias="VAD_THRESHOLD_DB")
    hangover_ms: int = Field(300, alias="VAD_HANGOVER_MS")
    
    @property
    def hangover_frames(self) -> int:
        """Calculate hangover in frames (assuming 20ms chunks)"""
        return max(1, self.hangover_ms // 20)


class RAGSettings(BaseSettings):
    """RAG and vector store settings"""
    milvus_uri: str = Field("http://localhost:19530", alias="MILVUS_URI")
    collection: str = Field("documents", alias="RAG_COLLECTION")
    top_k: int = Field(5, alias="RAG_TOP_K")
    similarity_threshold: float = Field(0.7, alias="RAG_SIMILARITY_THRESHOLD")
    groq_api_key: str = Field(..., alias="GROQ_API_KEY")
    groq_model: str = Field("qwen-2.5-32b", alias="GROQ_MODEL")
    embedding_model: str = Field("sentence-transformers/all-MiniLM-L6-v2", alias="EMBEDDING_MODEL")
    tool_port: int = Field(8001, alias="RAG_TOOL_PORT")


class Settings(BaseSettings):
    """Combined application settings"""
    ultravox: UltravoxSettings = Field(default_factory=UltravoxSettings)
    audio: AudioSettings = Field(default_factory=AudioSettings)
    vad: VADSettings = Field(default_factory=VADSettings)
    rag: RAGSettings = Field(default_factory=RAGSettings)
    log_level: str = Field("INFO", alias="LOG_LEVEL")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()


# Direct access to individual setting groups
def get_ultravox_settings() -> UltravoxSettings:
    return UltravoxSettings()


def get_audio_settings() -> AudioSettings:
    return AudioSettings()


def get_vad_settings() -> VADSettings:
    return VADSettings()


def get_rag_settings() -> RAGSettings:
    return RAGSettings()
