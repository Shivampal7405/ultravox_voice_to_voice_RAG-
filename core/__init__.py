"""
Core Module
Configuration and Ultravox client
"""
from core.config import (
    Settings,
    get_settings,
    get_ultravox_settings,
    get_audio_settings,
    get_vad_settings,
    get_rag_settings
)
from core.ultravox_client import (
    UltravoxClient,
    CallConfig,
    create_rag_tool,
    build_system_prompt
)

__all__ = [
    "Settings",
    "get_settings",
    "get_ultravox_settings",
    "get_audio_settings",
    "get_vad_settings",
    "get_rag_settings",
    "UltravoxClient",
    "CallConfig",
    "create_rag_tool",
    "build_system_prompt"
]
