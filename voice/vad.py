"""
Voice Activity Detection (VAD)
RMS-based energy detection with hangover
"""
import numpy as np
from typing import Optional
from loguru import logger


class VoiceActivityDetector:
    """
    Simple RMS-based Voice Activity Detector
    
    Uses energy threshold with hangover frames to avoid
    cutting off speech mid-word.
    """
    
    def __init__(
        self,
        threshold_db: float = -35.0,
        hangover_frames: int = 15,  # ~300ms at 20ms chunks
        sample_rate: int = 16000
    ):
        """
        Initialize VAD
        
        Args:
            threshold_db: RMS threshold in dB (typical speech is -20 to -10 dB)
            hangover_frames: Number of frames to keep active after speech ends
            sample_rate: Audio sample rate
        """
        self.threshold_db = threshold_db
        self.threshold_linear = 10 ** (threshold_db / 20)
        self.hangover_frames = hangover_frames
        self.sample_rate = sample_rate
        
        # State
        self._hangover_counter = 0
        self._is_speaking = False
        self._speech_started = False
    
    def reset(self):
        """Reset VAD state"""
        self._hangover_counter = 0
        self._is_speaking = False
        self._speech_started = False
    
    def _compute_rms(self, audio_chunk: bytes) -> float:
        """
        Compute RMS of audio chunk
        
        Args:
            audio_chunk: Raw s16le audio bytes
            
        Returns:
            RMS value (0.0 to 1.0)
        """
        # Convert s16le bytes to numpy array
        samples = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32)
        
        if len(samples) == 0:
            return 0.0
        
        # Normalize to -1.0 to 1.0 range
        samples = samples / 32768.0
        
        # Compute RMS
        rms = np.sqrt(np.mean(samples ** 2))
        
        return rms
    
    def _rms_to_db(self, rms: float) -> float:
        """Convert RMS to dB"""
        if rms <= 0:
            return -100.0
        return 20 * np.log10(rms)
    
    def is_speech(self, audio_chunk: bytes) -> bool:
        """
        Determine if audio chunk contains speech
        
        Args:
            audio_chunk: Raw s16le audio bytes
            
        Returns:
            True if speech detected (including hangover period)
        """
        rms = self._compute_rms(audio_chunk)
        rms_db = self._rms_to_db(rms)
        
        # Check if above threshold
        is_above_threshold = rms >= self.threshold_linear
        
        if is_above_threshold:
            # Speech detected
            if not self._speech_started:
                self._speech_started = True
                logger.debug(f"[VAD] Speech started (RMS: {rms_db:.1f} dB)")
            
            self._is_speaking = True
            self._hangover_counter = self.hangover_frames
        else:
            # Below threshold
            if self._hangover_counter > 0:
                # Still in hangover period
                self._hangover_counter -= 1
            else:
                # Hangover expired
                if self._speech_started and self._is_speaking:
                    logger.debug(f"[VAD] Speech ended (RMS: {rms_db:.1f} dB)")
                    self._speech_started = False
                self._is_speaking = False
        
        return self._is_speaking
    
    def get_rms_db(self, audio_chunk: bytes) -> float:
        """Get RMS in dB for debugging"""
        rms = self._compute_rms(audio_chunk)
        return self._rms_to_db(rms)
    
    @property
    def is_speaking(self) -> bool:
        """Current speaking state"""
        return self._is_speaking


class SilenceInjector:
    """
    Generates silence frames when no speech is detected
    
    Keeps the WebSocket connection alive without sending
    unnecessary audio data.
    """
    
    def __init__(self, sample_rate: int = 16000, chunk_ms: int = 20):
        self.sample_rate = sample_rate
        self.chunk_ms = chunk_ms
        self.chunk_size = int(sample_rate * chunk_ms / 1000)
        self._silence_chunk = bytes(self.chunk_size * 2)  # s16le = 2 bytes
    
    def get_silence(self) -> bytes:
        """Get a chunk of silence"""
        return self._silence_chunk
