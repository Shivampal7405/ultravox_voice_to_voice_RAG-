"""
Audio Player Module
TTS audio playback with interruption support
"""
import asyncio
import queue
import threading
import numpy as np
import sounddevice as sd
from typing import Optional
from loguru import logger


class AudioPlayer:
    """
    Audio player for TTS output
    
    Features:
    - Lock-free audio queue
    - Barge-in support (clear buffer)
    - Async and sync interfaces
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        device: Optional[int] = None,
        buffer_size_ms: int = 100
    ):
        """
        Initialize audio player
        
        Args:
            sample_rate: Audio sample rate
            channels: Number of channels
            device: Audio device index (None = default)
            buffer_size_ms: Buffer size in milliseconds
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.device = device
        
        # Audio buffer
        self._queue: queue.Queue = queue.Queue()
        self._stream: Optional[sd.OutputStream] = None
        self._running = False
        self._playing = False
        
        # Pre-calculate buffer size
        self._buffer_samples = int(sample_rate * buffer_size_ms / 1000)
    
    def _audio_callback(self, outdata: np.ndarray, frames: int, time_info, status):
        """Callback for sounddevice output stream"""
        if status:
            logger.warning(f"[AUDIO PLAYER] Status: {status}")
        
        # Try to get audio from queue
        bytes_needed = frames * self.channels * 2  # s16le
        audio_data = b""
        
        while len(audio_data) < bytes_needed:
            try:
                chunk = self._queue.get_nowait()
                audio_data += chunk
            except queue.Empty:
                break
        
        if len(audio_data) >= bytes_needed:
            # Convert s16le bytes to float32
            samples = np.frombuffer(audio_data[:bytes_needed], dtype=np.int16)
            outdata[:] = samples.reshape(-1, self.channels).astype(np.float32) / 32768.0
            
            # Put back any extra
            if len(audio_data) > bytes_needed:
                self._queue.put(audio_data[bytes_needed:])
        else:
            # Not enough data, pad with silence
            if audio_data:
                samples = np.frombuffer(audio_data, dtype=np.int16)
                samples_float = samples.astype(np.float32) / 32768.0
                outdata[:len(samples_float)] = samples_float.reshape(-1, self.channels)
                outdata[len(samples_float):] = 0
            else:
                outdata.fill(0)
    
    def start(self):
        """Start the audio output stream"""
        if self._running:
            return
        
        logger.info(f"[AUDIO PLAYER] Starting: {self.sample_rate}Hz, {self.channels}ch")
        
        self._stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype=np.float32,
            blocksize=1024,
            device=self.device,
            callback=self._audio_callback
        )
        self._stream.start()
        self._running = True
        
        logger.info("[AUDIO PLAYER] Started")
    
    def stop(self):
        """Stop the audio output stream"""
        if not self._running:
            return
        
        logger.info("[AUDIO PLAYER] Stopping...")
        self._running = False
        
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        
        self.clear_buffer()
        logger.info("[AUDIO PLAYER] Stopped")
    
    def play(self, audio_bytes: bytes):
        """
        Queue audio for playback
        
        Args:
            audio_bytes: Raw s16le audio bytes
        """
        if not self._running:
            self.start()
        
        self._queue.put(audio_bytes)
        self._playing = True
    
    async def play_async(self, audio_bytes: bytes):
        """Async version of play"""
        self.play(audio_bytes)
    
    def clear_buffer(self):
        """
        Clear audio buffer (for barge-in/interruption)
        
        Called when user interrupts the assistant
        """
        cleared = 0
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
                cleared += 1
            except queue.Empty:
                break
        
        self._playing = False
        
        if cleared > 0:
            logger.debug(f"[AUDIO PLAYER] Cleared {cleared} chunks from buffer")
    
    def is_playing(self) -> bool:
        """Check if audio is currently playing"""
        return self._playing and not self._queue.empty()
    
    @property
    def buffer_size(self) -> int:
        """Current buffer size in chunks"""
        return self._queue.qsize()
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


class AudioPlayerAsync(AudioPlayer):
    """
    Async-first audio player with non-blocking operations
    """
    
    async def wait_until_done(self, timeout: float = 30.0):
        """Wait until all queued audio has finished playing"""
        start = asyncio.get_event_loop().time()
        
        while not self._queue.empty():
            if asyncio.get_event_loop().time() - start > timeout:
                logger.warning("[AUDIO PLAYER] Timeout waiting for playback")
                break
            await asyncio.sleep(0.05)
    
    async def play_stream(self, audio_generator):
        """
        Play audio from an async generator
        
        Args:
            audio_generator: Async generator yielding audio bytes
        """
        if not self._running:
            self.start()
        
        async for chunk in audio_generator:
            if not self._running:
                break
            self.play(chunk)
