"""
Audio Capture Module
Microphone input using sounddevice
"""
import asyncio
import queue
import numpy as np
import sounddevice as sd
from typing import AsyncGenerator, Optional
from loguru import logger


class AudioCapture:
    """
    Asynchronous microphone audio capture
    
    Captures audio from default microphone and yields chunks
    as an async generator.
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        chunk_ms: int = 20,
        device: Optional[int] = None
    ):
        """
        Initialize audio capture
        
        Args:
            sample_rate: Audio sample rate in Hz
            channels: Number of audio channels (1 = mono)
            chunk_ms: Chunk duration in milliseconds
            device: Audio device index (None = default)
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_ms = chunk_ms
        self.device = device
        
        # Calculate chunk size
        self.chunk_size = int(sample_rate * chunk_ms / 1000)
        
        # Audio queue for async streaming
        self._queue: queue.Queue = queue.Queue()
        self._stream: Optional[sd.InputStream] = None
        self._running = False
    
    def _audio_callback(self, indata: np.ndarray, frames: int, time_info, status):
        """Callback for sounddevice stream"""
        if status:
            logger.warning(f"[AUDIO CAPTURE] Status: {status}")
        
        # Convert to s16le bytes and queue
        audio_bytes = (indata * 32767).astype(np.int16).tobytes()
        self._queue.put(audio_bytes)
    
    def start(self):
        """Start audio capture"""
        if self._running:
            return
        
        logger.info(f"[AUDIO CAPTURE] Starting: {self.sample_rate}Hz, {self.channels}ch, {self.chunk_ms}ms chunks")
        
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype=np.float32,
            blocksize=self.chunk_size,
            device=self.device,
            callback=self._audio_callback
        )
        self._stream.start()
        self._running = True
        
        logger.info("[AUDIO CAPTURE] Started")
    
    def stop(self):
        """Stop audio capture"""
        if not self._running:
            return
        
        logger.info("[AUDIO CAPTURE] Stopping...")
        self._running = False
        
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        
        # Clear queue
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        
        logger.info("[AUDIO CAPTURE] Stopped")
    
    async def stream(self) -> AsyncGenerator[bytes, None]:
        """
        Async generator yielding audio chunks
        
        Yields:
            Raw s16le audio bytes
        """
        if not self._running:
            self.start()
        
        try:
            while self._running:
                try:
                    # Non-blocking queue get with timeout
                    chunk = self._queue.get(timeout=0.1)
                    yield chunk
                except queue.Empty:
                    # Allow other coroutines to run
                    await asyncio.sleep(0.001)
        finally:
            pass  # Cleanup handled by stop()
    
    def get_chunk_nowait(self) -> Optional[bytes]:
        """Get audio chunk without blocking (returns None if empty)"""
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            return None
    
    @property
    def is_running(self) -> bool:
        return self._running
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


def list_audio_devices():
    """List available audio devices"""
    logger.info("[AUDIO] Available devices:")
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        logger.info(f"  [{i}] {device['name']} (in: {device['max_input_channels']}, out: {device['max_output_channels']})")
    return devices
