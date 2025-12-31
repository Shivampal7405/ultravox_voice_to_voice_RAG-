"""
Ultravox WebSocket Handler
Handles the Ultravox protocol over WebSocket
"""
import asyncio
import json
from enum import Enum
from typing import Optional, Callable, Awaitable, Dict, Any
from dataclasses import dataclass, field
import websockets
from websockets.client import WebSocketClientProtocol
from loguru import logger


class AgentState(Enum):
    """Ultravox agent states"""
    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"


@dataclass
class Transcript:
    """Transcript message"""
    role: str  # "user" or "agent"
    text: str
    is_final: bool
    medium: str  # "voice" or "text"
    ordinal: int


@dataclass
class ToolInvocation:
    """Tool invocation from Ultravox"""
    tool_name: str
    invocation_id: str
    parameters: Dict[str, Any]


@dataclass
class WebSocketHandlerCallbacks:
    """Callbacks for WebSocket events"""
    on_state_change: Optional[Callable[[AgentState], Awaitable[None]]] = None
    on_transcript: Optional[Callable[[Transcript], Awaitable[None]]] = None
    on_audio: Optional[Callable[[bytes], Awaitable[None]]] = None
    on_tool_invocation: Optional[Callable[[ToolInvocation], Awaitable[None]]] = None
    on_playback_clear: Optional[Callable[[], Awaitable[None]]] = None
    on_call_started: Optional[Callable[[str], Awaitable[None]]] = None
    on_debug: Optional[Callable[[str], Awaitable[None]]] = None


class UltravoxWebSocketHandler:
    """
    Handles Ultravox WebSocket protocol
    
    Processes:
    - Binary messages: Audio data from agent
    - JSON messages: State, transcripts, tool invocations, etc.
    """
    
    def __init__(self, callbacks: Optional[WebSocketHandlerCallbacks] = None):
        """
        Initialize WebSocket handler
        
        Args:
            callbacks: Event callbacks
        """
        self.callbacks = callbacks or WebSocketHandlerCallbacks()
        self._socket: Optional[WebSocketClientProtocol] = None
        self._state = AgentState.IDLE
        self._call_id: Optional[str] = None
        self._running = False
        self._receive_task: Optional[asyncio.Task] = None
        self._ping_task: Optional[asyncio.Task] = None
    
    @property
    def state(self) -> AgentState:
        return self._state
    
    @property
    def is_connected(self) -> bool:
        return self._socket is not None and not self._socket.closed
    
    async def connect(self, join_url: str):
        """
        Connect to Ultravox WebSocket
        
        Args:
            join_url: WebSocket URL from call creation
        """
        logger.info(f"[WS] Connecting to Ultravox...")
        
        self._socket = await websockets.connect(
            join_url,
            ping_interval=20,
            ping_timeout=10,
            close_timeout=5
        )
        
        self._running = True
        
        # Start receive loop
        self._receive_task = asyncio.create_task(self._receive_loop())
        
        # Start ping task
        self._ping_task = asyncio.create_task(self._ping_loop())
        
        logger.info("[WS] Connected to Ultravox")
    
    async def disconnect(self):
        """Disconnect from WebSocket"""
        logger.info("[WS] Disconnecting...")
        self._running = False
        
        # Cancel tasks
        if self._ping_task:
            self._ping_task.cancel()
            try:
                await self._ping_task
            except asyncio.CancelledError:
                pass
        
        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
        
        # Close socket
        if self._socket:
            try:
                await self._socket.close()
            except Exception as e:
                logger.warning(f"[WS] Error closing socket: {e}")
            self._socket = None
        
        self._state = AgentState.IDLE
        logger.info("[WS] Disconnected")
    
    async def send_audio(self, audio_bytes: bytes):
        """
        Send audio data to Ultravox
        
        Args:
            audio_bytes: Raw s16le audio bytes
        """
        if not self.is_connected:
            return
        
        try:
            await self._socket.send(audio_bytes)
        except Exception as e:
            logger.error(f"[WS] Error sending audio: {e}")
    
    async def send_tool_result(
        self,
        invocation_id: str,
        result: str,
        error_type: Optional[str] = None,
        error_message: Optional[str] = None
    ):
        """
        Send tool execution result
        
        Args:
            invocation_id: ID from tool invocation
            result: Tool result string
            error_type: Optional error type
            error_message: Optional error message
        """
        message = {
            "type": "client_tool_result",
            "invocationId": invocation_id,
            "result": result,
            "responseType": "tool-response",
            "agentReaction": "speaks"
        }
        
        if error_type:
            message["errorType"] = error_type
            message["errorMessage"] = error_message or "Unknown error"
        
        await self._send_json(message)
        logger.debug(f"[WS] Sent tool result for {invocation_id}")
    
    async def send_text_message(self, text: str, urgency: str = "soon"):
        """
        Send text message (for text-based input)
        
        Args:
            text: Message text
            urgency: "immediate", "soon", or "later"
        """
        message = {
            "type": "user_text_message",
            "text": text,
            "urgency": urgency
        }
        await self._send_json(message)
    
    async def send_hang_up(self, message: str = "Goodbye"):
        """End the call"""
        await self._send_json({
            "type": "hang_up",
            "message": message
        })
    
    async def _send_json(self, data: Dict[str, Any]):
        """Send JSON message"""
        if not self.is_connected:
            return
        
        try:
            await self._socket.send(json.dumps(data))
        except Exception as e:
            logger.error(f"[WS] Error sending JSON: {e}")
    
    async def _receive_loop(self):
        """Main receive loop"""
        try:
            async for message in self._socket:
                if not self._running:
                    break
                
                if isinstance(message, bytes):
                    # Binary = audio data from agent
                    await self._handle_audio(message)
                else:
                    # Text = JSON message
                    await self._handle_json(message)
                    
        except websockets.exceptions.ConnectionClosed as e:
            logger.info(f"[WS] Connection closed: {e}")
        except Exception as e:
            logger.error(f"[WS] Receive error: {e}")
        finally:
            self._running = False
    
    async def _ping_loop(self):
        """Send periodic pings"""
        import time
        try:
            while self._running:
                await asyncio.sleep(15)
                if self.is_connected:
                    await self._send_json({
                        "type": "ping",
                        "timestamp": time.time()
                    })
        except asyncio.CancelledError:
            pass
    
    async def _handle_audio(self, audio_bytes: bytes):
        """Handle incoming audio from agent"""
        if self.callbacks.on_audio:
            await self.callbacks.on_audio(audio_bytes)
    
    async def _handle_json(self, message: str):
        """Handle JSON message"""
        try:
            data = json.loads(message)
            msg_type = data.get("type")
            
            if msg_type == "state":
                await self._handle_state(data)
            elif msg_type == "transcript":
                await self._handle_transcript(data)
            elif msg_type == "client_tool_invocation":
                await self._handle_tool_invocation(data)
            elif msg_type == "playback_clear_buffer":
                await self._handle_playback_clear()
            elif msg_type == "call_started":
                await self._handle_call_started(data)
            elif msg_type == "pong":
                pass  # Ignore pong responses
            elif msg_type == "debug":
                await self._handle_debug(data)
            else:
                logger.debug(f"[WS] Unknown message type: {msg_type}")
                
        except json.JSONDecodeError as e:
            logger.error(f"[WS] Invalid JSON: {e}")
    
    async def _handle_state(self, data: Dict[str, Any]):
        """Handle state change"""
        state_str = data.get("state", "idle")
        try:
            new_state = AgentState(state_str)
        except ValueError:
            new_state = AgentState.IDLE
        
        if new_state != self._state:
            old_state = self._state
            self._state = new_state
            logger.info(f"[WS] State: {old_state.value} → {new_state.value}")
            
            if self.callbacks.on_state_change:
                await self.callbacks.on_state_change(new_state)
    
    async def _handle_transcript(self, data: Dict[str, Any]):
        """Handle transcript message"""
        transcript = Transcript(
            role=data.get("role", "unknown"),
            text=data.get("text", "") or data.get("delta", ""),
            is_final=data.get("final", False),
            medium=data.get("medium", "voice"),
            ordinal=data.get("ordinal", 0)
        )
        
        if transcript.text:
            logger.debug(f"[WS] Transcript ({transcript.role}, final={transcript.is_final}): {transcript.text[:50]}...")
        
        if self.callbacks.on_transcript:
            await self.callbacks.on_transcript(transcript)
    
    async def _handle_tool_invocation(self, data: Dict[str, Any]):
        """Handle tool invocation request"""
        invocation = ToolInvocation(
            tool_name=data.get("toolName", ""),
            invocation_id=data.get("invocationId", ""),
            parameters=data.get("parameters", {})
        )
        
        logger.info(f"[WS] Tool invocation: {invocation.tool_name}({invocation.parameters})")
        
        if self.callbacks.on_tool_invocation:
            await self.callbacks.on_tool_invocation(invocation)
    
    async def _handle_playback_clear(self):
        """Handle playback clear buffer (barge-in)"""
        logger.debug("[WS] Playback clear buffer (barge-in)")
        
        if self.callbacks.on_playback_clear:
            await self.callbacks.on_playback_clear()
    
    async def _handle_call_started(self, data: Dict[str, Any]):
        """Handle call started message"""
        self._call_id = data.get("callId")
        logger.info(f"[WS] Call started: {self._call_id}")
        
        if self.callbacks.on_call_started:
            await self.callbacks.on_call_started(self._call_id)
    
    async def _handle_debug(self, data: Dict[str, Any]):
        """Handle debug message"""
        message = data.get("message", "")
        logger.debug(f"[WS] Debug: {message}")
        
        if self.callbacks.on_debug:
            await self.callbacks.on_debug(message)
