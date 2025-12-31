"""
Voice Pipeline Orchestrator
Main controller for the voice-to-voice system
"""
import asyncio
from enum import Enum
from typing import Optional, Callable, Awaitable
from loguru import logger

from voice.audio_capture import AudioCapture
from voice.audio_player import AudioPlayer
from voice.vad import VoiceActivityDetector
from voice.websocket_handler import (
    UltravoxWebSocketHandler,
    WebSocketHandlerCallbacks,
    AgentState,
    Transcript,
    ToolInvocation
)
from voice.conversation_manager import ConversationManager
from core.ultravox_client import UltravoxClient, CallConfig, create_rag_tool, build_system_prompt
from core.config import get_ultravox_settings, get_audio_settings, get_vad_settings, get_rag_settings


class PipelineState(Enum):
    """Voice pipeline states"""
    INITIALIZING = "initializing"
    READY = "ready"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
    ERROR = "error"
    STOPPED = "stopped"


class VoicePipeline:
    """
    Main voice pipeline orchestrator
    
    Coordinates:
    - Audio capture and VAD
    - Ultravox WebSocket connection
    - Audio playback
    - Conversation state
    - RAG tool handling
    """
    
    def __init__(
        self,
        rag_tool_endpoint: Optional[str] = None,
        on_state_change: Optional[Callable[[PipelineState], Awaitable[None]]] = None,
        on_transcript: Optional[Callable[[str, str, bool], Awaitable[None]]] = None
    ):
        """
        Initialize voice pipeline
        
        Args:
            rag_tool_endpoint: URL of RAG tool endpoint
            on_state_change: Callback for pipeline state changes
            on_transcript: Callback for transcripts (role, text, is_final)
        """
        # Load settings
        self._ultravox_settings = get_ultravox_settings()
        self._audio_settings = get_audio_settings()
        self._vad_settings = get_vad_settings()
        self._rag_settings = get_rag_settings()
        
        # RAG tool endpoint
        self._rag_endpoint = rag_tool_endpoint or f"http://localhost:{self._rag_settings.tool_port}/searchKnowledge"
        
        # Callbacks
        self._on_state_change = on_state_change
        self._on_transcript = on_transcript
        
        # Components (initialized in start())
        self._ultravox_client: Optional[UltravoxClient] = None
        self._ws_handler: Optional[UltravoxWebSocketHandler] = None
        self._audio_capture: Optional[AudioCapture] = None
        self._audio_player: Optional[AudioPlayer] = None
        self._vad: Optional[VoiceActivityDetector] = None
        self._conversation: Optional[ConversationManager] = None
        
        # State
        self._state = PipelineState.INITIALIZING
        self._running = False
        self._audio_send_task: Optional[asyncio.Task] = None
        self._call_id: Optional[str] = None
        
        # Transcript building
        self._current_user_transcript = ""
        self._current_agent_transcript = ""
    
    @property
    def state(self) -> PipelineState:
        return self._state
    
    async def _set_state(self, state: PipelineState):
        """Set pipeline state and notify"""
        if state != self._state:
            old_state = self._state
            self._state = state
            logger.info(f"[PIPELINE] State: {old_state.value} → {state.value}")
            
            if self._on_state_change:
                await self._on_state_change(state)
    
    async def start(self):
        """Start the voice pipeline"""
        logger.info("[PIPELINE] Starting...")
        
        try:
            # Initialize components
            self._ultravox_client = UltravoxClient(
                api_key=self._ultravox_settings.api_key,
                api_base=self._ultravox_settings.api_base
            )
            
            self._audio_capture = AudioCapture(
                sample_rate=self._audio_settings.sample_rate,
                channels=self._audio_settings.channels,
                chunk_ms=self._audio_settings.chunk_ms
            )
            
            self._audio_player = AudioPlayer(
                sample_rate=self._audio_settings.sample_rate,
                channels=self._audio_settings.channels
            )
            
            self._vad = VoiceActivityDetector(
                threshold_db=self._vad_settings.threshold_db,
                hangover_frames=self._vad_settings.hangover_frames,
                sample_rate=self._audio_settings.sample_rate
            )
            
            self._conversation = ConversationManager()
            self._conversation.create_session()
            
            # Create WebSocket handler with callbacks
            callbacks = WebSocketHandlerCallbacks(
                on_state_change=self._handle_state_change,
                on_transcript=self._handle_transcript,
                on_audio=self._handle_audio,
                on_tool_invocation=self._handle_tool_invocation,
                on_playback_clear=self._handle_playback_clear,
                on_call_started=self._handle_call_started
            )
            self._ws_handler = UltravoxWebSocketHandler(callbacks)
            
            # Create call with RAG tool
            tools = [create_rag_tool(self._rag_endpoint)]
            
            call_config = CallConfig(
                system_prompt=build_system_prompt(),
                model=self._ultravox_settings.model,
                voice=self._ultravox_settings.voice,
                input_sample_rate=self._audio_settings.sample_rate,
                output_sample_rate=self._audio_settings.sample_rate,
                tools=tools
            )
            
            result = await self._ultravox_client.create_call(call_config)
            join_url = result.get("joinUrl")
            
            if not join_url:
                raise ValueError("No joinUrl in response")
            
            # Connect WebSocket
            await self._ws_handler.connect(join_url)
            
            # Start audio capture and player
            self._audio_capture.start()
            self._audio_player.start()
            
            # Start audio streaming task
            self._running = True
            self._audio_send_task = asyncio.create_task(self._audio_send_loop())
            
            await self._set_state(PipelineState.READY)
            logger.info("[PIPELINE] Started successfully")
            
        except Exception as e:
            logger.error(f"[PIPELINE] Start failed: {e}")
            await self._set_state(PipelineState.ERROR)
            raise
    
    async def stop(self):
        """Stop the voice pipeline"""
        logger.info("[PIPELINE] Stopping...")
        self._running = False
        
        # Cancel audio send task
        if self._audio_send_task:
            self._audio_send_task.cancel()
            try:
                await self._audio_send_task
            except asyncio.CancelledError:
                pass
        
        # Stop components
        if self._ws_handler:
            await self._ws_handler.disconnect()
        
        if self._audio_capture:
            self._audio_capture.stop()
        
        if self._audio_player:
            self._audio_player.stop()
        
        if self._ultravox_client:
            await self._ultravox_client.close()
        
        if self._conversation:
            self._conversation.end_session()
        
        await self._set_state(PipelineState.STOPPED)
        logger.info("[PIPELINE] Stopped")
    
    async def _audio_send_loop(self):
        """Main loop for capturing and sending audio"""
        logger.info("[PIPELINE] Audio send loop started")
        
        try:
            async for chunk in self._audio_capture.stream():
                if not self._running:
                    break
                
                # Check VAD
                is_speech = self._vad.is_speech(chunk)
                
                # Always send audio (Ultravox handles its own VAD too)
                await self._ws_handler.send_audio(chunk)
                
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"[PIPELINE] Audio send error: {e}")
        finally:
            logger.info("[PIPELINE] Audio send loop ended")
    
    async def _handle_state_change(self, state: AgentState):
        """Handle Ultravox state change"""
        state_map = {
            AgentState.IDLE: PipelineState.READY,
            AgentState.LISTENING: PipelineState.LISTENING,
            AgentState.THINKING: PipelineState.THINKING,
            AgentState.SPEAKING: PipelineState.SPEAKING
        }
        
        pipeline_state = state_map.get(state, PipelineState.READY)
        await self._set_state(pipeline_state)
        
        # Commit transcripts on state transitions
        if state == AgentState.THINKING:
            # User finished speaking
            if self._current_user_transcript:
                self._conversation.add_user_turn(self._current_user_transcript)
                self._current_user_transcript = ""
        
        elif state == AgentState.LISTENING:
            # Agent finished speaking
            if self._current_agent_transcript:
                self._conversation.add_assistant_turn(self._current_agent_transcript)
                self._current_agent_transcript = ""
    
    async def _handle_transcript(self, transcript: Transcript):
        """Handle transcript message"""
        if transcript.role == "user":
            self._current_user_transcript = transcript.text
        else:
            self._current_agent_transcript = transcript.text
        
        if self._on_transcript:
            await self._on_transcript(
                transcript.role,
                transcript.text,
                transcript.is_final
            )
    
    async def _handle_audio(self, audio_bytes: bytes):
        """Handle incoming agent audio"""
        self._audio_player.play(audio_bytes)
    
    async def _handle_tool_invocation(self, invocation: ToolInvocation):
        """Handle tool invocation request"""
        if invocation.tool_name == "searchKnowledge":
            # This is handled by the HTTP tool endpoint
            # Ultravox will call our RAG server directly
            logger.info(f"[PIPELINE] searchKnowledge tool called with query: {invocation.parameters.get('query')}")
        else:
            # Unknown tool - send error
            await self._ws_handler.send_tool_result(
                invocation_id=invocation.invocation_id,
                result="",
                error_type="undefined",
                error_message=f"Unknown tool: {invocation.tool_name}"
            )
    
    async def _handle_playback_clear(self):
        """Handle barge-in (user interruption)"""
        logger.info("[PIPELINE] Barge-in detected, clearing playback")
        self._audio_player.clear_buffer()
        self._current_agent_transcript = ""  # Discard incomplete response
    
    async def _handle_call_started(self, call_id: str):
        """Handle call started"""
        self._call_id = call_id
        logger.info(f"[PIPELINE] Call active: {call_id}")
    
    async def run_until_complete(self):
        """Run pipeline until stopped or error"""
        try:
            await self.start()
            
            # Keep running until stopped
            while self._running and self._ws_handler.is_connected:
                await asyncio.sleep(0.5)
                
        except KeyboardInterrupt:
            logger.info("[PIPELINE] Interrupted by user")
        except Exception as e:
            logger.error(f"[PIPELINE] Error: {e}")
        finally:
            await self.stop()


async def run_voice_pipeline(rag_endpoint: Optional[str] = None):
    """
    Convenience function to run the voice pipeline
    
    Args:
        rag_endpoint: Optional RAG tool endpoint URL
    """
    pipeline = VoicePipeline(rag_tool_endpoint=rag_endpoint)
    
    async def on_transcript(role: str, text: str, is_final: bool):
        prefix = "👤" if role == "user" else "🤖"
        status = "✓" if is_final else "..."
        print(f"{prefix} [{status}] {text}")
    
    pipeline._on_transcript = on_transcript
    
    await pipeline.run_until_complete()
