"""
Voice Pipeline using Official Ultravox SDK
Uses client-side tools for RAG (no HTTPS required)
"""
import asyncio
import signal
import json
from typing import Any, Dict, Optional
from loguru import logger

import ultravox_client as uv

from voice.audio_player import AudioPlayer
from voice.conversation_manager import ConversationManager
from rag.rag_service import RAGService
from core.config import get_ultravox_settings, get_audio_settings, get_rag_settings


class UltravoxVoicePipeline:
    """
    Voice pipeline using official Ultravox SDK
    
    Uses client-side tools for RAG integration (runs locally)
    """
    
    def __init__(self):
        """Initialize voice pipeline"""
        self._ultravox_settings = get_ultravox_settings()
        self._audio_settings = get_audio_settings()
        self._rag_settings = get_rag_settings()
        
        # Session and state
        self._session: Optional[uv.UltravoxSession] = None
        self._conversation = ConversationManager()
        self._done = asyncio.Event()
        
        # RAG service (lazy init)
        self._rag_service: Optional[RAGService] = None
        
        # Current transcript tracking
        self._last_transcript = None
    
    def _get_rag_service(self) -> RAGService:
        """Get or create RAG service"""
        if self._rag_service is None:
            logger.info("[PIPELINE] Initializing RAG service...")
            self._rag_service = RAGService(
                # No Groq needed - Ultravox LLM handles responses
                milvus_uri=self._rag_settings.milvus_uri,
                embedding_model=self._rag_settings.embedding_model
            )
        return self._rag_service
    
    def _search_knowledge(self, params: Dict[str, Any]) -> str:
        """
        Client-side RAG tool implementation
        
        Called by Ultravox when the agent uses searchKnowledge tool
        Uses Groq LLM to generate answers from retrieved context
        """
        query = params.get("query", "")
        logger.info(f"[RAG TOOL] Searching for: {query}")
        
        try:
            rag = self._get_rag_service()
            
            # Step 1: Retrieve context from Milvus
            context = rag.retrieve_context(
                query_text=query,
                top_k=self._rag_settings.top_k,
                collection_name=self._rag_settings.collection
            )
            
            if not context or context.startswith("Error"):
                logger.warning(f"[RAG TOOL] No results for: {query}")
                return json.dumps({
                    "found": False,
                    "message": "No relevant information found in the knowledge base."
                })
            
            source_count = context.count("[Source")
            logger.info(f"[RAG TOOL] Found {source_count} sources, generating answer with Groq...")
            
            # Step 2: Use Groq to generate answer from context
            if self._rag_settings.groq_api_key:
                try:
                    from groq import Groq
                    
                    client = Groq(api_key=self._rag_settings.groq_api_key)
                    
                    # Generate answer using Groq
                    response = client.chat.completions.create(
                        model=self._rag_settings.groq_model or "llama-3.1-70b-versatile",
                        messages=[
                            {
                                "role": "system",
                                "content": """You are a helpful assistant answering questions based on the provided context.
                                
Rules:
- Answer ONLY based on the context provided
- Keep answers concise (1-3 sentences for voice)
- If the context doesn't contain the answer, say so
- Don't make up information"""
                            },
                            {
                                "role": "user",
                                "content": f"""Context from knowledge base:
{context}

Question: {query}

Provide a concise answer based on the context above:"""
                            }
                        ],
                        temperature=0.3,
                        max_tokens=200
                    )
                    
                    answer = response.choices[0].message.content
                    logger.info(f"[RAG TOOL] Groq answer: {answer[:100]}...")
                    
                    return json.dumps({
                        "found": True,
                        "sources": source_count,
                        "answer": answer
                    })
                    
                except Exception as e:
                    logger.error(f"[RAG TOOL] Groq error: {e}, falling back to raw context")
            
            # Fallback: return raw context if Groq fails or not configured
            return json.dumps({
                "found": True,
                "sources": source_count,
                "content": context
            })
            
        except Exception as e:
            logger.error(f"[RAG TOOL] Error: {e}")
            return json.dumps({
                "found": False,
                "error": str(e)
            })
    
    async def run(self, join_url: str):
        """
        Run the voice pipeline with a join URL
        
        Args:
            join_url: Ultravox call join URL
        """
        logger.info("[PIPELINE] Starting with official Ultravox SDK...")
        
        # Create session
        self._session = uv.UltravoxSession()
        self._conversation.create_session()
        
        # Register event handlers
        @self._session.on("status")
        def on_status():
            status = self._session.status
            logger.info(f"[PIPELINE] Status: {status}")
            
            if status == uv.UltravoxSessionStatus.LISTENING:
                print("\n" + "="*50)
                print("🎤 LISTENING... (speak now)")
                print("="*50)
            elif status == uv.UltravoxSessionStatus.THINKING:
                print("\n🤔 Processing your request...")
            elif status == uv.UltravoxSessionStatus.SPEAKING:
                print("\n🔊 Speaking:")
            elif status == uv.UltravoxSessionStatus.DISCONNECTED:
                self._done.set()
        
        @self._session.on("transcripts")
        def on_transcript():
            if not self._session.transcripts:
                return
            
            transcript = self._session.transcripts[-1]
            speaker = transcript.speaker
            text = transcript.text
            is_final = transcript.final
            
            # Debug: log all transcripts
            logger.debug(f"[TRANSCRIPT] Speaker: {speaker}, Final: {is_final}, Text: {text[:50] if text else ''}...")
            
            # Display format
            if speaker == "user":
                prefix = "👤 User"
            else:
                prefix = "🤖 Assistant"
            
            # Print transcript
            display = f"{prefix}: {text}"
            if is_final:
                print(f"\r{display}                    ")  # Clear line + print
                # Track in conversation manager
                if speaker == "user":
                    self._conversation.add_user_turn(text)
                    logger.info(f"[USER SAID] {text}")
                else:
                    self._conversation.add_assistant_turn(text)
            else:
                # Partial transcript - show on same line
                print(f"\r{display}", end="", flush=True)
        
        @self._session.on("error") 
        def on_error(error):
            logger.error(f"[PIPELINE] Error: {error}")
            self._done.set()
        
        # Register client-side RAG tool
        self._session.register_tool_implementation("searchKnowledge", self._search_knowledge)
        logger.info("[PIPELINE] Registered searchKnowledge tool")
        
        try:
            # Join the call with custom audio source at 44100 Hz (matches system default)
            # The Ultravox SDK will handle resampling if needed
            from ultravox_client.audio import LocalAudioSource, LocalAudioSink
            
            # Use 44100 Hz to match your microphone's native sample rate
            audio_source = LocalAudioSource(sample_rate=44100, channels=1)
            audio_sink = LocalAudioSink(sample_rate=48000, num_channels=1)
            
            await self._session.join_call(
                join_url, 
                source=audio_source,
                sink=audio_sink
            )
            logger.info("[PIPELINE] Joined call successfully (mic @ 44100 Hz)")
            
            print("\n" + "="*50)
            print("🎤 Voice Assistant Ready! Speak now...")
            print("="*50 + "\n")
            
            # Set up signal handlers
            loop = asyncio.get_running_loop()
            try:
                loop.add_signal_handler(signal.SIGINT, lambda: self._done.set())
                loop.add_signal_handler(signal.SIGTERM, lambda: self._done.set())
            except NotImplementedError:
                # Windows doesn't support add_signal_handler
                pass
            
            # Wait until done
            await self._done.wait()
            
        except Exception as e:
            logger.error(f"[PIPELINE] Failed: {e}")
            raise
        finally:
            await self._session.leave_call()
            self._conversation.end_session()
            logger.info("[PIPELINE] Stopped")


async def create_call_and_run():
    """Create a call via API and run the pipeline"""
    import httpx
    
    settings = get_ultravox_settings()
    rag_settings = get_rag_settings()
    audio_settings = get_audio_settings()
    
    # Build system prompt
    system_prompt = """You are a helpful, friendly voice assistant with access to a knowledge base.

When the user asks questions, use the searchKnowledge tool to find relevant information.
After searching, summarize the findings naturally in your response.

Voice guidelines:
- Keep responses concise and conversational
- Speak naturally, avoid reading like text
- If you can't find info, say so honestly
"""
    
    # Configure the client-side RAG tool
    tools = [{
        "temporaryTool": {
            "modelToolName": "searchKnowledge",
            "description": "Search the knowledge base for information. Use when user asks questions about documents or specific topics.",
            "dynamicParameters": [{
                "name": "query",
                "location": "PARAMETER_LOCATION_BODY",
                "schema": {
                    "type": "string",
                    "description": "The search query"
                },
                "required": True
            }],
            "client": {}  # <-- This makes it a CLIENT tool, not HTTP!
        }
    }]
    
    # Create call
    async with httpx.AsyncClient() as client:
        logger.info("[MAIN] Creating Ultravox call...")
        
        response = await client.post(
            f"{settings.api_base}/calls",
            headers={
                "X-API-Key": settings.api_key,
                "Content-Type": "application/json"
            },
            json={
                "systemPrompt": system_prompt,
                "model": settings.model,
                "voice": settings.voice,
                "selectedTools": tools
            },
            timeout=30.0
        )
        
        if response.status_code != 200 and response.status_code != 201:
            logger.error(f"[MAIN] API Error: {response.text}")
            raise Exception(f"Failed to create call: {response.status_code}")
        
        result = response.json()
        join_url = result.get("joinUrl")
        call_id = result.get("callId")
        
        logger.info(f"[MAIN] Call created: {call_id}")
        logger.info(f"[MAIN] Join URL: {join_url[:50]}...")
    
    # Run pipeline
    pipeline = UltravoxVoicePipeline()
    await pipeline.run(join_url)


if __name__ == "__main__":
    asyncio.run(create_call_and_run())
