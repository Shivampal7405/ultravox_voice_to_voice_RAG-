"""
Ultravox REST API Client
Handles call creation and management
"""
import httpx
from typing import Dict, Any, Optional, List
from loguru import logger
from dataclasses import dataclass


@dataclass
class CallConfig:
    """Configuration for creating an Ultravox call"""
    system_prompt: str
    model: str = "ultravox-v0.7"
    voice: str = "Mark"
    input_sample_rate: int = 16000
    output_sample_rate: int = 16000
    client_buffer_size_ms: int = 60
    tools: Optional[List[Dict[str, Any]]] = None


class UltravoxClient:
    """Ultravox REST API client for call management"""
    
    def __init__(self, api_key: str, api_base: str = "https://api.ultravox.ai/api"):
        self.api_key = api_key
        self.api_base = api_base.rstrip("/")
        self.client = httpx.AsyncClient(
            headers={
                "X-API-Key": api_key,
                "Content-Type": "application/json"
            },
            timeout=30.0
        )
    
    async def create_call(self, config: CallConfig) -> Dict[str, Any]:
        """
        Create a new WebSocket-based call
        
        Returns:
            Dict with 'joinUrl' for WebSocket connection and 'callId'
        """
        payload = {
            "systemPrompt": config.system_prompt,
            "model": config.model,
            "voice": config.voice,
            "medium": {
                "serverWebSocket": {
                    "inputSampleRate": config.input_sample_rate,
                    "outputSampleRate": config.output_sample_rate,
                }
            }
        }
        
        # Add tools if specified
        if config.tools:
            payload["selectedTools"] = config.tools
        
        logger.info(f"[ULTRAVOX] Creating call with model={config.model}, voice={config.voice}")
        logger.debug(f"[ULTRAVOX] Payload: {payload}")
        
        response = await self.client.post(
            f"{self.api_base}/calls",
            json=payload
        )
        
        # Better error handling - show the actual error message
        if response.status_code != 200 and response.status_code != 201:
            error_body = response.text
            logger.error(f"[ULTRAVOX] API Error {response.status_code}: {error_body}")
            response.raise_for_status()
        
        result = response.json()
        
        logger.info(f"[ULTRAVOX] Call created: {result.get('callId', 'unknown')}")
        return result
    
    async def get_call(self, call_id: str) -> Dict[str, Any]:
        """Get call details"""
        response = await self.client.get(f"{self.api_base}/calls/{call_id}")
        response.raise_for_status()
        return response.json()
    
    async def send_message(self, call_id: str, message: Dict[str, Any]) -> None:
        """Send a message to an active call via REST API"""
        response = await self.client.post(
            f"{self.api_base}/calls/{call_id}/messages",
            json=message
        )
        response.raise_for_status()
    
    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()


def create_rag_tool(tool_endpoint: str) -> Dict[str, Any]:
    """
    Create a RAG tool configuration for Ultravox
    
    Args:
        tool_endpoint: Full URL of the RAG search endpoint
        
    Returns:
        Tool configuration dict for Ultravox selectedTools
    """
    return {
        "temporaryTool": {
            "modelToolName": "searchKnowledge",
            "description": (
                "Search the knowledge base for relevant information. "
                "Use this tool when the user asks questions that might be answered "
                "by documents, product info, or stored knowledge. The tool returns "
                "relevant text chunks from the knowledge base."
            ),
            "dynamicParameters": [
                {
                    "name": "query",
                    "location": "PARAMETER_LOCATION_BODY",
                    "schema": {
                        "type": "string",
                        "description": "The search query to find relevant information"
                    },
                    "required": True
                }
            ],
            "http": {
                "baseUrlPattern": tool_endpoint,
                "httpMethod": "POST"
            }
        }
    }


def build_system_prompt(context: str = "") -> str:
    """
    Build the system prompt for the voice assistant
    
    Args:
        context: Optional additional context to include
    """
    base_prompt = """You are a helpful, friendly voice assistant with access to a knowledge base.

Key behaviors:
- Keep responses concise and conversational (voice-first)
- Use natural speech patterns, not formal writing
- When you don't know something, use the searchKnowledge tool to look it up
- After using searchKnowledge, summarize the findings naturally
- If the knowledge base doesn't have the answer, say so honestly

Voice output guidelines:
- Speak in short, clear sentences
- Avoid jargon unless the user uses it first
- Don't use markdown, bullet points, or formatting
- Numbers should be spoken naturally (say "twenty-three" not "23")
"""
    
    if context:
        base_prompt += f"\n\nAdditional context:\n{context}"
    
    return base_prompt
