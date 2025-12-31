"""
Groq LLM Client
Wrapper for Groq API with streaming support
"""
from typing import List, Dict, Any, Optional, AsyncGenerator
from groq import Groq, AsyncGroq
from loguru import logger


class GroqLLM:
    """Groq LLM client with chat history"""
    
    def __init__(
        self,
        api_key: str,
        model_name: str = "llama-3.1-70b-versatile",
        temperature: float = 0.7,
        max_tokens: int = 1024
    ):
        """
        Initialize Groq LLM
        
        Args:
            api_key: Groq API key
            model_name: Model to use
            temperature: Sampling temperature
            max_tokens: Maximum output tokens
        """
        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        self.client = Groq(api_key=api_key)
        self.async_client = AsyncGroq(api_key=api_key)
        
        self.system_instruction = ""
        self.history: List[Dict[str, str]] = []
    
    def set_system_instruction(self, instruction: str):
        """Set system instruction"""
        self.system_instruction = instruction
    
    def add_to_history(self, role: str, content: str):
        """Add message to history"""
        self.history.append({"role": role, "content": content})
    
    def clear_history(self):
        """Clear conversation history"""
        self.history = []
    
    def generate_response(self, user_message: str) -> str:
        """
        Generate a response (synchronous)
        
        Args:
            user_message: User's message
            
        Returns:
            Assistant's response
        """
        messages = []
        
        # Add system instruction
        if self.system_instruction:
            messages.append({"role": "system", "content": self.system_instruction})
        
        # Add history
        messages.extend(self.history)
        
        # Add current message
        messages.append({"role": "user", "content": user_message})
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            content = response.choices[0].message.content
            
            # Add to history
            self.add_to_history("user", user_message)
            self.add_to_history("assistant", content)
            
            return content
            
        except Exception as e:
            logger.error(f"[GROQ] Error generating response: {e}")
            raise
    
    async def generate_response_stream(self, user_message: str) -> AsyncGenerator[str, None]:
        """
        Generate a streaming response (async)
        
        Args:
            user_message: User's message
            
        Yields:
            Text chunks as they arrive
        """
        messages = []
        
        # Add system instruction
        if self.system_instruction:
            messages.append({"role": "system", "content": self.system_instruction})
        
        # Add history
        messages.extend(self.history)
        
        # Add current message
        messages.append({"role": "user", "content": user_message})
        
        try:
            stream = await self.async_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True
            )
            
            full_response = ""
            
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    text = chunk.choices[0].delta.content
                    full_response += text
                    yield text
            
            # Add to history after complete
            self.add_to_history("user", user_message)
            self.add_to_history("assistant", full_response)
            
        except Exception as e:
            logger.error(f"[GROQ] Error in streaming response: {e}")
            raise
