from anthropic import Anthropic
from typing import Optional, Dict, List
import logging
import asyncio
from config.settings import settings

class ClaudeService:
    """Service for interacting with Claude API"""
    
    def __init__(self):
        self.client = Anthropic(api_key=settings.ANTHROPIC_API_KEY)
        self.logger = logging.getLogger(__name__)
        
    async def enhance_transcription(
        self, 
        text: str,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Enhance transcription using Claude
        
        Args:
            text: Raw transcription text
            system_prompt: Optional custom system prompt
            
        Returns:
            Enhanced transcription text
        """
        try:
            if not system_prompt:
                system_prompt = """You are an expert in audio transcription enhancement. 
                Your task is to:
                1. Correct obvious transcription errors
                2. Add proper punctuation and capitalization
                3. Format the text for readability (paragraphs, etc.)
                4. Fix grammar while preserving the original meaning
                5. Add speaker labels if multiple speakers are detected
                6. Note any uncertainties in [brackets]
                
                Maintain the original tone and style while improving clarity."""
            
            # Run in executor to avoid blocking
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                self._call_claude,
                system_prompt,
                f"Please enhance this transcription:\n\n{text}"
            )
            
            return response
            
        except Exception as e:
            self.logger.error(f"Claude API error: {e}")
            raise
    
    async def get_contextual_response(
        self,
        prompt: str,
        context: str,
        chat_history: Optional[List[Dict]] = None
    ) -> str:
        """
        Get AI response with context
        
        Args:
            prompt: User's question
            context: Context about the audio/transcription
            chat_history: Previous chat messages
            
        Returns:
            AI response
        """
        try:
            system_prompt = """You are an AI assistant helping with audio analysis and transcription.
            You have access to information about the audio file and its transcription.
            Be helpful, accurate, and concise in your responses."""
            
            # Build message with context
            full_prompt = f"""Context about the audio:
{context}

User question: {prompt}"""
            
            # Include chat history if available
            messages = []
            if chat_history:
                for msg in chat_history[-10:]:  # Last 10 messages
                    messages.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })
            
            messages.append({
                "role": "user",
                "content": full_prompt
            })
            
            # Run in executor
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                self._call_claude_with_history,
                system_prompt,
                messages
            )
            
            return response
            
        except Exception as e:
            self.logger.error(f"Claude contextual response error: {e}")
            raise
    
    def _call_claude(self, system_prompt: str, user_prompt: str) -> str:
        """Synchronous Claude API call"""
        response = self.client.messages.create(
            model=settings.CLAUDE_MODEL,
            max_tokens=4000,
            temperature=0.3,
            system=system_prompt,
            messages=[{
                "role": "user",
                "content": user_prompt
            }]
        )
        return response.content[0].text
    
    def _call_claude_with_history(
        self, 
        system_prompt: str, 
        messages: List[Dict]
    ) -> str:
        """Synchronous Claude API call with message history"""
        response = self.client.messages.create(
            model=settings.CLAUDE_MODEL,
            max_tokens=2000,
            temperature=0.7,
            system=system_prompt,
            messages=messages
        )
        return response.content[0].text 