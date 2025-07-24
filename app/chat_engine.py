"""
Chat engine module for RAG Agent application.
"""

from typing import List, Any
from pathlib import Path

from anthropic import AsyncAnthropic

from app.config import get_settings
from app.services.rag_engine import get_relevant_chunks
from app.services.memory import load_chat_history, save_message


class ChatEngine:
    """Main chat engine that orchestrates the RAG agent conversation flow."""
    
    def __init__(self):
        """Initialize the chat engine with Anthropic client."""
        self.settings = get_settings()
        self.client = AsyncAnthropic(api_key=self.settings.ANTHROPIC_API_KEY)
        self.model = self.settings.DEFAULT_MODEL
        self.system_prompt = self._load_system_prompt()
        
    def _load_system_prompt(self) -> str:
        """Load the system prompt from file."""
        try:
            prompt_path = Path(self.settings.SYSTEM_PROMPT_PATH)
            with open(prompt_path, "r", encoding="utf-8") as f:
                return f.read().strip()
        except Exception as e:
            print(f"Error loading system prompt: {e}")
            return "You are a helpful assistant."
    
    def _build_context_prompt(self, rag_chunks: List[str]) -> str:
        """Build the context section from RAG chunks."""
        if not rag_chunks:
            return ""
        
        try:
            template_path = Path(self.settings.CONTEXT_PROMPT_PATH)
            with open(template_path, "r", encoding="utf-8") as f:
                template = f.read().strip()
            
            context_text = "\n\n".join(rag_chunks)
            return template.format(context=context_text)
        except Exception as e:
            print(f"Error building context prompt: {e}")
            return ""
    
    def _extract_text(self, content_blocks: List[Any]) -> str:
        """Extract text content from Anthropic response content blocks."""
        return "".join(block["text"] for block in content_blocks if "text" in block)
    
    async def chat(self, session_id: str, user_input: str) -> str:
        """Main entrypoint for handling user queries."""
        try:
            chat_history = await load_chat_history(session_id)
            rag_chunks = await get_relevant_chunks(f"query: {user_input}")
            
            context_prompt = self._build_context_prompt(rag_chunks)
            full_system_prompt = self.system_prompt
            if context_prompt:
                full_system_prompt = f"{self.system_prompt}\n\n---\n\n{context_prompt}"
            
            messages = []
            for msg in chat_history:
                if msg.get("role") in ["user", "assistant"] and msg.get("content"):
                    messages.append({"role": msg["role"], "content": msg["content"]})
            
            messages.append({"role": "user", "content": user_input})
            
            request_params = {
                "model": self.model,
                "max_tokens": 1024,
                "system": full_system_prompt,
                "messages": messages
            }
            
            response = await self.client.messages.create(**request_params)
            response_text = self._extract_text(response.content)
            
            await save_message(session_id, "user", user_input)
            await save_message(session_id, "assistant", response_text)
            
            return response_text.strip()
            
        except Exception as e:
            print(f"Error in chat processing: {e}")
            error_message = "I'm sorry, I encountered an error while processing your request. Please try again."
            
            try:
                await save_message(session_id, "user", user_input)
                await save_message(session_id, "assistant", error_message)
            except Exception as e:
                print(f"Error saving error message to memory: {e}")
            
            return error_message


# Global chat engine instance
chat_engine = ChatEngine()


async def chat(session_id: str, user_input: str) -> str:
    """Convenience function to access the global chat engine."""
    return await chat_engine.chat(session_id, user_input) 