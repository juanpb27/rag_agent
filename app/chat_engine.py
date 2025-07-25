"""
Chat engine module for RAG Agent application.
"""

import json
from typing import List, Any, Dict, Optional
from pathlib import Path

from anthropic import AsyncAnthropic

from app.config import get_settings
from app.retriever import get_relevant_chunks
from app.memory import load_chat_history, save_message
from app.tools import TOOL_SCHEMAS, TOOL_FUNCTIONS_MAP


class ChatEngine:
    """Main chat engine that orchestrates the RAG agent conversation flow."""
    
    def __init__(self):
        """Initialize the chat engine with Anthropic client."""
        self.settings = get_settings()
        self.client = AsyncAnthropic(api_key=self.settings.ANTHROPIC_API_KEY)
        self.model = self.settings.DEFAULT_MODEL
        self.max_tokens = self.settings.MAX_TOKENS
        self.system_prompt_template = self._load_system_prompt()
        
    def _load_system_prompt(self) -> str:
        """Load the system prompt from file."""
        try:
            prompt_path = Path(self.settings.SYSTEM_PROMPT_PATH)
            with open(prompt_path, "r", encoding="utf-8") as f:
                return f.read().strip()
        except Exception as e:
            print(f"Error loading system prompt: {e}")
            return "You are a helpful assistant."
    
    def _extract_text(self, content_blocks: List[Any]) -> str:
        """Extract text content from Anthropic response content blocks."""
        return "".join(block.text for block in content_blocks if hasattr(block, "text"))

    def execute_tool_call(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool call and return the result."""
        try:
            if tool_name not in TOOL_FUNCTIONS_MAP:
                return {"error": f"Tool '{tool_name}' not found"}
            
            tool_function = TOOL_FUNCTIONS_MAP[tool_name]
            result = tool_function(**parameters)
            return result
        except Exception as e:
            return {"error": f"Error executing tool '{tool_name}': {str(e)}"}

    def _has_tool_use(self, content_blocks: List[Any]) -> Optional[Dict[str, Any]]:
        """Check if response contains tool use and return tool details."""
        for block in content_blocks:
            if hasattr(block, "type") and block.type == "tool_use":
                return {
                    "name": block.name,
                    "parameters": block.input,
                    "id": getattr(block, "id", None)
                }
        return None
    
    def _prepare_messages(self, chat_history: List[dict], user_input: str) -> List[dict]:
        """Prepare messages array with chat history and current user input."""
        messages = []
        
        # Add previous conversation history
        for msg in chat_history:
            if msg.get("role") in ["user", "assistant", "tool"] and msg.get("content"):
                messages.append({
                    "role": msg["role"], 
                    "content": msg["content"]
                })
        
        # Add current user input
        messages.append({"role": "user", "content": user_input})
        
        return messages
 
    async def chat(self, session_id: str, user_input: str) -> str:
        """Main entrypoint for handling user queries."""
        try:
            chat_history = load_chat_history(session_id)
            rag_chunks = get_relevant_chunks(f"query: {user_input}")

            if not rag_chunks:
                print("No relevant chunks found for the user query.")
            
            # Prepare the retrieved chunks text
            retrieved_chunks_text = "\n\n".join(rag_chunks) if rag_chunks else ""
            
            # Replace placeholders in the system prompt template
            full_system_prompt = self.system_prompt_template.format(
                retrieved_chunks=retrieved_chunks_text,
                user_query=user_input
            )
            
            # Prepare messages with chat history
            messages = self._prepare_messages(chat_history, user_input)

            print(f"Full system prompt: {full_system_prompt}")
            print(f"Messages: {messages}")

            request_params = {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "system": full_system_prompt,
                "messages": messages,
                "tools": TOOL_SCHEMAS
            }
            
            response = await self.client.messages.create(**request_params)
            
            # Check if response contains tool use
            tool_use = self._has_tool_use(response.content)
            
            if tool_use:
                print(f"🔧 TOOL SELECTED: {tool_use['name']}")
                print(f"📝 PARAMETERS: {json.dumps(tool_use['parameters'], indent=2)}")
                
                # Execute the tool
                tool_result = self.execute_tool_call(tool_use["name"], tool_use["parameters"])
                
                # Add assistant response with tool_use
                messages.append({
                    "role": "assistant", 
                    "content": response.content
                })
                
                # Add tool result as tool_result block
                messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_use["id"],
                            "content": json.dumps(tool_result)
                        }
                    ]
                })
                
                # Make second call to get final response
                final_request_params = {
                    "model": self.model,
                    "max_tokens": self.max_tokens,
                    "system": full_system_prompt,
                    "messages": messages,
                    "tools": TOOL_SCHEMAS
                }
                
                final_response = await self.client.messages.create(**final_request_params)
                response_text = self._extract_text(final_response.content)
                
                # Save messages including tool interaction
                save_message(session_id, "user", user_input)
                save_message(session_id, "assistant", f"[Used tool: {tool_use['name']}] {response_text}")
                
            else:
                # Regular response without tools
                response_text = self._extract_text(response.content)
                save_message(session_id, "user", user_input)
                save_message(session_id, "assistant", response_text)
            
            return response_text.strip()
            
        except Exception as e:
            print(f"Error in chat processing: {e}")
            error_message = "I'm sorry, I encountered an error while processing your request. Please try again."
            
            try:
                save_message(session_id, "user", user_input)
                save_message(session_id, "assistant", error_message)
            except Exception as e:
                print(f"Error saving error message to memory: {e}")
            
            return error_message


# Global chat engine instance
chat_engine = ChatEngine()


async def chat(session_id: str, user_input: str) -> str:
    """Convenience function to access the global chat engine."""
    return await chat_engine.chat(session_id, user_input) 