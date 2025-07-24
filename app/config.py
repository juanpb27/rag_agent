"""
Configuration module for RAG Agent application.
"""

from pydantic import BaseSettings, Field
from dotenv import load_dotenv

load_dotenv(override=True)


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    """
    
    ANTHROPIC_API_KEY: str = Field(
        ..., 
        description="Anthropic API key for AI models"
    )

    VECTORSTORE_PATH: str = Field(
        default="data/vectorstore",
        description="Path to the vectorstore directory"
    )
    
    KNOWLEDGE_FILE_PATH: str = Field(
        default="data/knowledge.md",
        description="Path to the knowledge base markdown file"
    )
    
    CHUNK_SIZE: int = Field(
        default=500,
        ge=100,
        le=2000,
        description="Size of text chunks for processing (between 100-2000 characters)"
    )
    
    EMBEDDING_MODEL: str = Field(
        default="BAAI/bge-small-en-v1.5",
        description="Sentence transformer model name for embeddings"
    )
    
    DEFAULT_MODEL: str = Field(
        default="claude-3-haiku-20240307",
        description="Default Anthropic model to use for chat completions"
    )
    
    SYSTEM_PROMPT_PATH: str = Field(
        default="app/prompts/system_prompt.txt",
        description="Path to the system prompt file"
    )
    
    CONTEXT_PROMPT_PATH: str = Field(
        default="app/prompts/context_template.txt",
        description="Path to the context prompt template file"
    )
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    return settings 