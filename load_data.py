"""
Embedding module for RAG Agent application.

This module handles reading knowledge base markdown files, 
chunking text content, generating embeddings, and persisting 
them into a local Chroma vectorstore.
"""

from pathlib import Path
from typing import List

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from app.config import get_settings


def read_knowledge_file(file_path: str) -> str:
    """Read the knowledge base markdown file from the specified path."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            return content
    except FileNotFoundError:
        print(f"Error: Knowledge file not found: {file_path}")
        raise
    except IOError as e:
        print(f"Error: Failed to read knowledge file {file_path}: {str(e)}")
        raise


def split_text_into_chunks(text: str, chunk_size: int) -> List[str]:
    """Split text content into chunks using sliding window approach with max character length."""
    
    # Remove excessive whitespace and normalize text
    text = ' '.join(text.split())
    
    if not text.strip():
        return []
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        # Get chunk of specified size
        end = start + chunk_size
        
        # If we're not at the end, try to break at word boundary
        if end < len(text):
            # Look for last space within the chunk to avoid breaking words
            last_space = text.rfind(' ', start, end)
            if last_space > start:
                end = last_space
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        # Move start position (sliding window)
        start = end + 1 if end < len(text) else len(text)
    
    return chunks


def build_vectorstore() -> None:
    """
    Main function to build the vectorstore from the knowledge base file.
    
    This function:
    1. Reads the knowledge base markdown file
    2. Splits content into chunks using sliding window approach
    3. Generates embeddings using the configured model
    4. Persists embeddings into a local Chroma vectorstore
    5. Creates the 'docs' collection
    
    Uses settings from get_settings() for all configuration paths.
    """
    try:
        print("Starting vectorstore build process")
        
        # Get settings
        settings = get_settings()
        
        # Read knowledge file
        content = read_knowledge_file(settings.KNOWLEDGE_FILE_PATH)
        
        # Split into chunks
        chunks = split_text_into_chunks(content, settings.CHUNK_SIZE)
        
        if not chunks:
            print("Warning: No chunks found after splitting content")
            return
        
        # Ensure vectorstore directory exists
        Path(settings.VECTORSTORE_PATH).mkdir(parents=True, exist_ok=True)
        
        # Initialize Chroma client
        client = chromadb.PersistentClient(path=settings.VECTORSTORE_PATH)
        
        # Setup embedding function
        embedding_function = SentenceTransformerEmbeddingFunction(model_name=settings.EMBEDDING_MODEL)
        
        # Create collection
        collection = client.create_collection(name="docs", embedding_function=embedding_function)
        
        # Generate numeric IDs for chunks
        ids = [str(i) for i in range(len(chunks))]
        
        # Add chunks to collection
        collection.add(documents=chunks, ids=ids)
        
        print("Vectorstore build process completed successfully")
        
    except Exception as e:
        print(f"Error: Vectorstore build process failed: {str(e)}")
        raise 