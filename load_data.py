"""
Embedding module for RAG Agent application.

This module handles reading knowledge base markdown files, 
chunking text content, generating embeddings, and persisting 
them into a local FAISS vectorstore.
"""

from pathlib import Path
from typing import List
import argparse
import json
import numpy as np

import faiss
from sentence_transformers import SentenceTransformer

from app.config import get_settings


def read_knowledge_file(file_path: str) -> str:
    """Read the knowledge base markdown file from the specified path."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        print(f"Error: Failed to read knowledge file {file_path}: {str(e)}")
        raise


def split_text_into_chunks(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """
    Split text content into overlapping chunks using a sliding window approach.

    Args:
        text: The input text string.
        chunk_size: The maximum number of characters per chunk.
        chunk_overlap: Number of characters that overlap between chunks.

    Returns:
        A list of text chunks.
    """
    # Normalize whitespace
    text = ' '.join(text.split())
    
    if not text:
        return []

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - chunk_overlap  # Slide window

    return chunks


def build_vectorstore(force_rebuild: bool = False) -> None:
    """
    Main function to build the vectorstore from the knowledge base file.
    
    This function:
    1. Reads the knowledge base markdown file
    2. Splits content into chunks using sliding window approach
    3. Generates embeddings using the configured SentenceTransformer model
    4. Normalizes embeddings for better similarity search
    5. Creates a FAISS IndexFlatL2 index
    6. Persists the index to index.faiss and chunks to chunks.json
    
    Uses settings from get_settings() for all configuration paths.
    """
    try:
        print("Starting vectorstore build process...")
        
        # Get settings
        settings = get_settings()
        
        # Define file paths
        vectorstore_path = Path(settings.VECTORSTORE_PATH)
        index_path = vectorstore_path / "index.faiss"
        chunks_path = vectorstore_path / "chunks.json"
        
        # Check if vectorstore already exists
        if index_path.exists() and chunks_path.exists():
            if force_rebuild:
                print("Force rebuild enabled. Deleting existing vectorstore files...")
                index_path.unlink()
                chunks_path.unlink()
            else:
                print("Vectorstore already exists (index.faiss and chunks.json found). Skipping rebuild.")
                return
        
        # Read knowledge file
        content = read_knowledge_file(settings.KNOWLEDGE_FILE_PATH)
        
        # Split into chunks
        chunks = split_text_into_chunks(content, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)
        print(f"Generated {len(chunks)} chunks from knowledge file.")
        
        if not chunks:
            print("Warning: No chunks found after splitting content")
            return
        
        # Ensure vectorstore directory exists
        vectorstore_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize SentenceTransformer model
        print(f"Loading embedding model: {settings.EMBEDDING_MODEL}")
        model = SentenceTransformer(settings.EMBEDDING_MODEL)
        
        # Generate embeddings with normalization
        print("Generating embeddings...")
        embeddings = model.encode(chunks, normalize_embeddings=True, show_progress_bar=True)
        embeddings = np.array(embeddings, dtype=np.float32)
        
        print(f"Generated embeddings with shape: {embeddings.shape}")
        
        # Create FAISS index (IndexFlatL2 for L2 distance)
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        
        # Add embeddings to index
        index.add(embeddings)
        print(f"Added {index.ntotal} vectors to FAISS index.")
        
        # Save FAISS index
        faiss.write_index(index, str(index_path))
        print(f"Saved FAISS index to: {index_path}")
        
        # Save chunks as JSON
        with open(chunks_path, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(chunks)} chunks to: {chunks_path}")
        
        print("Vectorstore build process completed successfully")
        
    except Exception as e:
        print(f"Error: Vectorstore build process failed: {str(e)}")
        raise 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build vectorstore from knowledge base")
    parser.add_argument("--force", action="store_true", help="Force rebuild by deleting existing vectorstore files")
    args = parser.parse_args()

    build_vectorstore(force_rebuild=args.force)
