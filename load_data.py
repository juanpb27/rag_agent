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
    Split text content into overlapping chunks using recursive chunking with natural separators.
    
    Uses priority order of separators:
    1. Double newlines (\n\n) - paragraph breaks
    2. Single newlines (\n) - line breaks  
    3. Periods (.) - sentence boundaries
    4. Spaces (" ") - word boundaries
    
    Preserves semantic boundaries while maintaining chunk size limits and overlap.
    """
    if not text:
        return []
    
    # Normalize whitespace but preserve paragraph breaks
    text = text.strip()
    
    def recursive_split(text_segment: str, separators: List[str]) -> List[str]:
        """Recursively split text using natural separators in priority order."""
        if len(text_segment) <= chunk_size:
            return [text_segment.strip()] if text_segment.strip() else []
        
        if not separators:
            # Fallback: force split at chunk_size to avoid infinite chunks
            chunks = []
            for i in range(0, len(text_segment), chunk_size):
                chunk = text_segment[i:i + chunk_size].strip()
                if chunk:
                    chunks.append(chunk)
            return chunks
        
        separator = separators[0]
        remaining_separators = separators[1:]
        
        # Split by current separator
        parts = text_segment.split(separator)
        
        if len(parts) == 1:
            # Current separator not found, try next separator
            return recursive_split(text_segment, remaining_separators)
        
        chunks = []
        current_chunk = ""
        
        for i, part in enumerate(parts):
            # Reconstruct with separator (except for last part)
            if i < len(parts) - 1:
                part_with_sep = part + separator
            else:
                part_with_sep = part
            
            # Check if adding this part would exceed chunk size
            if current_chunk and len(current_chunk + part_with_sep) > chunk_size:
                # Current chunk is ready, process it
                if current_chunk.strip():
                    # If current chunk is still too large, recursively split it
                    if len(current_chunk) > chunk_size:
                        chunks.extend(recursive_split(current_chunk, remaining_separators))
                    else:
                        chunks.append(current_chunk.strip())
                
                # Start new chunk with current part
                current_chunk = part_with_sep
            else:
                # Add part to current chunk
                current_chunk += part_with_sep
        
        # Process final chunk
        if current_chunk.strip():
            if len(current_chunk) > chunk_size:
                chunks.extend(recursive_split(current_chunk, remaining_separators))
            else:
                chunks.append(current_chunk.strip())
        
        return chunks
    
    # Define separators in priority order
    separators = ["\n\n", "\n", ".", " "]
    
    # Get initial chunks without overlap
    initial_chunks = recursive_split(text, separators)
    
    if not initial_chunks:
        return []
    
    # Apply overlap between chunks
    final_chunks = []
    
    for i, chunk in enumerate(initial_chunks):
        if i == 0:
            # First chunk - no previous overlap needed
            final_chunks.append(chunk)
        else:
            # Add overlap from previous chunk
            prev_chunk = initial_chunks[i-1]
            
            # Get overlap from end of previous chunk
            overlap_text = ""
            if len(prev_chunk) > chunk_overlap:
                overlap_text = prev_chunk[-chunk_overlap:]
            else:
                overlap_text = prev_chunk
            
            # Combine overlap with current chunk, but respect chunk_size limit
            combined = overlap_text + " " + chunk
            if len(combined) <= chunk_size:
                final_chunks.append(combined.strip())
            else:
                # If combined exceeds limit, just use current chunk
                final_chunks.append(chunk)
    
    # Filter out empty chunks
    return [chunk for chunk in final_chunks if chunk.strip()]


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
        
        # Generate embeddings with normalization and instruction prefix
        print("Generating embeddings...")
        # Add instruction prefix for document chunks as recommended by BGE model authors
        chunks_with_prefix = ["Represent this document for retrieval: " + chunk for chunk in chunks]
        embeddings = model.encode(chunks_with_prefix, normalize_embeddings=True, show_progress_bar=True)
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
