"""
RAG retriever module for getting relevant document chunks from FAISS.
"""

from typing import List
import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import get_settings


# Global variables for caching
_model = None
_index = None
_chunks = None


def _load_vectorstore():
    """Load FAISS index and chunks if not already loaded."""
    global _model, _index, _chunks
    
    if _model is None or _index is None or _chunks is None:
        settings = get_settings()
        
        # Define file paths
        vectorstore_path = Path(settings.VECTORSTORE_PATH)
        index_path = vectorstore_path / "index.faiss"
        chunks_path = vectorstore_path / "chunks.json"
        
        # Check if files exist
        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index not found at {index_path}. Please run load_data.py first.")
        
        if not chunks_path.exists():
            raise FileNotFoundError(f"Chunks file not found at {chunks_path}. Please run load_data.py first.")
        
        # Load SentenceTransformer model
        _model = SentenceTransformer(settings.EMBEDDING_MODEL)
        
        # Load FAISS index
        _index = faiss.read_index(str(index_path))
        
        # Load chunks
        with open(chunks_path, 'r', encoding='utf-8') as f:
            _chunks = json.load(f)
        
        print(f"Loaded FAISS index with {_index.ntotal} vectors and {len(_chunks)} chunks.")


def get_relevant_chunks(query: str) -> List[str]:
    """
    Retrieve relevant document chunks from FAISS based on the query.
    
    Args:
        query: The search query string
        
    Returns:
        List of relevant document chunks (strings)
    """
    try:
        # Load vectorstore components if not already loaded
        _load_vectorstore()
        
        # Get settings
        settings = get_settings()
        
        # Generate normalized embedding for the query
        query_embedding = _model.encode([query], normalize_embeddings=True)
        query_embedding = np.array(query_embedding, dtype=np.float32)
        
        # Validate embedding dimension matches FAISS index
        if query_embedding.shape[1] != _index.d:
            raise ValueError(f"Query embedding dimension ({query_embedding.shape[1]}) does not match FAISS index dimension ({_index.d})")
        
        # Perform similarity search
        distances, indices = _index.search(query_embedding, settings.TOP_K)
        
        # Extract the relevant chunks
        relevant_chunks: List[str] = []
        for idx in indices[0]:  # indices[0] because we only have one query
            if idx < len(_chunks):  # Ensure index is valid
                relevant_chunks.append(_chunks[idx])
        
        return relevant_chunks
        
    except Exception as e:
        print(f"Error retrieving chunks: {e}")
        return []
