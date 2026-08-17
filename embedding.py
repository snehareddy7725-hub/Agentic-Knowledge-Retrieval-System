"""
Embedding setup for hybrid search (dense + sparse).
Combines semantic understanding with keyword matching.
"""

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant.fastembed_sparse import FastEmbedSparse
from app.config import DENSE_EMBEDDING_MODEL, SPARSE_EMBEDDING_MODEL

def setup_embeddings():
    """
    Initialize embeddings for hybrid search.
    
    Returns:
        tuple: (dense_embeddings, sparse_embeddings)
    
    Dense: Semantic understanding using neural networks
    Sparse: Keyword matching using BM25 algorithm
    
    Example:
        dense, sparse = setup_embeddings()
        vector = dense.embed_query("What is AI?")  # 768-dimensional vector
    """
    
    print("📊 Loading dense embeddings...")
    # Dense embeddings: Understands meaning and semantics
    # - all-mpnet-base-v2: 768 dimensions, high quality
    # - Trained on 1B+ sentences
    dense_embeddings = HuggingFaceEmbeddings(
        model_name=DENSE_EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},  # Use CPU (or 'cuda' for GPU)
        encode_kwargs={'normalize_embeddings': True}  # For cosine similarity
    )
    print("✅ Dense embeddings loaded")
    
    print("📊 Loading sparse embeddings...")
    # Sparse embeddings: Matches exact keywords
    # - BM25: Classic information retrieval algorithm
    # - Good for matching specific terms like "fraud detection"
    sparse_embeddings = FastEmbedSparse(
        model_name=SPARSE_EMBEDDING_MODEL
    )
    print("✅ Sparse embeddings loaded")
    
    return dense_embeddings, sparse_embeddings