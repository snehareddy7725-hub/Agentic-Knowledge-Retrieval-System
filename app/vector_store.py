"""
Vector database setup with Qdrant (in-memory mode).
No persistent storage - avoids file locking issues.
"""

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from langchain_qdrant import QdrantVectorStore
from langchain_qdrant.qdrant import RetrievalMode
from app.config import CHILD_COLLECTION

def setup_vector_store(dense_embeddings, sparse_embeddings):
    """
    Initialize Qdrant vector database IN-MEMORY.
    This avoids file locking issues during development.
    
    Args:
        dense_embeddings: Semantic embeddings
        sparse_embeddings: Keyword embeddings
    
    Returns:
        tuple: (client, vector_store)
    """
    
    print("🗄️ Setting up vector database (in-memory mode)...")
    
    # Use in-memory mode (no file locking issues!)
    client = QdrantClient(":memory:")
    
    # Get embedding dimension
    embedding_dimension = len(dense_embeddings.embed_query("test"))
    
    # Create collection
    if not client.collection_exists(CHILD_COLLECTION):
        client.create_collection(
            collection_name=CHILD_COLLECTION,
            vectors_config=qmodels.VectorParams(
                size=embedding_dimension,
                distance=qmodels.Distance.COSINE
            ),
            sparse_vectors_config={
                "sparse": qmodels.SparseVectorParams()
            },
        )
        print(f"✅ Created collection: {CHILD_COLLECTION}")
    else:
        print(f"✅ Collection exists: {CHILD_COLLECTION}")
    
    # Create vector store with hybrid search
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=CHILD_COLLECTION,
        embedding=dense_embeddings,
        sparse_embedding=sparse_embeddings,
        retrieval_mode=RetrievalMode.HYBRID,
        sparse_vector_name="sparse"
    )
    
    print("✅ Vector store ready with hybrid search (in-memory)")
    return client, vector_store
