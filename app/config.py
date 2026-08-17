"""
Configuration settings for the Agentic RAG system.
"""

import os
from pathlib import Path

# ============================================
# BASE DIRECTORIES
# ============================================
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"

# ============================================
# DIRECTORY CONFIGURATION
# ============================================
DOCS_DIR = DATA_DIR / "docs"
MARKDOWN_DIR = DATA_DIR / "markdown"
PARENT_STORE_PATH = DATA_DIR / "parent_store"

# ============================================
# DATABASE CONFIGURATION
# ============================================
CHILD_COLLECTION = "document_child_chunks"

# ============================================
# EMBEDDING CONFIGURATION
# ============================================
DENSE_EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
SPARSE_EMBEDDING_MODEL = "Qdrant/bm25"

# ============================================
# LLM CONFIGURATION
# ============================================
LLM_MODEL = "llama3.1"

# ============================================
# CHUNKING CONFIGURATION
# ============================================
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
MIN_PARENT_SIZE = 2000
MAX_PARENT_SIZE = 10000

# ============================================
# RETRIEVAL CONFIGURATION
# ============================================
TOP_K = 5
SCORE_THRESHOLD = 0.3

# ============================================
# Create directories (with error handling)
# ============================================
def create_directories():
    """Create all required directories with proper error handling"""
    directories = [DOCS_DIR, MARKDOWN_DIR, PARENT_STORE_PATH]
    
    for directory in directories:
        try:
            directory.mkdir(parents=True, exist_ok=True)
            print(f"✅ Directory ready: {directory}")
        except PermissionError:
            print(f"⚠️ Permission denied: {directory}")
        except Exception as e:
            print(f"⚠️ Error creating {directory}: {e}")

# Create directories when module loads
create_directories()
