"""
Pydantic models for data validation.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

# ============================================
# CHAT MODELS
# ============================================

class ChatMessage(BaseModel):
    """Single chat message"""
    role: str = Field(..., description="'user' or 'assistant'")
    content: str = Field(..., description="Message content")
    timestamp: datetime = Field(default_factory=datetime.now)

class ChatHistory(BaseModel):
    """Conversation history"""
    thread_id: str
    messages: List[ChatMessage] = []

# ============================================
# DOCUMENT MODELS
# ============================================

class DocumentInfo(BaseModel):
    """Information about a document"""
    filename: str
    size: int
    chunks: int
    uploaded_at: datetime

# ============================================
# SEARCH MODELS
# ============================================

class SearchResult(BaseModel):
    """Individual search result"""
    content: str
    source: str
    score: float
    metadata: Dict[str, Any]

# ============================================
# AGENT STATE
# ============================================

class AgentStatus(BaseModel):
    """Current status of the agent"""
    is_ready: bool = False
    has_documents: bool = False
    document_count: int = 0
    vector_store_size: int = 0
    message: str = ""