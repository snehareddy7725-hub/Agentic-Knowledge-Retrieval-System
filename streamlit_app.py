"""
Streamlit UI for Agentic RAG System.
Provides a chat interface with document upload and status monitoring.
"""

import streamlit as st
import sys
from pathlib import Path
import uuid

# Add the parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from app.config import DOCS_DIR
from app.embedding import setup_embeddings
from app.vector_store import setup_vector_store
from app.document_processor import process_documents
from app.agent import setup_agent, get_agent_response
from app.utils import create_sample_documents, convert_to_markdown

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="Agentic RAG System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #6C63FF;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #F0F2F6;
        border-left: 4px solid #6C63FF;
    }
    .assistant-message {
        background-color: #FFFFFF;
        border: 1px solid #E0E0E0;
        border-left: 4px solid #2ECC71;
    }
    .source-reference {
        font-size: 0.85rem;
        color: #666;
        background-color: #F8F9FA;
        padding: 0.5rem;
        border-radius: 0.25rem;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# SESSION STATE INITIALIZATION
# ============================================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "agent_initialized" not in st.session_state:
    st.session_state.agent_initialized = False
if "thread_id" not in st.session_state:
    import uuid
    st.session_state.thread_id = str(uuid.uuid4())
if "documents_processed" not in st.session_state:
    st.session_state.documents_processed = False
# NEW: track the last file we already handled, so the uploader
# widget "remembering" the file across reruns doesn't keep
# resetting our ready-state flags back to False.
if "last_uploaded_name" not in st.session_state:
    st.session_state.last_uploaded_name = None

# ============================================
# SIDEBAR - Configuration & Status
# ============================================
with st.sidebar:
    st.markdown("### 🤖 Agentic RAG")
    st.markdown("---")
    
    # Status indicators
    st.markdown("#### 📊 System Status")
    
    if st.session_state.agent_initialized:
        st.success("✅ Agent Ready")
    else:
        st.warning("⏳ Initializing...")
    
    if st.session_state.documents_processed:
        st.success("✅ Documents Loaded")
    else:
        st.warning("⏳ No Documents")
    
    st.markdown("---")
    
    # Document Management
    st.markdown("#### 📄 Document Management")
    
    if st.button("📥 Load Sample Documents"):
        with st.spinner("Creating sample documents..."):
            create_sample_documents()
            convert_to_markdown()
            st.session_state.documents_processed = False
            st.session_state.agent_initialized = False
            st.success("✅ Sample documents created!")
            st.rerun()
    
    uploaded_file = st.file_uploader(
        "Upload Document",
        type=["pdf", "md", "txt"],
        help="Upload PDF, Markdown, or Text files"
    )
    
    if uploaded_file:
        # Only treat this as a NEW upload if we haven't already
        # processed a file with this exact name. Without this check,
        # Streamlit keeps re-submitting the same uploaded_file on every
        # rerun (e.g. after clicking "Process Documents"), which would
        # silently flip agent_initialized / documents_processed back to
        # False right after they were just set to True.
        if uploaded_file.name != st.session_state.last_uploaded_name:
            file_path = DOCS_DIR / uploaded_file.name
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"✅ Uploaded: {uploaded_file.name}")
            st.session_state.documents_processed = False
            st.session_state.agent_initialized = False
            st.session_state.last_uploaded_name = uploaded_file.name
    
    if st.button("🔄 Process Documents", type="primary"):
        with st.spinner("Processing documents..."):
            try:
                # Convert only the file(s) uploaded this session —
                # not everything sitting in DOCS_DIR. Falls back to
                # converting everything only if nothing was uploaded
                # (e.g. sample documents were loaded instead).
                if st.session_state.last_uploaded_name:
                    convert_to_markdown(filenames=[st.session_state.last_uploaded_name])
                else:
                    convert_to_markdown()
                
                # Setup embeddings and vector store
                dense, sparse = setup_embeddings()
                client, vector_store = setup_vector_store(dense, sparse)
                
                # Process and index documents
                count = process_documents(vector_store)
                st.session_state.documents_processed = True
                st.session_state._vector_store = vector_store
                st.session_state._client = client
                
                # Initialize agent
                agent_graph, llm = setup_agent(vector_store)
                st.session_state._agent_graph = agent_graph
                st.session_state.agent_initialized = True
                
                st.success(f"✅ Processed {count} documents!")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    
    st.markdown("---")
    
    # Statistics
    st.markdown("#### 📊 Statistics")
    if st.session_state.documents_processed:
        # Count documents
        import glob
        doc_count = len(glob.glob(str(DOCS_DIR / "*.*")))
        st.metric("Documents", doc_count)
        
        # Check vector store
        if hasattr(st.session_state, '_vector_store'):
            st.metric("Status", "Hybrid Search Active")
    
    st.markdown("---")
    st.caption("💡 Tip: Upload documents or use sample data to get started!")

# ============================================
# MAIN CHAT INTERFACE
# ============================================
st.markdown('<div class="main-header">🤖 Agentic RAG Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Ask questions about your documents • Powered by LangGraph + Hybrid Search</div>', unsafe_allow_html=True)

# Check if agent is ready
if not st.session_state.agent_initialized:
    st.info("👋 Welcome! Please upload documents or load sample data, then click 'Process Documents' to get started.")
    
    # Show quick start guide
    with st.expander("🚀 Quick Start Guide"):
        st.markdown("""
        1. **Upload Documents** (PDF, MD, TXT) or click "Load Sample Documents"
        2. **Process Documents** - This indexes your documents for search
        3. **Start Chatting** - Ask questions about your documents!
        
        ### Sample Questions to Try:
        - "What are the key technologies in AI?"
        - "Explain machine learning"
        - "What are the applications of data science?"
        """)
else:
    # Display chat history
    # Display chat history
    for message in st.session_state.messages:
        role = message["role"]
        content = message["content"]

        if role == "user":
            with st.chat_message("user", avatar="🧑"):
                st.markdown(content)
        else:
            sources = message.get("sources", [])
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(content)
                if sources:
                    st.caption(f"📚 Sources: {', '.join(sources)}")
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Generate response
        with st.spinner("🤔 Thinking..."):
            try:
                agent_graph = st.session_state._agent_graph
                response = get_agent_response(
                    agent_graph,
                    prompt,
                    st.session_state.thread_id
                )
                
                # Extract sources from the response (if any)
                # In a full implementation, you'd track sources properly
                sources = []  # Placeholder
                
                # Add assistant message
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "sources": sources
                })
                
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    
    # Clear chat button
    if st.button("🗑️ Clear Chat", type="secondary"):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()

# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.caption("⚡ Built with LangChain, LangGraph, Qdrant, and Streamlit")