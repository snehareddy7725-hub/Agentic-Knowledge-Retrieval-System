"""
Minimal Streamlit UI for Agentic RAG System.
"""

import streamlit as st
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from app.embedding import setup_embeddings
from app.vector_store import setup_vector_store
from app.document_processor import process_documents
from app.agent import setup_agent, get_agent_response
from app.utils import create_sample_documents, convert_to_markdown

st.set_page_config(page_title="Agentic RAG", page_icon="🤖", layout="wide")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "agent_initialized" not in st.session_state:
    st.session_state.agent_initialized = False
if "thread_id" not in st.session_state:
    import uuid
    st.session_state.thread_id = str(uuid.uuid4())

# Sidebar
with st.sidebar:
    st.markdown("### 🤖 Agentic RAG")
    
    if st.session_state.agent_initialized:
        st.success("✅ Agent Ready")
    else:
        st.warning("⏳ Not Ready")
    
    if st.button("📥 Load Sample Documents"):
        with st.spinner("Creating sample documents..."):
            create_sample_documents()
            convert_to_markdown()
            st.success("✅ Sample documents created!")
            st.rerun()
    
    if st.button("🔄 Process Documents", type="primary"):
        with st.spinner("Processing documents..."):
            try:
                convert_to_markdown()
                dense, sparse = setup_embeddings()
                client, vector_store = setup_vector_store(dense, sparse)
                count = process_documents(vector_store)
                agent_graph, llm = setup_agent(vector_store)
                st.session_state._agent_graph = agent_graph
                st.session_state.agent_initialized = True
                st.success(f"✅ Processed {count} documents!")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

# Main area
st.markdown("## 🤖 Agentic RAG Assistant")

if st.session_state.agent_initialized:
    # Display chat messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        
        # Generate response
        with st.spinner("🤔 Thinking..."):
            try:
                agent_graph = st.session_state._agent_graph
                response = get_agent_response(agent_graph, prompt, st.session_state.thread_id)
                st.session_state.messages.append({"role": "assistant", "content": response})
                with st.chat_message("assistant"):
                    st.write(response)
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
else:
    st.info("👋 Welcome! Please load sample documents or upload your own, then click 'Process Documents' to start chatting.")
    
    with st.expander("📖 Quick Start Guide"):
        st.markdown("""
        1. Click **'Load Sample Documents'** in the sidebar
        2. Click **'Process Documents'** 
        3. Wait for processing to complete
        4. Start asking questions!
        
        **Sample questions to try:**
        - "What is artificial intelligence?"
        - "Explain machine learning"
        - "What are the applications of AI?"
        """)
