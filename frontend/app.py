"""Streamlit frontend for the Agentic RAG System using API communication."""

import os
import logging
import requests
from typing import List, Dict, Any, Optional
import streamlit as st
from streamlit_chat import message
import time

# Configure Streamlit page
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
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
        color: #333;
    }
    
    .upload-section {
        background-color: #f8f9fa;
        padding: 2rem;
        border-radius: 10px;
        border: 2px dashed #dee2e6;
        margin-bottom: 2rem;
    }
    
    .status-box {
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    
    .status-success {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    
    .status-warning {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
    }
    
    .status-error {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
    }
    
    .file-info {
        font-size: 0.9rem;
        color: #666;
        margin-top: 0.5rem;
    }
    
    .metric-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Backend API configuration
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# Session state initialization
def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []
    
    if 'processing_status' not in st.session_state:
        st.session_state.processing_status = {}
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'current_query' not in st.session_state:
        st.session_state.current_query = ""
    
    if 'backend_healthy' not in st.session_state:
        st.session_state.backend_healthy = False
    
    if 'system_config' not in st.session_state:
        st.session_state.system_config = {}

# API interaction functions
def check_backend_health():
    """Check if the backend API is running."""
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

def get_system_config():
    """Get system configuration from backend."""
    try:
        response = requests.get(f"{BACKEND_URL}/config", timeout=5)
        if response.status_code == 200:
            return response.json()
        return {}
    except requests.exceptions.RequestException as e:
        logger.error(f"Error getting config: {e}")
        return {}

def get_system_status():
    """Get system status from backend."""
    try:
        response = requests.get(f"{BACKEND_URL}/status", timeout=5)
        if response.status_code == 200:
            return response.json()
        return {"status": "error", "components": {}, "storage_info": {}}
    except requests.exceptions.RequestException as e:
        logger.error(f"Error getting system status: {e}")
        return {"status": "error", "components": {}, "storage_info": {}}

def upload_files_to_backend(files):
    """Upload files to the backend API."""
    if not files:
        return []
    
    results = []
    
    try:
        # Prepare files for upload
        files_data = []
        for uploaded_file in files:
            files_data.append(
                ("files", (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type))
            )
        
        # Upload to backend
        response = requests.post(f"{BACKEND_URL}/upload", files=files_data, timeout=180)
        
        if response.status_code == 200:
            result = response.json()
            results.extend(result.get("results", []))
        else:
            for uploaded_file in files:
                results.append({
                    "filename": uploaded_file.name,
                    "status": "error",
                    "message": f"HTTP {response.status_code}: {response.text}"
                })
                
    except Exception as e:
        logger.error(f"Error uploading files: {e}")
        for uploaded_file in files:
            results.append({
                "filename": uploaded_file.name,
                "status": "error",
                "message": str(e)
            })
    
    return results

def query_backend(question, conversation_history=None):
    """Query the backend API with a question."""
    try:
        payload = {
            "question": question,
            "conversation_history": conversation_history or []
        }
        
        response = requests.post(
            f"{BACKEND_URL}/query",
            json=payload,
            timeout=180
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {
                "error": f"HTTP {response.status_code}: {response.text}",
                "status": "error"
            }
            
    except Exception as e:
        logger.error(f"Error querying backend: {e}")
        return {
            "error": str(e),
            "status": "error"
        }

def clear_system():
    """Clear the system via backend API."""
    try:
        response = requests.delete(f"{BACKEND_URL}/clear", timeout=30)
        return response.status_code == 200
    except Exception as e:
        logger.error(f"Error clearing system: {e}")
        return False

# UI Components
def display_system_status():
    """Display system status in the sidebar."""
    st.sidebar.subheader("⚙️ System Status")
    
    # Check backend health
    backend_healthy = check_backend_health()
    st.session_state.backend_healthy = backend_healthy
    
    if backend_healthy:
        st.sidebar.success(" Backend API: Online")
        
        # Get and cache system config
        if not st.session_state.system_config:
            st.session_state.system_config = get_system_config()
        
        config = st.session_state.system_config
        if config:
            st.sidebar.write(f"**LLM:** {config.get('llm_provider', 'Unknown')} - {config.get('llm_model', 'Unknown')}")
            st.sidebar.write(f"**Embedding:** {config.get('embedding_model', 'Unknown')}")
            
        
        # Get system status
        status_info = get_system_status()
        components = status_info.get("components", {})
        storage_info = status_info.get("storage_info", {})
        
        st.sidebar.write("**Components:**")
        for component, status in components.items():
            icon = "" if status else ""
            st.sidebar.write(f"{icon} {component.replace('_', ' ').title()}")
        
        if storage_info:
            st.sidebar.write(f"**Storage:** {storage_info.get('storage_type', 'unknown')}")
            st.sidebar.write(f"**Documents:** {storage_info.get('total_entities', 0)}")
    else:
        st.sidebar.error(" Backend API: Offline")
        st.sidebar.info("Make sure the backend server is running on port 8000")

def display_file_upload():
    """Display file upload section."""
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    
    uploaded_files = st.file_uploader(
        "📁 Choose files to upload",
        type=['pdf', 'docx', 'pptx', 'csv', 'xlsx', 'xls', 'txt', 'md'],
        accept_multiple_files=True,
        help="Supported formats: PDF, DOCX, PPTX, CSV, Excel, TXT, MD",
        key="file_uploader"
    )
    
    if uploaded_files and st.session_state.backend_healthy:
        if st.button("🚀 Process Files", type="primary"):
            handle_file_upload(uploaded_files)
    elif uploaded_files and not st.session_state.backend_healthy:
        st.error(" Cannot upload files: Backend API is not available")
    
    st.markdown('</div>', unsafe_allow_html=True)

def handle_file_upload(uploaded_files: List[Any]):
    """Handle file upload and processing."""
    if not uploaded_files:
        return
    
    # Filter new files
    new_files = []
    for uploaded_file in uploaded_files:
        if uploaded_file.name not in [f["name"] for f in st.session_state.uploaded_files]:
            new_files.append(uploaded_file)
    
    if not new_files:
        st.info("ℹ️ No new files to process.")
        return
    
    # Process files
    with st.spinner(f"📤 Uploading and processing {len(new_files)} files..."):
        results = upload_files_to_backend(new_files)
    
    # Update session state
    for i, uploaded_file in enumerate(new_files):
        file_info = {
            "name": uploaded_file.name,
            "size": uploaded_file.size,
            "type": uploaded_file.type
        }
        st.session_state.uploaded_files.append(file_info)
        
        # Update processing status
        if i < len(results):
            result = results[i]
            if result["status"] == "success":
                st.session_state.processing_status[uploaded_file.name] = {
                    'status': 'completed',
                    'chunks': result.get('chunks_processed', 0),
                    'message': result.get('message', 'Processed successfully')
                }
                st.success(f" {uploaded_file.name}: {result.get('chunks_processed', 0)} chunks")
            else:
                st.session_state.processing_status[uploaded_file.name] = {
                    'status': 'error',
                    'message': result.get('message', 'Unknown error')
                }
                st.error(f" {uploaded_file.name}: {result.get('message')}")

def display_file_status():
    """Display status of uploaded files."""
    if not st.session_state.uploaded_files:
        st.info(" No files uploaded yet. Upload documents to start building your knowledge base.")
        return
    
    st.subheader("File Processing Status")
    
    # Summary metrics
    total_files = len(st.session_state.uploaded_files)
    processed_files = sum(1 for status in st.session_state.processing_status.values() 
                         if status.get('status') == 'completed')
    total_chunks = sum(status.get('chunks', 0) for status in st.session_state.processing_status.values() 
                      if status.get('status') == 'completed')
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="metric-box">
            <h3>{total_files}</h3>
            <p>Total Files</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-box">
            <h3>{processed_files}</h3>
            <p>Processed</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-box">
            <h3>{total_chunks}</h3>
            <p>Total Chunks</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # File list
    for file_info in st.session_state.uploaded_files:
        status_info = st.session_state.processing_status.get(file_info["name"], {})
        status = status_info.get('status', 'pending')
        
        col1, col2, col3 = st.columns([3, 2, 1])
        
        with col1:
            st.write(f"**{file_info['name']}**")
            st.write(f"<span class='file-info'>Size: {file_info['size']:,} bytes | Type: {file_info.get('type', 'unknown')}</span>", unsafe_allow_html=True)
        
        with col2:
            if status == 'completed':
                st.markdown(f"<div class='status-box status-success'> {status_info.get('chunks', 0)} chunks</div>", unsafe_allow_html=True)
            elif status == 'error':
                st.markdown(f"<div class='status-box status-error'> Error</div>", unsafe_allow_html=True)
                st.caption(status_info.get('message', 'Unknown error'))
            else:
                st.markdown(f"<div class='status-box status-warning'>⏳ Pending</div>", unsafe_allow_html=True)
        
        with col3:
            if st.button("🗑️", key=f"remove_{file_info['name']}", help="Remove file"):
                # Remove from session state
                st.session_state.uploaded_files = [f for f in st.session_state.uploaded_files if f["name"] != file_info["name"]]
                if file_info["name"] in st.session_state.processing_status:
                    del st.session_state.processing_status[file_info["name"]]
                st.experimental_rerun()

def handle_query():
    """Handle user query and generate response."""
    if not st.session_state.current_query.strip() or not st.session_state.backend_healthy:
        return
    
    query = st.session_state.current_query.strip()
    
    # Add user message to chat history
    user_message = {"role": "user", "content": query, "timestamp": time.time()}
    st.session_state.chat_history.append(user_message)
    
    # Clear input
    st.session_state.current_query = ""
    
    with st.spinner("🤖 Processing your question..."):
        # Prepare conversation history (exclude current query)
        conversation_history = [
            {"role": msg["role"], "content": msg["content"]} 
            for msg in st.session_state.chat_history[:-1]
        ]
        
        # Query backend API
        response = query_backend(query, conversation_history)
        
        if "error" not in response:
            # Add assistant response to chat history
            assistant_message = {
                "role": "assistant",
                "content": response.get('response', 'No response generated'),
                "sources": response.get('sources', []),
                "confidence": response.get('confidence', 0.0),
                "query_type": response.get('query_type', 'unknown'),
                "retrieval_method": response.get('retrieval_method', 'unknown'),
                "chunks_used": response.get('chunks_used', 0),
                "timestamp": time.time()
            }
        else:
            # Add error message to chat history
            assistant_message = {
                "role": "assistant",
                "content": f"Sorry, I encountered an error: {response.get('error', 'Unknown error')}",
                "timestamp": time.time()
            }
        
        st.session_state.chat_history.append(assistant_message)

def display_chat_interface():
    """Display the chat interface."""
    st.subheader("💬 Chat with Your Documents")
    
    # Always allow chatting - check if backend is healthy
    if not st.session_state.backend_healthy:
        st.error(" Cannot chat: Backend API is not available")
        st.info("Please make sure the backend server is running on port 8000")
        return
    
    # Show warning if no documents are uploaded, but still allow chatting
    processed_files = [f for f in st.session_state.uploaded_files 
                      if st.session_state.processing_status.get(f["name"], {}).get('status') == 'completed']
    
    if not processed_files:
        st.warning("⚠️ No documents uploaded yet. The system will respond based on general knowledge.")
        st.info("For best results, upload documents in the 'Upload Documents' tab to build your knowledge base.")
    
    # Chat history display
    chat_container = st.container()
    with chat_container:
        if st.session_state.chat_history:
            for i, msg in enumerate(st.session_state.chat_history):
                if msg["role"] == "user":
                    message(msg["content"], is_user=True, key=f"user_{i}")
                else:
                    # Display assistant message
                    message(msg["content"], is_user=False, key=f"assistant_{i}")
                    
                    # Display metadata if available
                    if "confidence" in msg and "query_type" in msg:
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            confidence = msg["confidence"]
                            if confidence > 0.7:
                                st.write(f"🟢 Confidence: {confidence:.2f}")
                            elif confidence > 0.5:
                                st.write(f"🟡 Confidence: {confidence:.2f}")
                            else:
                                st.write(f"🔴 Confidence: {confidence:.2f}")
                        
                        with col2:
                            st.write(f"📊 Type: {msg['query_type']}")
                        
                        with col3:
                            st.write(f"🔍 Method: {msg['retrieval_method']}")
                        
                        with col4:
                            st.write(f"📚 Chunks: {msg['chunks_used']}")
                    
                    # Display sources if available
                    if "sources" in msg and msg["sources"]:
                        with st.expander(f"📚 Sources ({len(msg['sources'])})"):
                            for j, source in enumerate(msg["sources"]):
                                st.write(f"**{j+1}.** {source.get('source', 'Unknown')}")
                                if source.get('page'):
                                    st.write(f"   • Page: {source['page']}")
                                if source.get('section'):
                                    st.write(f"   • Section: {source['section']}")
    
    # Query input
    st.text_input(
        "Ask a question about your documents:",
        key="current_query",
        on_change=handle_query,
        placeholder="What would you like to know about your documents?",
        disabled=not st.session_state.backend_healthy
    )

def display_sidebar_controls():
    """Display sidebar controls."""
    st.sidebar.subheader("🎛️ Controls")
    
    if st.sidebar.button("🧹 Clear Chat"):
        st.session_state.chat_history = []
        st.experimental_rerun()
    
    if st.sidebar.button("🗑️ Clear All Files") and st.session_state.backend_healthy:
        if clear_system():
            st.session_state.uploaded_files = []
            st.session_state.processing_status = {}
            st.success("🧹 System cleared successfully!")
        else:
            st.error(" Failed to clear system")
        st.experimental_rerun()
    
    if st.sidebar.button("🔄 Refresh Status"):
        st.session_state.system_config = {}
        st.experimental_rerun()

def main():
    """Main Streamlit application."""
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">🤖 Agentic RAG System</h1>', unsafe_allow_html=True)
    st.markdown("**Upload documents and chat with your knowledge base using advanced AI agents**")
    
    # Sidebar
    display_system_status()
    display_sidebar_controls()
    
    # Main content
    if not st.session_state.backend_healthy:
        st.error("🔌 Backend API is not available")
        st.info("Please make sure the backend server is running:")
        st.code("cd backend && python api.py")
        st.stop()
    
    # Main tabs
    tab1, tab2 = st.tabs(["📤 Upload Documents", "💬 Chat"])
    
    with tab1:
        st.markdown('<h2 class="sub-header">Document Upload & Processing</h2>', unsafe_allow_html=True)
        
        display_file_upload()
        display_file_status()
        
        # Instructions
        with st.expander("📖 Instructions"):
            st.markdown("""
            **How to use the Agentic RAG System:**
            
            1. **Upload Documents**: Use the file uploader to add supported document types
            2. **Process Files**: Click "Process Files" to index your documents
            3. **Monitor Status**: Check processing status in the file list
            4. **Start Chatting**: Switch to the Chat tab to ask questions
            
            **Supported formats**: PDF, DOCX, PPTX, CSV, Excel, TXT, Markdown
            """)
    
    with tab2:
        display_chat_interface()

if __name__ == "__main__":
    main()