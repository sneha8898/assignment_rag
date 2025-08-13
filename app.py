import streamlit as st
import os
import sys
import time
import tempfile
from typing import Generator, List, Dict
from pathlib import Path

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src.rag_pipeline import RAGPipeline
    from src.document_processor import DocumentProcessor
    from src.vector_store import VectorStore
except ImportError:
    # Fallback imports
    from rag_pipeline import RAGPipeline
    from document_processor import DocumentProcessor
    from vector_store import VectorStore

# Page configuration
st.set_page_config(
    page_title="RAG Chatbot with Streaming",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    
    .user-message {
        background-color: #e3f2fd;
        align-items: flex-end;
    }
    
    .bot-message {
        background-color: #f5f5f5;
        align-items: flex-start;
    }
    
    .source-chunk {
        background-color: #fff3cd;
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin: 0.5rem 0;
        border-left: 4px solid #ffc107;
        font-size: 0.9rem;
    }
    
    .streaming-cursor {
        animation: blink 1s infinite;
    }
    
    @keyframes blink {
        0%, 50% { opacity: 1; }
        51%, 100% { opacity: 0; }
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_rag_pipeline():
    """Initialize the RAG pipeline (cached to avoid reloading)"""
    try:
        pipeline = RAGPipeline()
        return pipeline
    except Exception as e:
        st.error(f"Failed to initialize RAG pipeline: {str(e)}")
        return None

def display_chat_message(role: str, content: str, sources: List[Dict] = None):
    """Display a chat message with proper formatting"""
    css_class = "user-message" if role == "user" else "bot-message"
    
    with st.container():
        st.markdown(f'<div class="chat-message {css_class}">', unsafe_allow_html=True)
        
        if role == "user":
            st.markdown("**You:**")
        else:
            st.markdown("**Assistant:**")
        
        st.markdown(content)
        
        # Display sources if available
        if sources:
            with st.expander("Source Documents", expanded=False):
                for i, source in enumerate(sources, 1):
                    st.markdown(f"""
                    <div class="source-chunk">
                        <strong>Source {i}:</strong><br>
                        {source.get('content', 'No content available')}
                        <br><small><i>Relevance Score: {source.get('score', 'N/A'):.3f}</i></small>
                    </div>
                    """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

def stream_response(response_generator: Generator) -> str:
    """Display streaming response with real-time updates"""
    response_container = st.empty()
    full_response = ""
    
    for chunk in response_generator:
        full_response += chunk
        # Update the container with current response + blinking cursor
        response_container.markdown(f"{full_response}<span class='streaming-cursor'>▊</span>", unsafe_allow_html=True)
        time.sleep(0.02)  # Small delay for visual effect
    
    # Final response without cursor
    response_container.markdown(full_response)
    return full_response

def save_uploaded_file(uploaded_file, data_dir="data"):
    """Save uploaded file to the data directory"""
    try:
        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
        
        # Save the uploaded file
        file_path = os.path.join(data_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        return file_path
    except Exception as e:
        st.error(f"Error saving file: {str(e)}")
        return None

def process_uploaded_documents(pipeline, file_paths):
    """Process uploaded documents and update the vector store"""
    try:
        with st.spinner("Processing uploaded documents..."):
            # Create a temporary directory for processing
            with tempfile.TemporaryDirectory() as temp_dir:
                # Copy files to temp directory for processing
                temp_files = []
                for file_path in file_paths:
                    temp_file = os.path.join(temp_dir, os.path.basename(file_path))
                    with open(file_path, 'rb') as src, open(temp_file, 'wb') as dst:
                        dst.write(src.read())
                    temp_files.append(temp_file)
                
                # Process documents
                processor = DocumentProcessor(chunk_size=300, chunk_overlap=50)
                chunks = processor.process_documents(temp_dir, save_chunks=False)
                
                if chunks:
                    # Add to existing vector store
                    pipeline.vector_store.add_chunks(chunks)
                    pipeline.vector_store.save_vector_store()
                    
                    return len(chunks)
                else:
                    return 0
                    
    except Exception as e:
        st.error(f"Error processing documents: {str(e)}")
        return 0

def delete_database():
    """Delete all vector database files and reset the system"""
    import shutil
    
    try:
        # List of paths to delete
        paths_to_delete = [
            "vectordb",
            "chunks", 
            "data"  # Optional: also delete uploaded documents
        ]
        
        deleted_items = []
        for path in paths_to_delete:
            if os.path.exists(path):
                if os.path.isdir(path):
                    shutil.rmtree(path)
                    deleted_items.append(f"Directory: {path}")
                else:
                    os.remove(path)
                    deleted_items.append(f"File: {path}")
        
        return True, deleted_items
        
    except Exception as e:
        return False, str(e)

def display_database_management():
    """Display database management section"""
    st.header("Database Management")
    
    # Get current database stats
    pipeline = st.session_state.get('pipeline')
    if pipeline:
        stats = pipeline.get_stats()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Current Database Stats")
            st.metric("Total Documents", stats.get('total_documents', 0))
            st.metric("Total Chunks", stats.get('total_chunks', 0))
            st.metric("Embedding Model", stats.get('embedding_model', 'Unknown'))
            
            # Show vector store status
            if stats.get('vector_store_loaded', False):
                st.success("Vector database is active")
            else:
                st.warning("Vector database not loaded")
        
        with col2:
            st.subheader("Danger Zone")
            st.warning(
                "**Warning:** This action will permanently delete:\n\n"
                "• All vector embeddings\n"
                "• All processed document chunks\n"
                "• All uploaded documents\n"
                "• Chat history\n\n"
                "This action cannot be undone!"
            )
            
            # Confirmation checkbox
            confirm_delete = st.checkbox(
                "I understand this will permanently delete all data",
                key="confirm_delete_checkbox"
            )
            
            # Delete button (only enabled if confirmed)
            if st.button(
                "Delete Entire Database",
                type="secondary",
                disabled=not confirm_delete,
                help="This will delete all documents, embeddings, and reset the system"
            ):
                if confirm_delete:
                    with st.spinner("Deleting database..."):
                        success, result = delete_database()
                        
                        if success:
                            st.success("Database deleted successfully!")
                            st.info("Deleted items:")
                            for item in result:
                                st.write(f"• {item}")
                            
                            # Clear session state
                            if 'pipeline' in st.session_state:
                                del st.session_state.pipeline
                            if 'messages' in st.session_state:
                                st.session_state.messages = []
                            
                            # Clear cache and reinitialize
                            st.cache_resource.clear()
                            st.info("Please refresh the page to reinitialize the system.")
                            
                            # Auto-refresh after a delay
                            time.sleep(2)
                            st.rerun()
                        else:
                            st.error(f"Error deleting database: {result}")
    else:
        st.warning("RAG pipeline not initialized. Nothing to delete.")

def display_upload_section():
    """Display document upload section"""
    st.header("Upload Documents")
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Choose documents to upload",
        type=['txt', 'pdf', 'docx', 'doc'],
        accept_multiple_files=True,
        help="Supported formats: TXT, PDF, DOCX, DOC"
    )
    
    if uploaded_files:
        st.write(f"{len(uploaded_files)} file(s) selected:")
        for file in uploaded_files:
            st.write(f"• {file.name} ({file.size} bytes)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Process Documents", type="primary"):
                # Save uploaded files
                file_paths = []
                for uploaded_file in uploaded_files:
                    file_path = save_uploaded_file(uploaded_file)
                    if file_path:
                        file_paths.append(file_path)
                
                if file_paths:
                    # Get pipeline from session state or initialize
                    pipeline = st.session_state.get('pipeline')
                    if pipeline:
                        # Process documents
                        num_chunks = process_uploaded_documents(pipeline, file_paths)
                        
                        if num_chunks > 0:
                            st.success(f"Successfully processed {len(file_paths)} document(s) and created {num_chunks} chunks!")
                            st.info("You can now ask questions about the uploaded documents.")
                            
                            # Clear the cache to refresh stats
                            st.cache_resource.clear()
                            st.rerun()
                        else:
                            st.error("No content could be extracted from the uploaded documents.")
                    else:
                        st.error("RAG pipeline not initialized. Please refresh the page.")
        
        with col2:
            if st.button("Clear Selection"):
                st.rerun()

def main():
    # Title and description
    st.title("RAG Chatbot with Streaming")
    st.markdown("Ask questions about the documents and get AI-powered answers with source references!")
    
    # Initialize pipeline and store in session state
    if 'pipeline' not in st.session_state:
        st.session_state.pipeline = initialize_rag_pipeline()
    
    pipeline = st.session_state.pipeline
    
    # Create tabs for different sections
    tab1, tab2, tab3 = st.tabs(["Chat", "Upload Documents", "Database Management"])
    
    with tab2:
        display_upload_section()
    
    with tab3:
        display_database_management()
    
    with tab1:
        # Sidebar
        with st.sidebar:
            st.header("Configuration")
            
            if pipeline:
                st.success("RAG Pipeline Loaded")
                
                # Display stats
                st.subheader("System Stats")
                stats = pipeline.get_stats()
                st.metric("Documents Indexed", stats.get('total_documents', 0))
                st.metric("Total Chunks", stats.get('total_chunks', 0))
                st.metric("Current Model", stats.get('model_name', 'Unknown'))
                
                # Show optimized settings
                st.subheader("Optimized Configuration")
                st.info("The system is pre-configured with optimal settings for best answer quality:")
                st.write(f"• **Max Retrieved Chunks**: {stats.get('max_chunks', 5)} (Comprehensive context)")
                st.write(f"• **Temperature**: {stats.get('temperature', 0.1)} (Precise, factual responses)")
                st.write(f"• **Embedding Model**: {stats.get('embedding_model', 'all-MiniLM-L6-v2')}")
                st.success("Settings automatically optimized for highest answer quality!")
            else:
                st.error("Failed to load RAG Pipeline")
                st.stop()
            
            # Clear chat button
            if st.button("Clear Chat"):
                st.session_state.messages = []
                st.rerun()
        
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Display chat history
        for message in st.session_state.messages:
            display_chat_message(
                message["role"], 
                message["content"], 
                message.get("sources")
            )
        
        # Chat input
        if prompt := st.chat_input("Ask your question here..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            display_chat_message("user", prompt)
            
            # Generate response
            with st.spinner("Thinking..."):
                try:
                    # Get streaming response
                    response_stream = pipeline.get_streaming_response(prompt)
                    
                    # Display streaming response
                    st.markdown("**Assistant:**")
                    full_response = stream_response(response_stream)
                    
                    # Get sources for the last query
                    sources = pipeline.get_last_sources()
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": full_response,
                        "sources": sources
                    })
                    
                    # Display sources
                    if sources:
                        with st.expander("Source Documents", expanded=False):
                            for i, source in enumerate(sources, 1):
                                st.markdown(f"""
                                <div class="source-chunk">
                                    <strong>Source {i}:</strong><br>
                                    {source.get('content', 'No content available')}
                                    <br><small><i>Relevance Score: {source.get('score', 'N/A'):.3f}</i></small>
                                </div>
                                """, unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    st.error("Please check your configuration and try again.")

    # Footer
    st.markdown("---")

    
if __name__ == "__main__":
    main()
