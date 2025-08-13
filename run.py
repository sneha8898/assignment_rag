#!/usr/bin/env python3
"""
Simple script to run the RAG chatbot project
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    print("🤖 RAG Chatbot with Streaming")
    print("=" * 40)
    
    # Check if we're in the right directory
    if not os.path.exists("app.py"):
        print("❌ Error: Please run this script from the project root directory")
        sys.exit(1)
    
    # Check if data directory exists and has files
    data_dir = Path("data")
    if not data_dir.exists():
        print("📁 Creating data directory...")
        data_dir.mkdir()
    
    # Check for documents
    document_files = list(data_dir.glob("*.txt")) + list(data_dir.glob("*.pdf")) + list(data_dir.glob("*.docx"))
    
    if not document_files:
        print("📄 No documents found in data/ directory")
        print("✅ Sample document already created: data/sample_document.txt")
        print("💡 You can add more documents to the data/ directory")
    else:
        print(f"📄 Found {len(document_files)} document(s) in data/ directory")
    
    # Check if vector store exists
    vectordb_dir = Path("vectordb")
    if not vectordb_dir.exists() or not list(vectordb_dir.glob("*")):
        print("\n🔧 Vector store not found. Processing documents...")
        
        try:
            # Import and run document processing
            sys.path.append('src')
            from document_processor import DocumentProcessor
            from vector_store import VectorStore
            
            # Process documents
            processor = DocumentProcessor(chunk_size=300, chunk_overlap=50)
            chunks = processor.process_documents("data/", save_chunks=True)
            
            if chunks:
                print(f"✅ Created {len(chunks)} chunks")
                
                # Build vector store
                vector_store = VectorStore()
                vector_store.build_vector_store(chunks, save=True)
                print("✅ Vector store created successfully")
            else:
                print("❌ No chunks created. Please check your documents.")
                return
                
        except Exception as e:
            print(f"❌ Error processing documents: {str(e)}")
            print("💡 Try running: python setup.py --data-path data/")
            return
    else:
        print("✅ Vector store found")
    
    # Start Streamlit app
    print("\n🚀 Starting Streamlit app...")
    print("💡 The app will open in your browser automatically")
    print("🔗 If it doesn't open, go to: http://localhost:8501")
    print("\n📝 Sample questions to try:")
    print("   • What is the privacy policy about?")
    print("   • How is user data protected?")
    print("   • What are the refund policies?")
    print("   • What are the prohibited uses?")
    print("\n⏹️  Press Ctrl+C to stop the server")
    print("=" * 40)
    
    try:
        # Run Streamlit
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"], check=True)
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error starting Streamlit: {e}")
        print("💡 Make sure Streamlit is installed: pip install streamlit")

if __name__ == "__main__":
    main()
