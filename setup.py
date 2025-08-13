#!/usr/bin/env python3
"""
Setup script for the RAG Chatbot project
This script helps initialize the project and process documents
"""

import os
import sys
import argparse
from pathlib import Path

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.document_processor import DocumentProcessor
from src.vector_store import VectorStore
from src.rag_pipeline import RAGPipeline

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        'streamlit', 'langchain', 'groq', 'faiss-cpu', 
        'sentence-transformers', 'nltk', 'spacy'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("Missing required packages:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\nPlease install them using: pip install -r requirements.txt")
        return False
    
    return True

def download_models():
    """Download required models"""
    print("Downloading required models...")
    
    try:
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        print("✓ NLTK models downloaded")
    except Exception as e:
        print(f"✗ Error downloading NLTK models: {e}")
    
    try:
        import spacy
        # Try to load the model, download if not available
        try:
            spacy.load("en_core_web_sm")
            print("✓ spaCy model already available")
        except OSError:
            print("Downloading spaCy model... (this may take a while)")
            os.system("python -m spacy download en_core_web_sm")
            print("✓ spaCy model downloaded")
    except Exception as e:
        print(f"⚠ Warning: spaCy model download failed: {e}")
        print("  You can still use the basic text splitter")

def setup_directories():
    """Create necessary directories"""
    directories = ['data', 'chunks', 'vectordb', 'notebooks', 'src']
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✓ Created directory: {directory}")

def process_documents(data_path="data/"):
    """Process documents and build vector store"""
    print(f"\nProcessing documents from {data_path}...")
    
    # Check if data directory has files
    data_dir = Path(data_path)
    if not data_dir.exists():
        print(f"✗ Data directory not found: {data_path}")
        print("Please add your documents to the data directory first.")
        return False
    
    files = list(data_dir.glob("*"))
    document_files = [f for f in files if f.is_file() and f.suffix.lower() in ['.txt', '.pdf', '.docx', '.doc']]
    
    if not document_files:
        print(f"✗ No supported document files found in {data_path}")
        print("Supported formats: .txt, .pdf, .docx, .doc")
        return False
    
    print(f"Found {len(document_files)} document(s):")
    for file in document_files:
        print(f"  - {file.name}")
    
    try:
        # Initialize document processor
        processor = DocumentProcessor(chunk_size=300, chunk_overlap=50)
        
        # Process documents
        chunks = processor.process_documents(data_path, save_chunks=True)
        
        if not chunks:
            print("✗ No chunks created from documents")
            return False
        
        print(f"✓ Created {len(chunks)} chunks")
        
        # Build vector store
        print("Building vector store...")
        vector_store = VectorStore()
        vector_store.build_vector_store(chunks, save=True)
        
        print("✓ Vector store built and saved")
        
        # Display statistics
        stats = processor.get_chunk_statistics()
        print("\nDocument Processing Statistics:")
        print("=" * 40)
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"{key}: {value:.2f}")
            else:
                print(f"{key}: {value}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error processing documents: {e}")
        return False

def test_rag_pipeline():
    """Test the RAG pipeline"""
    print("\nTesting RAG pipeline...")
    
    try:
        pipeline = RAGPipeline()
        
        # Test with a simple query
        test_query = "What is this document about?"
        print(f"Test query: {test_query}")
        
        response = pipeline.get_response(test_query)
        print(f"Response: {response['answer'][:200]}...")
        print(f"Sources: {len(response['sources'])} chunks retrieved")
        
        print("✓ RAG pipeline test successful")
        return True
        
    except Exception as e:
        print(f"✗ RAG pipeline test failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Setup RAG Chatbot project")
    parser.add_argument("--data-path", default="data/", help="Path to documents directory")
    parser.add_argument("--skip-models", action="store_true", help="Skip model downloads")
    parser.add_argument("--skip-processing", action="store_true", help="Skip document processing")
    parser.add_argument("--test-only", action="store_true", help="Only run tests")
    
    args = parser.parse_args()
    
    print("RAG Chatbot Project Setup")
    print("=" * 40)
    
    if args.test_only:
        test_rag_pipeline()
        return
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    print("✓ All dependencies available")
    
    # Setup directories
    setup_directories()
    
    # Download models
    if not args.skip_models:
        download_models()
    
    # Process documents
    if not args.skip_processing:
        success = process_documents(args.data_path)
        if success:
            # Test the pipeline
            test_rag_pipeline()
    
    print("\n" + "=" * 40)
    print("Setup complete!")
    print("\nNext steps:")
    print("1. Add your documents to the 'data/' directory")
    print("2. Run: python setup.py --data-path data/")
    print("3. Run: streamlit run app.py")
    print("\nFor more information, see README.md")

if __name__ == "__main__":
    main()
