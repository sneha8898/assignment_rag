import os
import re
import json
from typing import List, Dict, Any
from pathlib import Path
import nltk
import spacy
from langchain_text_splitters import RecursiveCharacterTextSplitter, SpacyTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from bs4 import BeautifulSoup

class DocumentProcessor:
    """
    Handles document loading, cleaning, and chunking for RAG pipeline
    """
    
    def __init__(self, chunk_size: int = 300, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunks = []
        self.metadata = []
        
        # Download required NLTK data
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
        except:
            pass
        
        # Try to load spaCy model
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Warning: spaCy model 'en_core_web_sm' not found. Using basic text splitter.")
            self.nlp = None
        
        # Initialize text splitter
        if self.nlp:
            self.text_splitter = SpacyTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                pipeline="en_core_web_sm"
            )
        else:
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
            )
    
    def load_documents(self, data_path: str) -> List[str]:
        """
        Load documents from a directory or file
        
        Args:
            data_path: Path to directory or file containing documents
            
        Returns:
            List of document contents
        """
        documents = []
        data_path = Path(data_path)
        
        if data_path.is_file():
            content = self._load_single_file(data_path)
            if content:
                documents.append(content)
        elif data_path.is_dir():
            for file_path in data_path.rglob("*"):
                if file_path.is_file() and not file_path.name.startswith('.'):
                    content = self._load_single_file(file_path)
                    if content:
                        documents.append(content)
        else:
            raise FileNotFoundError(f"Path not found: {data_path}")
        
        return documents
    
    def _load_single_file(self, file_path: Path) -> str:
        """
        Load content from a single file
        
        Args:
            file_path: Path to the file
            
        Returns:
            File content as string
        """
        try:
            extension = file_path.suffix.lower()
            
            if extension == '.txt':
                loader = TextLoader(str(file_path), encoding='utf-8')
                documents = loader.load()
                return documents[0].page_content if documents else ""
            
            elif extension == '.pdf':
                loader = PyPDFLoader(str(file_path))
                documents = loader.load()
                return "\n".join([doc.page_content for doc in documents])
            
            elif extension in ['.docx', '.doc']:
                loader = Docx2txtLoader(str(file_path))
                documents = loader.load()
                return documents[0].page_content if documents else ""
            
            elif extension == '.html':
                with open(file_path, 'r', encoding='utf-8') as f:
                    soup = BeautifulSoup(f.read(), 'html.parser')
                    return soup.get_text()
            
            else:
                # Try to read as plain text
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
                    
        except Exception as e:
            print(f"Error loading file {file_path}: {str(e)}")
            return ""
    
    def clean_text(self, text: str) -> str:
        """
        Clean and preprocess text
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove HTML tags if any
        text = BeautifulSoup(text, 'html.parser').get_text()
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.\!\?\,\;\:\-\(\)\[\]\{\}\"\'\/]', '', text)
        
        # Remove repeated punctuation
        text = re.sub(r'([.!?])\1+', r'\1', text)
        
        # Clean up spacing around punctuation
        text = re.sub(r'\s+([.!?,:;])', r'\1', text)
        text = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', text)
        
        return text.strip()
    
    def chunk_documents(self, documents: List[str]) -> List[Dict[str, Any]]:
        """
        Split documents into chunks
        
        Args:
            documents: List of document contents
            
        Returns:
            List of chunks with metadata
        """
        all_chunks = []
        
        for doc_idx, document in enumerate(documents):
            # Clean the document
            cleaned_doc = self.clean_text(document)
            
            if not cleaned_doc:
                continue
            
            # Split into chunks
            chunks = self.text_splitter.split_text(cleaned_doc)
            
            # Create chunk objects with metadata
            for chunk_idx, chunk in enumerate(chunks):
                if len(chunk.strip()) < 10:  # Skip very small chunks
                    continue
                
                chunk_data = {
                    'content': chunk.strip(),
                    'metadata': {
                        'document_id': doc_idx,
                        'chunk_id': chunk_idx,
                        'chunk_size': len(chunk),
                        'word_count': len(chunk.split()),
                    }
                }
                all_chunks.append(chunk_data)
        
        return all_chunks
    
    def process_documents(self, data_path: str, save_chunks: bool = True) -> List[Dict[str, Any]]:
        """
        Complete document processing pipeline
        
        Args:
            data_path: Path to documents
            save_chunks: Whether to save chunks to file
            
        Returns:
            List of processed chunks
        """
        print("Loading documents...")
        documents = self.load_documents(data_path)
        print(f"Loaded {len(documents)} documents")
        
        print("Chunking documents...")
        chunks = self.chunk_documents(documents)
        print(f"Created {len(chunks)} chunks")
        
        # Save chunks if requested
        if save_chunks:
            self.save_chunks(chunks)
        
        self.chunks = chunks
        return chunks
    
    def save_chunks(self, chunks: List[Dict[str, Any]], output_dir: str = "chunks"):
        """
        Save chunks to JSON file
        
        Args:
            chunks: List of chunks to save
            output_dir: Directory to save chunks
        """
        os.makedirs(output_dir, exist_ok=True)
        
        output_file = os.path.join(output_dir, "chunks_data.json")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {len(chunks)} chunks to {output_file}")
    
    def load_chunks(self, chunks_path: str = "chunks/chunks_data.json") -> List[Dict[str, Any]]:
        """
        Load previously saved chunks
        
        Args:
            chunks_path: Path to saved chunks file
            
        Returns:
            List of loaded chunks
        """
        if os.path.exists(chunks_path):
            with open(chunks_path, 'r', encoding='utf-8') as f:
                chunks = json.load(f)
            self.chunks = chunks
            print(f"Loaded {len(chunks)} chunks from {chunks_path}")
            return chunks
        else:
            print(f"Chunks file not found: {chunks_path}")
            return []
    
    def get_chunk_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the processed chunks
        
        Returns:
            Dictionary with chunk statistics
        """
        if not self.chunks:
            return {}
        
        chunk_sizes = [chunk['metadata']['chunk_size'] for chunk in self.chunks]
        word_counts = [chunk['metadata']['word_count'] for chunk in self.chunks]
        
        return {
            'total_chunks': len(self.chunks),
            'avg_chunk_size': sum(chunk_sizes) / len(chunk_sizes),
            'min_chunk_size': min(chunk_sizes),
            'max_chunk_size': max(chunk_sizes),
            'avg_word_count': sum(word_counts) / len(word_counts),
            'min_word_count': min(word_counts),
            'max_word_count': max(word_counts),
        }

if __name__ == "__main__":
    # Example usage
    processor = DocumentProcessor(chunk_size=250, chunk_overlap=50)
    
    # Process documents
    chunks = processor.process_documents("data/", save_chunks=True)
    
    # Print statistics
    stats = processor.get_chunk_statistics()
    print("\nChunk Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")
