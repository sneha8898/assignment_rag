import os
import pickle
import numpy as np
from typing import List, Dict, Any, Tuple
import faiss
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
import json

class VectorStore:
    """
    Handles vector embeddings and similarity search using FAISS
    """
    
    def __init__(self, 
                 embedding_model: str = "all-MiniLM-L6-v2",
                 vector_db_path: str = "vectordb"):
        """
        Initialize the vector store
        
        Args:
            embedding_model: Name of the sentence transformer model
            vector_db_path: Path to save/load vector database
        """
        self.embedding_model_name = embedding_model
        self.vector_db_path = vector_db_path
        self.chunks_data = []
        
        # Initialize embedding model
        print(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Initialize LangChain embeddings wrapper
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Vector store will be initialized later
        self.vector_store = None
        self.index = None
        
        # Create directory if it doesn't exist
        os.makedirs(vector_db_path, exist_ok=True)
    
    def create_embeddings(self, chunks: List[Dict[str, Any]]) -> np.ndarray:
        """
        Create embeddings for document chunks
        
        Args:
            chunks: List of document chunks with content and metadata
            
        Returns:
            NumPy array of embeddings
        """
        print(f"Creating embeddings for {len(chunks)} chunks...")
        
        # Extract text content
        texts = [chunk['content'] for chunk in chunks]
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        print(f"Created embeddings with shape: {embeddings.shape}")
        return embeddings
    
    def build_vector_store(self, chunks: List[Dict[str, Any]], save: bool = True):
        """
        Build FAISS vector store from chunks
        
        Args:
            chunks: List of document chunks
            save: Whether to save the vector store
        """
        self.chunks_data = chunks
        
        # Create documents for LangChain
        documents = []
        for chunk in chunks:
            doc = Document(
                page_content=chunk['content'],
                metadata=chunk['metadata']
            )
            documents.append(doc)
        
        # Build FAISS vector store
        print("Building FAISS vector store...")
        self.vector_store = FAISS.from_documents(
            documents=documents,
            embedding=self.embeddings
        )
        
        if save:
            self.save_vector_store()
    
    def save_vector_store(self):
        """
        Save the vector store to disk
        """
        if self.vector_store is None:
            raise ValueError("Vector store not built yet. Call build_vector_store() first.")
        
        # Save FAISS vector store
        vector_store_path = os.path.join(self.vector_db_path, "faiss_index")
        self.vector_store.save_local(vector_store_path)
        
        # Save chunks data separately
        chunks_path = os.path.join(self.vector_db_path, "chunks_data.json")
        with open(chunks_path, 'w', encoding='utf-8') as f:
            json.dump(self.chunks_data, f, indent=2, ensure_ascii=False)
        
        # Save metadata
        metadata_path = os.path.join(self.vector_db_path, "metadata.json")
        metadata = {
            'embedding_model': self.embedding_model_name,
            'total_chunks': len(self.chunks_data),
            'embedding_dim': self.vector_store.index.d if hasattr(self.vector_store, 'index') else None
        }
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Vector store saved to {self.vector_db_path}")
    
    def load_vector_store(self) -> bool:
        """
        Load vector store from disk
        
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            vector_store_path = os.path.join(self.vector_db_path, "faiss_index")
            chunks_path = os.path.join(self.vector_db_path, "chunks_data.json")
            
            if not os.path.exists(vector_store_path) or not os.path.exists(chunks_path):
                print("Vector store files not found")
                return False
            
            # Load FAISS vector store
            self.vector_store = FAISS.load_local(
                vector_store_path, 
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            
            # Load chunks data
            with open(chunks_path, 'r', encoding='utf-8') as f:
                self.chunks_data = json.load(f)
            
            print(f"Vector store loaded with {len(self.chunks_data)} chunks")
            return True
            
        except Exception as e:
            print(f"Error loading vector store: {str(e)}")
            return False
    
    def similarity_search(self, 
                         query: str, 
                         k: int = 5,
                         score_threshold: float = 0.0) -> List[Dict[str, Any]]:
        """
        Perform similarity search
        
        Args:
            query: Search query
            k: Number of results to return
            score_threshold: Minimum similarity score threshold
            
        Returns:
            List of relevant chunks with scores
        """
        if self.vector_store is None:
            raise ValueError("Vector store not available. Build or load vector store first.")
        
        # Perform similarity search with scores
        results = self.vector_store.similarity_search_with_score(query, k=k)
        
        # Format results
        formatted_results = []
        for doc, score in results:
            if score >= score_threshold:
                result = {
                    'content': doc.page_content,
                    'metadata': doc.metadata,
                    'score': float(score)
                }
                formatted_results.append(result)
        
        return formatted_results
    
    def semantic_search(self, 
                       query: str, 
                       k: int = 5) -> List[Dict[str, Any]]:
        """
        Alternative semantic search method using direct embedding comparison
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of relevant chunks with scores
        """
        if not self.chunks_data:
            raise ValueError("No chunks data available")
        
        # Get query embedding
        query_embedding = self.embedding_model.encode([query], normalize_embeddings=True)
        
        # Get all chunk embeddings
        chunk_texts = [chunk['content'] for chunk in self.chunks_data]
        chunk_embeddings = self.embedding_model.encode(chunk_texts, normalize_embeddings=True)
        
        # Calculate similarities
        similarities = np.dot(query_embedding, chunk_embeddings.T)[0]
        
        # Get top k results
        top_indices = np.argsort(similarities)[::-1][:k]
        
        results = []
        for idx in top_indices:
            result = {
                'content': self.chunks_data[idx]['content'],
                'metadata': self.chunks_data[idx]['metadata'],
                'score': float(similarities[idx])
            }
            results.append(result)
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get vector store statistics
        
        Returns:
            Dictionary with statistics
        """
        stats = {
            'embedding_model': self.embedding_model_name,
            'total_chunks': len(self.chunks_data),
            'vector_store_loaded': self.vector_store is not None
        }
        
        if self.vector_store and hasattr(self.vector_store, 'index'):
            stats['embedding_dimension'] = self.vector_store.index.d
            stats['total_vectors'] = self.vector_store.index.ntotal
        
        return stats
    
    def add_chunks(self, new_chunks: List[Dict[str, Any]]):
        """
        Add new chunks to existing vector store
        
        Args:
            new_chunks: List of new chunks to add
        """
        if not new_chunks:
            return
        
        # Create documents for new chunks
        new_documents = []
        for chunk in new_chunks:
            doc = Document(
                page_content=chunk['content'],
                metadata=chunk['metadata']
            )
            new_documents.append(doc)
        
        if self.vector_store is None:
            # Create new vector store
            self.chunks_data = new_chunks
            self.build_vector_store(new_chunks)
        else:
            # Add to existing vector store
            self.vector_store.add_documents(new_documents)
            self.chunks_data.extend(new_chunks)
        
        print(f"Added {len(new_chunks)} new chunks to vector store")

if __name__ == "__main__":
    # Example usage
    from document_processor import DocumentProcessor
    
    # Process documents
    processor = DocumentProcessor()
    chunks = processor.process_documents("data/")
    
    # Build vector store
    vector_store = VectorStore()
    vector_store.build_vector_store(chunks)
    
    # Test search
    results = vector_store.similarity_search("privacy policy", k=3)
    
    print("\nSearch Results:")
    for i, result in enumerate(results, 1):
        print(f"\nResult {i}:")
        print(f"Score: {result['score']:.3f}")
        print(f"Content: {result['content'][:200]}...")
        print(f"Metadata: {result['metadata']}")
    
    # Print stats
    stats = vector_store.get_stats()
    print(f"\nVector Store Stats: {stats}")
