import os
import json
from typing import List, Dict, Any, Generator
from groq import Groq
try:
    from src.document_processor import DocumentProcessor
    from src.vector_store import VectorStore
except ImportError:
    from document_processor import DocumentProcessor
    from vector_store import VectorStore
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class RAGPipeline:
    """
    Complete RAG pipeline that combines retrieval and generation
    """
    
    def __init__(self, 
                 groq_api_key: str = None,
                 model_name: str = "llama-3.1-8b-instant",
                 max_chunks: int = 5,
                 temperature: float = 0.1):
        """
        Initialize the RAG pipeline
        
        Args:
            groq_api_key: Groq API key
            model_name: LLM model name
            max_chunks: Maximum number of chunks to retrieve
            temperature: Generation temperature
        """
        self.model_name = model_name
        self.max_chunks = max_chunks
        self.temperature = temperature
        self.last_sources = []
        
        # Initialize Groq client
        api_key = groq_api_key or os.getenv("GROQ_API_KEY") or "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"
        if not api_key:
            raise ValueError("Groq API key is required. Set GROQ_API_KEY environment variable or pass it directly.")
        
        self.groq_client = Groq(api_key=api_key)
        
        # Initialize components
        self.document_processor = DocumentProcessor()
        self.vector_store = VectorStore()
        
        # Try to load existing vector store
        if not self.vector_store.load_vector_store():
            print("No existing vector store found. You'll need to process documents first.")
        
        # System prompt template
        self.system_prompt = """You are an expert AI assistant that provides detailed, accurate, and well-structured answers. You have access to relevant information that you should use to answer questions comprehensively.

Instructions for high-quality responses:
1. Provide COMPREHENSIVE answers that fully address the user's question
2. Structure your response with clear sections, bullet points, or numbered lists when appropriate
3. Include SPECIFIC details and examples when relevant
4. Address ALL relevant aspects of the topic
5. Use a professional, informative tone while being accessible and easy to understand
6. When discussing procedures, policies, or processes, explain them step-by-step
7. If you cannot fully answer the question with the available information, clearly state what information is missing
8. Connect related information naturally within your response
9. Conclude with actionable insights or next steps when appropriate
10. Write naturally and conversationally - do not reference "context documents" or numbered sources in your response
11. Integrate information seamlessly as if it's your own knowledge
12. Focus on providing value to the user with clear, direct answers

Relevant Information:
{context}

Please provide a detailed and natural answer to the following question:"""
        
        self.user_prompt = """Question: {question}

Answer: """
    
    def initialize_from_documents(self, data_path: str):
        """
        Initialize the RAG pipeline by processing documents
        
        Args:
            data_path: Path to documents directory
        """
        print("Initializing RAG pipeline from documents...")
        
        # Process documents
        chunks = self.document_processor.process_documents(data_path, save_chunks=True)
        
        if not chunks:
            raise ValueError("No chunks created from documents")
        
        # Build vector store
        self.vector_store.build_vector_store(chunks, save=True)
        
        print("RAG pipeline initialized successfully!")
        return True
    
    def retrieve_context(self, query: str) -> List[Dict[str, Any]]:
        """
        Retrieve relevant context chunks for the query
        
        Args:
            query: User query
            
        Returns:
            List of relevant chunks
        """
        try:
            results = self.vector_store.similarity_search(
                query=query,
                k=self.max_chunks,
                score_threshold=0.1
            )
            
            # Store for later reference
            self.last_sources = results
            return results
            
        except Exception as e:
            print(f"Error retrieving context: {str(e)}")
            return []
    
    def format_context(self, chunks: List[Dict[str, Any]]) -> str:
        """
        Format retrieved chunks into context string
        
        Args:
            chunks: List of retrieved chunks
            
        Returns:
            Formatted context string
        """
        if not chunks:
            return "No relevant information found."
        
        # Simply concatenate all chunk contents without any references
        context_parts = []
        for chunk in chunks:
            context_parts.append(chunk['content'].strip())
        
        return "\n\n".join(context_parts)
    
    def generate_response(self, query: str, context: str) -> str:
        """
        Generate response using the LLM
        
        Args:
            query: User query
            context: Retrieved context
            
        Returns:
            Generated response
        """
        try:
            # Format prompts
            system_message = self.system_prompt.format(context=context)
            user_message = self.user_prompt.format(question=query)
            
            # Generate response
            response = self.groq_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                temperature=self.temperature,
                max_tokens=2048,
                stream=False
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def get_streaming_response(self, query: str) -> Generator[str, None, None]:
        """
        Generate streaming response
        
        Args:
            query: User query
            
        Yields:
            Response chunks
        """
        try:
            # Retrieve context
            chunks = self.retrieve_context(query)
            context = self.format_context(chunks)
            
            # Format prompts
            system_message = self.system_prompt.format(context=context)
            user_message = self.user_prompt.format(question=query)
            
            # Generate streaming response
            stream = self.groq_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                temperature=self.temperature,
                max_tokens=2048,
                stream=True
            )
            
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            yield f"Error generating streaming response: {str(e)}"
    
    def get_response(self, query: str) -> Dict[str, Any]:
        """
        Get complete response with sources
        
        Args:
            query: User query
            
        Returns:
            Response dictionary with answer and sources
        """
        # Retrieve context
        chunks = self.retrieve_context(query)
        context = self.format_context(chunks)
        
        # Generate response
        answer = self.generate_response(query, context)
        
        return {
            'answer': answer,
            'sources': chunks,
            'context': context
        }
    
    def get_last_sources(self) -> List[Dict[str, Any]]:
        """
        Get sources from the last query
        
        Returns:
            List of source chunks
        """
        return self.last_sources
    
    def update_settings(self, max_chunks: int = None, temperature: float = None):
        """
        Update pipeline settings
        
        Args:
            max_chunks: Maximum number of chunks to retrieve
            temperature: Generation temperature
        """
        if max_chunks is not None:
            self.max_chunks = max_chunks
        
        if temperature is not None:
            self.temperature = temperature
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get pipeline statistics
        
        Returns:
            Statistics dictionary
        """
        vector_stats = self.vector_store.get_stats()
        
        stats = {
            'model_name': self.model_name,
            'max_chunks': self.max_chunks,
            'temperature': self.temperature,
            'total_documents': 1,  # Will be updated based on actual document count
            'total_chunks': vector_stats.get('total_chunks', 0),
            'embedding_model': vector_stats.get('embedding_model', 'unknown'),
            'vector_store_loaded': vector_stats.get('vector_store_loaded', False)
        }
        
        return stats
    
    def add_documents(self, data_path: str):
        """
        Add new documents to existing pipeline
        
        Args:
            data_path: Path to new documents
        """
        # Process new documents
        new_chunks = self.document_processor.process_documents(data_path, save_chunks=False)
        
        # Add to vector store
        self.vector_store.add_chunks(new_chunks)
        self.vector_store.save_vector_store()
        
        print(f"Added {len(new_chunks)} new chunks to the pipeline")

def test_rag_pipeline():
    """
    Test the RAG pipeline with sample queries
    """
    # Initialize pipeline
    pipeline = RAGPipeline()
    
    # Sample queries
    test_queries = [
        "What is the privacy policy about?",
        "What are the terms and conditions?",
        "How is user data protected?",
        "What are the refund policies?",
        "Who can access my personal information?"
    ]
    
    print("Testing RAG Pipeline:")
    print("=" * 50)
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 30)
        
        try:
            response = pipeline.get_response(query)
            print(f"Answer: {response['answer']}")
            print(f"Sources: {len(response['sources'])} chunks retrieved")
            
            for i, source in enumerate(response['sources'][:2], 1):  # Show first 2 sources
                print(f"  Source {i}: {source['content'][:100]}... (Score: {source['score']:.3f})")
        
        except Exception as e:
            print(f"Error: {str(e)}")
        
        print()

if __name__ == "__main__":
    test_rag_pipeline()
