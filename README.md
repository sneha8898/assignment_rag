# RAG Chatbot Project

A modern Retrieval-Augmented Generation (RAG) chatbot built with Python, featuring real-time streaming responses, intelligent document processing, and advanced vector search capabilities.

## 🚀 Features

### Core RAG Pipeline
- **Advanced Document Processing**: Multi-format support (PDF, TXT, DOCX) with intelligent text extraction
- **Semantic Embeddings**: High-quality vector embeddings using Sentence Transformers
- **Vector Search**: Fast similarity search with FAISS vector database
- **Smart Generation**: Context-aware responses using Groq's Llama models

### Advanced Capabilities
- **Streaming Interface**: Real-time response generation with visual streaming effects
- **Source Attribution**: Detailed references with relevance scores for transparency
- **Dynamic Upload**: Add documents through the web interface
- **Database Management**: Complete vector database control and reset capabilities
- **Optimized Configuration**: Pre-tuned for maximum answer quality

## 📁 Project Structure

```
rag-chatbot-project/
├── /data/                       # Document storage directory
│   └── AI Training Document.pdf # Provided document files
├── /chunks/                     # Processed and embedded text segments
│   └── chunks_data.json        # Processed document chunks
├── /vectordb/                   # Saved vector database
│   ├── faiss_index/           # FAISS vector index files
│   └── metadata.json          # Database metadata
├── /notebooks/                  # Preprocessing, tuning, and evaluation
│   └── rag_testing_evaluation.ipynb # Testing and evaluation notebook
├── /src/                        # Retriever, generator, and pipeline scripts
│   ├── document_processor.py   # Document loading and chunking
│   ├── vector_store.py         # Vector embeddings and similarity search
│   ├── rag_pipeline.py         # Main RAG pipeline orchestration
│   └── __init__.py             # Package initialization
├── app.py                       # Streamlit app with streaming support
├── requirements.txt             # Python dependencies
├── README.md                    # Project documentation
└── .env                         # Environment variables (API keys)
```

## 🛠️ Technology Stack

### Core Technologies
- **Python 3.8+**: Main programming language
- **Streamlit**: Modern web application framework with streaming support
- **LangChain**: Document processing and RAG pipeline components
- **FAISS**: High-performance vector similarity search
- **Sentence Transformers**: State-of-the-art text embedding models
- **Groq**: Fast LLM inference platform

### Key Dependencies
```
streamlit>=1.28.0              # Web application framework
langchain>=0.0.350             # RAG pipeline components
langchain-community>=0.0.10    # Community extensions
sentence-transformers>=2.2.2   # Text embeddings
faiss-cpu>=1.7.4               # Vector search engine
groq>=0.4.1                    # LLM API client
numpy>=1.24.0                  # Numerical computing
pandas>=2.0.0                  # Data manipulation
python-dotenv>=1.0.0           # Environment management
```

## ⚡ Quick Start

### Prerequisites
- Python 3.8 or higher
- Groq API key ([Get one free](https://console.groq.com/))

### Installation

1. **Clone and Setup Environment**
   ```bash
   git clone <repository-url>
   cd rag-chatbot-project
   
   # Create virtual environment
   python -m venv venv
   
   # Activate environment
   # Windows:
   venv\Scripts\activate
   # macOS/Linux:
   source venv/bin/activate
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment**
   Create `.env` file:
   ```
   GROQ_API_KEY=your_groq_api_key_here
   ```

4. **Launch Application**
   ```bash
   streamlit run app.py
   ```

   Access at: `http://localhost:8501`

## 💡 Usage Guide

### Web Interface

#### 🔥 Chat Tab
- **Natural Conversations**: Ask questions in plain English
- **Real-time Streaming**: Watch responses generate in real-time
- **Source References**: Click to expand and view relevant document chunks
- **Conversation History**: Persistent chat history throughout session

#### 📄 Upload Documents Tab
- **Drag & Drop**: Easy file upload interface
- **Multi-format Support**: PDF, TXT, DOCX, DOC files
- **Automatic Processing**: Documents are chunked and indexed automatically
- **Real-time Feedback**: Progress indicators and success notifications

#### ⚙️ Database Management Tab
- **System Statistics**: View document count, chunk count, and model info
- **Database Status**: Real-time vector database status monitoring
- **Reset Functionality**: Complete system reset with confirmation
- **Safe Operations**: Built-in safeguards and confirmations

### Advanced Features

#### Optimized Configuration
Pre-configured for maximum answer quality:
- **Max Retrieved Chunks**: 5 (comprehensive context)
- **Temperature**: 0.1 (precise, factual responses)
- **Embedding Model**: all-MiniLM-L6-v2 (optimal speed/quality balance)
- **Chunk Size**: 300 characters with 50 character overlap

#### Smart Document Processing
- **Multi-format Support**: PDF, TXT, DOCX, DOC
- **Text Cleaning**: HTML removal, whitespace normalization
- **Intelligent Chunking**: Context-preserving text segmentation
- **Metadata Tracking**: Document and chunk-level metadata

## 🔧 Configuration

### Environment Variables
```bash
# Required
GROQ_API_KEY=your_groq_api_key

# Optional
CHUNK_SIZE=300
CHUNK_OVERLAP=50
MAX_CHUNKS=5
TEMPERATURE=0.1
```

### Customization Options

#### Document Processing
```python
# Modify in src/document_processor.py
processor = DocumentProcessor(
    chunk_size=300,        # Adjust chunk size
    chunk_overlap=50       # Adjust overlap
)
```

#### Vector Search
```python
# Modify in src/vector_store.py
vector_store = VectorStore(
    embedding_model="all-MiniLM-L6-v2",  # Change embedding model
    vector_db_path="vectordb"             # Change storage path
)
```

## 📊 Performance & Quality

### Optimizations
- **Fast Embedding Model**: all-MiniLM-L6-v2 for optimal speed/quality
- **FAISS Vector Search**: Millisecond-level similarity search
- **Streaming Responses**: Real-time user experience
- **Smart Chunking**: Context-preserving text segmentation

### Quality Features
- **Source Attribution**: Every answer includes source references
- **Relevance Scoring**: Confidence scores for retrieved chunks
- **Context Optimization**: Multiple chunks for comprehensive answers
- **Temperature Control**: Balanced creativity vs. accuracy

## 🧪 Development & Testing

### Testing Notebook
Use the included Jupyter notebook for evaluation:
```bash
jupyter notebook notebooks/rag_testing_evaluation.ipynb
```

### API Usage
```python
# Direct pipeline usage
from src.rag_pipeline import RAGPipeline

pipeline = RAGPipeline()

# Get streaming response
for chunk in pipeline.get_streaming_response("Your question"):
    print(chunk, end="")

# Get complete response with sources
response = pipeline.get_response("Your question")
print(response['answer'])
print(f"Sources: {len(response['sources'])}")
```

### Extending the System

1. **Custom Document Loaders**: Extend `DocumentProcessor` for new formats
2. **Different Embeddings**: Modify embedding model in `VectorStore`
3. **Alternative LLMs**: Replace Groq client in `RAGPipeline`
4. **Enhanced UI**: Customize Streamlit interface in `app.py`

## 🔍 Troubleshooting

### Common Issues

**"Groq API key is required"**
- Verify `.env` file exists and contains `GROQ_API_KEY`
- Check API key validity at [Groq Console](https://console.groq.com/)

**"Vector store files not found"**
- Ensure documents exist in `data/` folder
- Run application once to build initial vector database

**"No chunks created"**
- Verify document formats (PDF, TXT, DOCX, DOC)
- Check documents contain readable text content

**Import/Dependency Errors**
- Ensure virtual environment is activated
- Run `pip install -r requirements.txt`
- For Windows: May need Visual C++ Build Tools for some packages

### Performance Tuning

- **Large Documents**: Increase `chunk_size` for better context
- **Faster Responses**: Reduce `max_chunks` retrieved
- **Memory Optimization**: Use smaller embedding models
- **Quality vs Speed**: Adjust temperature and chunk parameters

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📧 Support

For support and questions:
- 📖 Check this README and troubleshooting section
- 🔍 Review the notebook examples
- 🐛 Open an issue for bugs
- 💡 Suggest features via issues

---

**Built with ❤️ using modern Python, Streamlit, and cutting-edge NLP technologies**
