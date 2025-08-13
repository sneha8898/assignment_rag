# Fine-Tuned RAG Chatbot with Streaming Responses

This project implements a Retrieval-Augmented Generation (RAG) chatbot with real-time streaming responses, using a fine-tuned LLM and a vector database.

## Project Architecture

The project follows a modular architecture, with clear separation of concerns:

- **Streamlit App (`app.py`):** The main user interface for the chatbot, with real-time streaming and source display.
- **RAG Pipeline (`src/rag_pipeline.py`):** Orchestrates the retrieval and generation process, combining the vector store and the language model.
- **Document Processor (`src/document_processor.py`):** Handles loading, cleaning, and chunking of documents.
- **Vector Store (`src/vector_store.py`):** Manages vector embeddings and similarity search using FAISS.

### Data Flow

1.  **Document Ingestion:** Documents from the `/data` directory are loaded, cleaned, and split into sentence-aware chunks (100–300 words).
2.  **Embedding & Indexing:** Text chunks are embedded using a pre-trained model (e.g., `all-MiniLM-L6-v2`) and indexed in a FAISS vector database, which is saved to the `/vectordb` directory.
3.  **User Query:** The user enters a query in the Streamlit interface.
4.  **Retrieval:** The RAG pipeline performs a semantic search on the vector database to find the most relevant chunks.
5.  **Generation:** The retrieved chunks and the user query are injected into a high-quality prompt template and sent to the LLM (e.g., LLaMA 3.1) via the Groq API.
6.  **Streaming Response:** The model's response is streamed token-by-token back to the UI, providing real-time feedback.
7.  **Source Display:** The source chunks used to generate the answer are displayed alongside the response for transparency.

## Getting Started

### Prerequisites

- Python 3.8+
- Pip (Python package installer)

### Installation

1.  **Clone the repository:**

  

2.  **Create a virtual environment:**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up your Groq API key:**

    Create a `.env` file in the project root and add your Groq API key:

    ```
    GROQ_API_KEY=""
    ```

    Alternatively, you can set it as an environment variable.

### Running the Project

1.  **Add your documents:**

    Place your `.txt`, `.pdf`, or `.docx` files in the `/data` directory.

2.  **Process the documents and build the vector store:**

    Run the following script to preprocess the documents, create embeddings, and build the FAISS vector database:

    ```bash
    python -m src.vector_store
    ```

    This will create the vector store in the `/vectordb` directory.

3.  **Run the Streamlit app:**

    ```bash
    streamlit run app.py
    ```

    The application will open in your browser, ready to answer questions!

## Model and Embedding Choices

-   **Language Model:** `meta-llama/llama-3.1-8b-instruct` is used via the Groq API for its fast inference and strong instruction-following capabilities.
-   **Embedding Model:** `all-MiniLM-L6-v2` is used for its balance of performance and efficiency in generating semantic embeddings.

## Demo

<img width="1126" height="908" alt="image" src="https://github.com/user-attachments/assets/98e90852-3b48-4ba8-b06e-1828c8423b9f" />
<img width="1918" height="972" alt="image" src="https://github.com/user-attachments/assets/3f9b8ca8-a875-40ac-afa9-9c5229db3a7b" />
<img width="1913" height="972" alt="image" src="https://github.com/user-attachments/assets/c3eb6054-7a2b-405a-996c-11955b79f1d0" />




