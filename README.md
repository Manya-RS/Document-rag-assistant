
# 📄 Document RAG Assistant

An AI-powered document question-answering assistant built using **Retrieval-Augmented Generation (RAG)**.

The application allows users to upload **PDF, TXT, and Markdown documents**, process and index their content, and ask questions that are answered using relevant information retrieved from the selected document.

## ✨ Features

- 📄 Supports PDF, TXT, and Markdown documents
- 🔍 Semantic search using Gemini embeddings
- 🗂️ Document-scoped FAISS vector indexes
- 🧩 Text chunking with overlap
- 🎯 Retrieval followed by lightweight lexical reranking
- 🤖 Grounded answer generation using Gemini
- 📚 Retrieved source chunks are returned with answers
- 🔄 LangGraph-based RAG workflow
- ⚡ Streaming answer endpoint
- ✅ Pydantic request/response validation
- 🚀 FastAPI backend
- 🖥️ Streamlit frontend
- 🐳 Docker and Docker Compose support
- 🔐 API key loaded through environment variables

## 🏗️ Architecture

```text
User
 │
 ▼
Streamlit Frontend
 │
 ▼
FastAPI API
 │
 ├── Document Upload
 │      │
 │      ├── PDF/TXT/MD Loader
 │      ├── Text Chunking
 │      ├── Gemini Embeddings
 │      └── FAISS Index
 │
 └── Question Answering
        │
        ▼
     LangGraph
        │
        ├── Retrieve relevant chunks
        ├── Rerank retrieved chunks
        └── Generate grounded answer
                │
                ▼
          Answer + Sources

📁 Project Structure
ShadowFox-Task3/
│
├── app/
│   ├── api/
│   │   ├── query.py
│   │   └── upload.py
│   │
│   ├── generation/
│   │   └── generator.py
│   │
│   ├── ingestion/
│   │   ├── document_loader.py
│   │   └── chunker.py
│   │
│   ├── models/
│   │   └── schemas.py
│   │
│   ├── retrieval/
│   │   ├── embeddings.py
│   │   ├── indexer.py
│   │   ├── retriever.py
│   │   └── vector_store.py
│   │
│   ├── workflow/
│   │   └── graph.py
│   │
│   └── main.py
│
├── frontend/
│   └── streamlit_app.py
│
├── data/
│   └── documents/
│
├── vectorstore/
├── tests/
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env
└── README.md
```

🛠️ Tech Stack
Technology	Purpose
Python	Core development
FastAPI	Backend REST API
Streamlit	User interface
LangGraph	RAG workflow orchestration
FAISS	Vector similarity search
Gemini	Embeddings and answer generation
Pydantic	Data validation
PyPDF	PDF text extraction
Docker	Containerization
Docker Compose	Multi-container deployment
🚀 Local Setup
1. Clone the repository
git clone https://github.com/Manya-RS/Document-rag-assistant.git
cd Document-rag-assistant
2. Create a virtual environment
python -m venv venv

Activate it on Windows:

venv\Scripts\activate
3. Install dependencies
pip install -r requirements.txt

4. Configure the Gemini API key
Create a .env file:
GEMINI_API_KEY=your_api_key_here

Do not commit the .env file.

5. Start the FastAPI backend
uvicorn app.main:app --reload

API: http://127.0.0.1:8000

Swagger documentation:

http://127.0.0.1:8000/docs
6. Start the Streamlit frontend

In another terminal:

streamlit run frontend/streamlit_app.py

Frontend:

http://localhost:8501
🐳 Docker Setup

Make sure Docker Desktop is running.

Build and start the application:

docker compose up --build

The services will be available at:

FastAPI:   http://localhost:8000
Swagger:   http://localhost:8000/docs
Streamlit: http://localhost:8501

Stop the services:

docker compose down
🔌 API Endpoints
Health Check
GET /health
Upload Document
POST /api/upload

Supported formats:

PDF
TXT
Markdown
Query Document
POST /api/query

Example request:

{
  "document_name": "example.pdf",
  "question": "What is the main topic of this document?",
  "top_k": 3
}
Streaming Query
POST /api/query/stream

The endpoint streams generated answer text to provide a responsive response experience.

🔍 RAG Pipeline

The question-answering pipeline follows these stages:

Document Selection
The user selects the document to query.
Query Embedding
The question is converted into an embedding.
Vector Retrieval
FAISS retrieves the most relevant document chunks.
Candidate Expansion
The system initially retrieves additional candidates for improved reranking.
Reranking
Retrieved chunks are reordered using lexical overlap together with vector distance.
Grounded Generation
The selected context is provided to Gemini with instructions to answer only from the retrieved document context.
Source Retrieval
The response includes the retrieved source chunks and their similarity distances.
🛡️ Reducing Hallucinations

The generation stage is designed to keep responses grounded in the retrieved document context.

The model is instructed to:

Use only the provided document context
Avoid outside knowledge
Avoid inventing information
Clearly state when the answer is not available in the provided document

Retrieved source chunks are also returned with the answer so the user can inspect the supporting context.

📊 Retrieval Quality Improvements

The system includes several retrieval improvements:

Overlapping text chunks
Semantic vector search
Candidate retrieval beyond the final top_k
Lightweight lexical reranking
Document-scoped vector indexes
Source chunks returned with the generated answer
🔄 LangGraph Workflow
START
  ↓
Retrieve
  ↓
Rerank
  ↓
Generate
  ↓
END

The workflow separates retrieval, reranking, and generation into modular stages.

🔐 Security

Sensitive configuration is stored in environment variables.

The following are intentionally excluded from version control:
.env
venv/
vectorstore/
data/documents/
__pycache__/

Never place API keys directly inside source code or commit them to GitHub.

🎯 Project Goal
The project demonstrates a production-style RAG pipeline capable of ingesting documents, creating vector indexes, retrieving relevant context, reranking results, and generating grounded answers through an API and interactive frontend.

👩‍💻 Author

Manya RS

GitHub: https://github.com/Manya-RS

⭐ Built as part of the ShadowFox Advanced AI Engineer internship task.
