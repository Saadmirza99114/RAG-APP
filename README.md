# 📁 Scalable RAG File Vault

A **Gradio-based** all-in-one web application that lets you **upload, embed, store, query, and delete** documents inside a fully-managed **RAG (Retrieval-Augmented Generation)** pipeline.

---

## 🧩 What It Does

| Step | Feature |
|------|---------|
| **0.** | One-time setup of API credentials (Google AI, Supabase, Qdrant) |
| **1.** | Drag-and-drop batch upload of **PDF, DOCX, TXT, PPTX, CSV, XLSX, MD, HTML** |
| **2.** | Automatic text extraction + chunking + embedding (Gemini 768-D) |
| **3.** | Stores raw files in **Supabase Storage** and vectors in **Qdrant** |
| **4.** | Conversational query interface with source citations |
| **5.** | Real-time storage usage dashboards & per-file deletion |

---

## 🚀 Quick Start

1. **Install requirements**
   ```bash
   pip install -r requirements.txt

    Grab your keys
        Google AI Studio → Google API Key
        Supabase → Project URL + Service Key
        Qdrant Cloud → REST URL + API Key
    Launch
    bash

    Copy

    python app.py

    A browser window will open at http://localhost:7860.

📦 Requirements (requirements.txt)
Copy

gradio>=4.0
google-generativeai
supabase
qdrant-client[fastembed]
PyMuPDF
pdfplumber
python-docx
python-pptx
pandas
beautifulsoup4
langchain-text-splitters
tqdm

🔧 Architecture
Table
Copy
Component	Purpose
Google Gemini	Embedding & LLM
Supabase Storage	Persistent object storage (1 GB free tier)
Qdrant	Vector similarity search (100 k vectors free tier)
Gradio	Zero-code web UI
🧪 Core Functions
Table
Copy
Name	Description
extract_text_from_file	Multi-engine PDF fallback → DOCX/PPTX/CSV/TXT/HTML
embed_text	LangChain chunker + batch Gemini embeddings
store_file_and_embeddings	Upserts to Supabase + Qdrant (batched)
rag_query	Embed query → top-k retrieval → Gemini answer
delete_file	Purge file from Supabase + vectors from Qdrant
setup_storage_and_vector_db	Auto-creates bucket & collection at startup
🔐 Environment Variables
The UI will ask for these once and keep them in memory.
Table
Copy
Variable	Example
GOOGLE_API_KEY	AIza...
SUPABASE_URL	https://xyz.supabase.co
SUPABASE_KEY	eyJhbGciOi...
QDRANT_URL	https://xyz.qdrant.io:6333
QDRANT_API_KEY	********
🛠️ Configuration Tweaks
Open the script and adjust:
Table
Copy
Constant	Default	Purpose
BUCKET_NAME	"rag-file"	Supabase bucket
COLLECTION_NAME	"rag_embeddings"	Qdrant collection
CHUNK_SIZE	1000	Tokens per chunk
BATCH_SIZE	128	Qdrant upsert batch
DEFAULT_EMBEDDING_BATCH_SIZE	50	Gemini embedding batch
📊 Usage Limits
Table
Copy
Service	Limit
Supabase free	1 GB storage
Qdrant free	100 k vectors
Gemini	60 RPM (embeddings)
🧼 File Deletion

    Exact filename required (case-insensitive).
    Deletes both the raw file and all related vectors.

🐞 Troubleshooting
Table
Copy
Symptom	Fix
pdfminer spam	Already silenced in code.
docx corruption	Detects invalid zip and skips gracefully.
Qdrant index error	Auto-creates keyword index on file_name.
