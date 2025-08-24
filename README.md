# 📁 RAG File Vault

An all-in-one scalable RAG (Retrieval-Augmented Generation) app built with **Gradio**, **Google GenAI**, **Supabase**, and **Qdrant**. This project allows you to upload, process, query, and manage documents in a complete RAG pipeline.

---

## 🚀 Features

* **Multi-format Document Upload**: Supports `.pdf`, `.docx`, `.txt`, `.pptx`, `.csv`, `.xlsx`, `.md`, `.html`.
* **Robust Text Extraction**: Uses PyMuPDF, pdfplumber, and format-specific parsers.
* **Embeddings Generation**: Splits documents into chunks and generates embeddings with `models/embedding-001`.
* **Cloud Storage**: Files stored in Supabase.
* **Vector Storage**: Embeddings stored in Qdrant.
* **Query Interface**: Retrieval-Augmented Generation powered by `gemini-1.5-flash`.
* **Storage Management**: Check used storage (files + vectors).
* **Deletion Support**: Delete both file and its embeddings.

---

## ⚙️ Configuration & Constants

```python
BUCKET_NAME = "rag-file"
COLLECTION_NAME = "rag_embeddings"
BATCH_SIZE = 128  # Qdrant upsert batch size
EMBED_MODEL = "models/embedding-001"
CHAT_MODEL_NAME = "gemini-1.5-flash"
SUPABASE_TOTAL_BYTES = 1073741824  # 1 GB
QDRANT_MAX_VECTORS = 100000
DEFAULT_EMBEDDING_BATCH_SIZE = 50
```

---

## 🔑 Setup

### 1. API Keys Required

* Google API Key
* Supabase URL & Service Key
* Qdrant URL & API Key

### 2. Install Dependencies

```bash
pip install gradio qdrant-client supabase pdfplumber pymupdf python-docx python-pptx pandas beautifulsoup4
```

### 3. Run the App

```bash
python app.py
```

The app will launch a **Gradio interface** in your browser.

---

## 🖥️ Gradio Tabs

1. **Set API Keys** – Configure Google, Supabase, and Qdrant credentials.
2. **Upload & Process** – Upload documents, extract text, embed, and store.
3. **Query Documents** – Ask questions and receive AI-generated answers with retrieved context.
4. **Manage Storage** – Monitor storage usage in Supabase & Qdrant.
5. **Delete Files** – Permanently delete files and embeddings.

---

## 🔄 Core Functions

* **`extract_text_from_file(file_path)`** – Extract text from multiple file formats.
* **`embed_text(text)`** – Generate embeddings in batches.
* **`store_file_and_embeddings(...)`** – Upload files to Supabase & embeddings to Qdrant.
* **`rag_query(...)`** – Perform retrieval-augmented query.
* **`delete_file(...)`** – Remove files and/or vectors.
* **`setup_storage_and_vector_db(...)`** – Initialize Supabase bucket & Qdrant collection.

---

## 🧩 Example Usage

```python
answer, context = rag_query(qdrant_client, "What are the laws of thermodynamics?")
print("Answer:", answer)
print("Context:", context)
```

---

## 📦 Project Structure

```
├── app.py               # Main Gradio app
├── requirements.txt     # Python dependencies
├── README.md            # Project documentation
```

---

## ✅ Status

* Supports multiple file formats ✅
* RAG query with Gemini Flash ✅
* Cloud storage + vector DB integration ✅
* File deletion + storage monitoring ✅

---

## 📜 License

This project is licensed under the MIT License.
