# RAG Research Assistant

A Retrieval-Augmented Generation (RAG) pipeline that answers computer science questions using academic papers downloaded from arXiv. The system retrieves relevant document chunks from a FAISS vector index and passes them as context to a language model, producing grounded answers.

---

## Overview

Large language models are powerful, but they hallucinate. This project addresses that by grounding responses using academic content as an exmaple.

When a user submits a question, the system embeds the query, searches a FAISS vector index of pre-processed documents, retrieves the most semantically relevant chunks, and passes them as context to Llama 3.2 running locally via Ollama. The model then generates an answer based on actual paper content rather than just LLM memory alone. Everything runs fully locally.

---

## How It Works

```
User Query
    │
    ▼
Embed Query
    │
    ▼
FAISS Similarity Search
    │
    ▼
Retrieve Top-K Chunks
    │
    ▼
LLM Generation
    │
    ▼
Grounded Answer
```

### Pipeline Steps

**1. Document Ingestion**
Research papers are fetched from arXiv and stored locally as PDFs.

**2. Chunking**
Documents are split into smaller semantic chunks using LangChain text splitters to improve retrieval precision and fit within context windows.

**3. Embedding**
Each chunk is converted into a vector embedding using a SentenceTransformer model. Embeddings are stored in a FAISS index.

**4. Retrieval**
At query time, the user's question is embedded and compared against the FAISS index using cosine similarity. The top-K most relevant chunks are returned.

**5. Generation**
Retrieved chunks are formatted as context and passed to Llama 3.2 (3B) via Ollama. The model generates a response grounded in the retrieved chunk content.

---
## DEMO

[![Demo]](media\Rag_demo.mp4)

---

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python |
| Vector Search | FAISS |
| Embeddings | SentenceTransformers |
| PDF Parsing | PyMuPDF |
| Text Splitting | LangChain |
| Paper Source | arXiv |
| Numerical Ops | NumPy |
| LLM | Llama 3.2 3B (via Ollama) |

---

## Installation

### Prerequisites

- Python 3.9+
- pip
- [Ollama](https://ollama.com) installed and running

### 1. Install Ollama

Download and install Ollama from [ollama.com](https://ollama.com), then pull the required models:

```bash
ollama pull llama3.2:3b
```

### 2. Clone & Install

```bash
git clone https://github.com/ArmandMeijers/CompSci-research-paper-RAG
cd CompSci-research-paper-RAG
pip install -r requirements.txt
```

### 3. Run

Make sure Ollama is running in the background, then:

```bash
python3 main.py
```

> **Note:** The `data/` directory is created automatically on first run. Expect ~2–3GB of disk usage depending on how many papers are downloaded.

---

## Project Structure

```
CompSci-research-paper-RAG/
├── data/               # All Raw/Processed data (created when main is run)
├── media/              # Demo content
├── src/
│   ├── ingest.py       # Chunking, Embedding generation and Indexing
│   ├── retrieve.py     # Query embedding and FAISS search
│   ├── downloader.py   # Downloads acedemic papers
│   └── generate.py     # LLM context formatting and generation
└── main.py             # Entry point
```