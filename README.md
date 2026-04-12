# RAG Research Assistant

A Retrieval-Augmented Generation (RAG) pipeline that answers computer science questions using academic papers downloaded from arXiv. The system retrieves relevant document chunks from a FAISS vector index and passes them as context to a language model, producing grounded awnsers.

---

## Overview

Large language models are powerful, but they hallucinate. This project addresses that by grounding responses.

When a user submits a question, the system embeds the query, searches a FAISS vector index of pre-processed documents, retrieves the most semantically relevant chunks, and passes them as context to an LLM model (i am using claude). The model then generates an answer based on actual paper content rather than just LLM memory alone.

---

## How It Works

```
User Query
    │
    ▼
Embed Query (SentenceTransformers)
    │
    ▼
FAISS Similarity Search
    │
    ▼
Retrieve Top-K Chunks
    │
    ▼
LLM Generation (with retrieved context)
    │
    ▼
Grounded Answer
```

### Pipeline Steps

**1. Document Ingestion**
Research papers are fetched from arXiv via API and stored locally as PDFs.

**2. Chunking**
Documents are split into smaller semantic chunks using LangChain text splitters to improve retrieval precision and fit within context windows.

**3. Embedding**
Each chunk is converted into a dense vector embedding using a SentenceTransformer model. Embeddings are stored persistently in a FAISS index.

**4. Retrieval**
At query time, the user's question is embedded and compared against the FAISS index using cosine similarity. The top-K most relevant chunks are returned.

**5. Generation**
Retrieved chunks are formatted as context and passed to an LLM. The model generates a response grounded in the retrieved paper content.

---

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python |
| Vector Search | FAISS |
| Embeddings | SentenceTransformers |
| PDF Parsing | PyMuPDF |
| Text Splitting | LangChain |
| Paper Source | arXiv API |
| Numerical Ops | NumPy |

---

## Installation

### Prerequisites

- Python 3.9+
- pip

### Clone & Install

```bash
git clone https://github.com/ArmandMeijers/CompSci-research-paper-RAG
cd CompSci-research-paper-RAG
pip install -r requirements.txt
```

Then just
```bash
Python3 main.py
```

---

## Project Structure

```
CompSci-research-paper-RAG/
├── data/               # All Raw/Processed data (created when main ran)
├── src/
│   ├── ingest.py       # Paper downloading and chunking
│   ├── embed.py        # Embedding generation and indexing
│   ├── retrieve.py     # Query embedding and FAISS search
│   └── generate.py     # LLM context formatting and generation
└── main.py             # Entry point
```

---

