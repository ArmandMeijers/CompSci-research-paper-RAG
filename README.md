# RAG Research Assistant

A Retrieval-Augmented Generation (RAG) research assistant that answers computer science questions using academic papers. The system retrieves relevant document chunks from a vector database and provides them as context to a language model, enabling more accurate and grounded responses.

---

## Overview

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline designed to answer computer science questions using academic research papers.

Instead of relying solely on a language model’s internal knowledge, the system retrieves relevant document chunks from a **vector database** and provides them as context when generating responses. This approach reduces hallucinations.

The system processes research papers, converts them into vector embeddings, stores them in a FAISS vector index, and retrieves the most relevant information when a user submits a query.

This relevenat information is then passed into A LLM, and the awnser is displayed.

---

## How It Works

1. **Document Ingestion**  
   Research papers are downloaded and stored locally.

2. **Chunking**  
   Documents are split into smaller semantic chunks to improve retrieval accuracy.

3. **Embedding**  
   SentenceTransformer models convert chunks into vector embeddings.
   Embeddings are stored in a **FAISS index**.

4. **Retrieval**  
   User queries are embedded and compared against the FAISS index to retrieve the most relevant chunks.

5. **Generation**  
   The retrieved context is passed to a language model to generate grounded answers.



---

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/rag-research-assistant.git
cd rag-research-assistant


# Technologies
- Python
- NumPy
- FAISS (vector similarity search)
- SentenceTransformers
- PyMuPDF
- LangChain text splitters
- arXiv APIs