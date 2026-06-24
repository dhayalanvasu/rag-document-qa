# RAG Document Q&A System

A Retrieval-Augmented Generation (RAG) pipeline built from scratch in Python.

## What it does
- Converts documents into semantic embeddings using sentence-transformers
- Stores and searches embeddings using FAISS vector database
- Retrieves relevant chunks and passes them to an LLM for grounded answers

## Tech Stack
- Python 3.14
- sentence-transformers (embeddings)
- FAISS (vector search)
- Ollama / Anthropic Claude (LLM)

## How to run
```bash
python3 -m pip install sentence-transformers faiss-cpu requests
python3 work.py
```
