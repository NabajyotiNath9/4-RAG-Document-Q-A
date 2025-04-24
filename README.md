GenAI-Powered Research Paper Q&A App
Welcome to the GenAI-Powered Research Paper Q&A App! This application leverages cutting-edge GenAI technology for contextual question-answering from research papers, powered by Groq, LangChain, and Streamlit.

Overview
This app allows users to ingest research papers (in PDF format), parse and embed their content, and then query the documents for context-based answers. It uses a Retrieval-Augmented Generation (RAG) pipeline, combining vector-based search with Groq's Deepseek-R1-Distill-Llama-70b model for fast and accurate results.

Key technologies used:

Groq LPU-powered model inference

LangChain for document processing and retrieval

Ollama Embeddings for semantic vectorization

FAISS for efficient vector-based similarity search

Streamlit for a user-friendly, interactive interface

Features
PDF Document Ingestion: Upload a folder of PDFs for processing.

Contextual Q&A: Ask questions and get accurate answers based on the provided research paper.

Vector Search: Utilize FAISS for fast document similarity scoring.

Interactive UI: Build with Streamlit for an easy-to-use web interface.

Model Integration: Powered by Groq's Deepseek-R1-Distill-Llama-70b model, optimized for rapid inference.

How It Works
Document Ingestion: The app loads PDFs from the provided directory and splits them into manageable chunks.

Vectorization: Text chunks are transformed into embeddings using Ollama embeddings.

Search & Retrieval: The app retrieves the most relevant document chunks based on the user's query using FAISS vector search.

Answer Generation: The retrieved context is fed into the Deepseek-R1-Distill-Llama-70b model to generate a context-aware response.
