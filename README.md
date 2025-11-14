# Groq-RAG-App

This is a Streamlit application that allows users to interact with large language models (LLMs) in two main ways:

1. **Simple Chat**: Chat directly with an LLM in a conversational interface.
2. **Chat with PDF files**: Upload a PDF document and ask questions based on its content. The app uses embeddings and vector search to provide relevant answers.

## Features

- Upload PDF files and generate embeddings for querying.
- Chat interface that supports streaming responses.
- Session state management to maintain chat history per session.
- Optional OpenAI API key validation in production mode.
- Modular design to support different LLMs (Groq, OpenAI, etc.).
