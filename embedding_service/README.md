# Embedding Service

## Overview
FastAPI microservice for generating sentence embeddings using Sentence-BERT model - sentence-transformers/all-MiniLM-L6-v2

## Files
- `app.py` - Main FastAPI application for embedding generation
- `requirements.txt` - Python dependencies for the service
- `Dockerfile` - Containerisation specs

## Features
- **Sentence-BERT**: High-quality 384-dimensional embeddings
- **FastAPI**: High-performance async REST API
- **Batch Processing**: Efficient bulk embedding generation
- **Docker Ready**: Containerized for easy deployment
- **Health Checks**: Built-in service monitoring endpoints

## Usage
```bash
# Local development
pip install -r requirements.txt
python3 app.py

# Docker deployment
docker build -t embedding-service .

# Service available at: http://localhost:8000
```

## API Endpoints
- `POST /embed` - Generate embeddings for text input
- `POST /embed_batch` - Bulk embedding generation
- `GET /health` - Service health check
- `GET /` - API documentation

## Dependencies
- sentence-transformers
- FastAPI
- uvicorn
- torch
- transformers
