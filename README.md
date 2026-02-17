# AI Engineer Portfolio Projects

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![LangChain](https://img.shields.io/badge/LangChain-0.2-green)](https://langchain.com/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-orange)](https://huggingface.co/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue?logo=docker)](https://www.docker.com/)

A production-grade AI Engineering portfolio showcasing end-to-end implementations across LLM fine-tuning, Retrieval-Augmented Generation (RAG), Agentic AI workflows, Computer Vision, and MLOps. Every module is built with clean, maintainable code and real-world deployment in mind.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Projects Included](#projects-included)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
- [Configuration](#configuration)
- [Running Tests](#running-tests)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

This repository is a collection of AI engineering projects that I have built over 6+ years working in AI/ML, Generative AI, and Data Science. Each project follows software engineering best practices:

- Clean code architecture (modules, separation of concerns)
- Type hints and docstrings throughout
- Configurable via YAML / environment variables
- Unit and integration tests
- Docker support for reproducible environments
- CI/CD ready

---

## Project Structure

```
AI-Engineer-Portfolio-Projects/
├── src/
│   ├── agents/                  # Agentic AI workflows (LangGraph, AutoGen)
│   │   ├── base_agent.py
│   │   ├── rag_agent.py
│   │   └── tool_calling_agent.py
│   ├── models/                  # LLM fine-tuning and inference
│   │   ├── fine_tune.py
│   │   ├── inference.py
│   │   └── embeddings.py
│   ├── pipelines/               # RAG and data pipelines
│   │   ├── rag_pipeline.py
│   │   ├── data_ingestion.py
│   │   └── vector_store.py
│   ├── computer_vision/         # CV models and inference
│   │   ├── object_detection.py
│   │   └── image_classifier.py
│   └── utils/                   # Shared utilities
│       ├── logger.py
│       ├── config_loader.py
│       └── helpers.py
├── tests/                       # Unit and integration tests
│   ├── test_agents.py
│   ├── test_rag_pipeline.py
│   └── test_models.py
├── notebooks/                   # Jupyter notebooks for exploration
├── configs/                     # YAML config files per environment
│   ├── config.yaml
│   └── config.dev.yaml
├── scripts/                     # Helper scripts (data prep, evaluation)
│   ├── prepare_data.py
│   └── evaluate_model.py
├── .env.example
├── .gitignore
├── requirements.txt
├── setup.py
├── Dockerfile
├── docker-compose.yml
├── CONTRIBUTING.md
├── CHANGELOG.md
└── README.md
```

---

## Projects Included

### 1. RAG Pipeline with LangChain + FAISS
Builds a full retrieval-augmented generation system. Documents are chunked, embedded, and stored in a FAISS vector store. A LangChain retrieval chain fetches relevant context before passing it to the LLM for generation.

### 2. LLM Fine-Tuning with LoRA / QLoRA
Fine-tunes open-source LLMs (Llama 3, Mistral) on custom instruction datasets using PEFT (LoRA/QLoRA). Includes training scripts, evaluation, and model push to Hugging Face Hub.

### 3. Agentic AI Workflow (LangGraph)
Multi-step autonomous agent that uses tool calling, memory, and conditional branching. Demonstrates ReAct-style reasoning with real API tools (web search, code execution, database lookup).

### 4. Computer Vision Object Detection (YOLOv8)
Real-time object detection pipeline using Ultralytics YOLOv8. Includes custom dataset training, inference on images/video, and REST API serving via FastAPI.

### 5. MLOps Pipeline with MLflow + FastAPI
End-to-end MLOps: experiment tracking with MLflow, model registry, automated retraining triggers, and model serving with FastAPI. Containerized with Docker.

---

## Tech Stack

| Category | Libraries / Tools |
|---|---|
| LLMs & GenAI | OpenAI, HuggingFace Transformers, LangChain, LlamaIndex |
| Fine-Tuning | PEFT, TRL, bitsandbytes, Unsloth |
| Agentic AI | LangGraph, AutoGen, CrewAI |
| Vector Stores | FAISS, ChromaDB, Pinecone |
| Computer Vision | Ultralytics YOLOv8, OpenCV, Pillow |
| ML Frameworks | PyTorch, scikit-learn |
| MLOps | MLflow, DVC, FastAPI |
| Infra | Docker, Docker Compose, GitHub Actions |
| Testing | pytest, unittest |

---

## Getting Started

### Prerequisites

- Python 3.10+
- pip or conda
- Docker (optional, for containerized runs)
- CUDA-compatible GPU (recommended for fine-tuning)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/SURENDER294/AI-Engineer-Portfolio-Projects.git
cd AI-Engineer-Portfolio-Projects

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
cp .env.example .env
# Edit .env with your API keys
```

---

## Configuration

All project settings live in `configs/config.yaml`. You can override values with environment variables or a `config.dev.yaml` file.

```yaml
# configs/config.yaml
llm:
  model_name: "meta-llama/Llama-3-8b-instruct"
  temperature: 0.7
  max_tokens: 1024

rag:
  chunk_size: 512
  chunk_overlap: 64
  top_k: 5

vector_store:
  type: "faiss"
  index_path: "./data/faiss_index"
```

---

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=src --cov-report=html
```

---

## Deployment

### Docker

```bash
# Build and run with Docker Compose
docker-compose up --build

# Or run a single container
docker build -t ai-engineer-portfolio .
docker run -p 8000:8000 --env-file .env ai-engineer-portfolio
```

The FastAPI server will be available at `http://localhost:8000`.
Swagger docs: `http://localhost:8000/docs`

---

## Contributing

Contributions, issues, and feature requests are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md) first.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

> Built with passion by [Surender Reddy](https://github.com/SURENDER294) — AI/ML Engineer | GenAI | LLMs | Agentic AI
