# Insurance RAG AI -- Production-Ready Generative AI System

A production-style Retrieval-Augmented Generation (RAG) system designed
for insurance policy and compliance analysis.

This project demonstrates how to build a grounded LLM application with
semantic caching, vector search, and cost-aware inference optimization
using modern AI engineering practices.

------------------------------------------------------------------------

## Features

-   Vector search using FAISS
-   Local embeddings with SentenceTransformers
-   LLM-powered answer generation (OpenAI GPT-4o-mini)
-   Grounded responses with source citation
-   Semantic caching (embedding similarity-based)
-   Confidence scoring
-   Latency tracking
-   REST API using FastAPI
-   Cold start mitigation for production environments

------------------------------------------------------------------------

## Architecture

User Query\
→ FastAPI Endpoint\
→ Semantic Cache Check\
→ (if miss) Vector Retrieval (FAISS)\
→ Context Assembly\
→ LLM Generation\
→ Structured JSON Response

------------------------------------------------------------------------

## Tech Stack

-   Python 3.9+
-   FastAPI
-   FAISS
-   SentenceTransformers
-   OpenAI API (gpt-4o-mini)
-   Pydantic
-   NumPy

------------------------------------------------------------------------

## Project Structure

insurance-rag-ai/\
│\
├── app/\
│ ├── main.py\
│ ├── rag.py\
│ ├── embeddings.py\
│ ├── cache.py\
│ └── ingest.py\
│\
├── data/\
├── vector_store/\
├── requirements.txt\
└── README.md

------------------------------------------------------------------------

## Example API Response

{ "answer": "The policy covers bodily injury and property damage up to
\$1,000,000 per occurrence.", "sources": \["sample_policy.txt"\],
"confidence": 0.47, "cache_hit": true, "latency_seconds": 0.03 }

------------------------------------------------------------------------

## Why This Project Matters

This implementation showcases:

-   Production-grade RAG architecture
-   Embedding-based semantic caching
-   Cost-aware AI system design
-   Latency optimization strategies
-   Grounded LLM responses with reduced hallucination risk

Designed as a minimal yet realistic blueprint for deploying Generative
AI systems in regulated domains such as insurance and compliance.

------------------------------------------------------------------------

## Setup

python3 -m venv .venv\
source .venv/bin/activate\
pip install -r requirements.txt

Create a `.env` file:

OPENAI_API_KEY=your_key_here\
LLM_MODEL=gpt-4o-mini

Build vector index:

python -m app.ingest

Or use the Makefile:

make setup

Run API:

uvicorn app.main:app --reload

Or:

make run

------------------------------------------------------------------------

## Development

### Testing

This project includes comprehensive test coverage:

```bash
# Run all tests (excluding slow tests)
make test

# Run only unit tests
make test-unit

# Run integration tests  
make test-integration

# Run all tests including slow ones
make test-all

# Test with coverage report
pytest --cov=app --cov-report=html
```

### Code Quality

```bash
# Format code
make format

# Run linting and security checks
make lint

# Install development dependencies
make dev-setup
```

### Project Data

The project now includes comprehensive realistic insurance policy documents:

- **Auto Insurance**: Comprehensive coverage, deductibles, exclusions
- **Health Insurance**: PPO plan with preventive care, specialist coverage
- **Homeowners Insurance**: Dwelling, liability, personal property coverage  
- **Life Insurance**: Whole life policy with cash value accumulation
- **Business Insurance**: Commercial general liability and property coverage

### CI/CD

GitHub Actions workflow includes:
- Automated testing across Python 3.9, 3.10, 3.11
- Code quality checks (black, isort, flake8, mypy)
- Security scanning (safety, bandit)
- Coverage reporting
- Integration testing

------------------------------------------------------------------------

## Future Improvements

-   Cosine similarity instead of L2
-   Token-aware chunking
-   Evaluation benchmarking framework
-   Observability and logging dashboard
-   Support for local LLM inference (Ollama)
-   Advanced caching strategies
-   Multi-tenant support
-   Real-time document updates
