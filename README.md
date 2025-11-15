# PrivaChat Agents 🔍

**Advanced AI-powered research service with RAG, document processing, and multi-agent intelligence.**

Built with Python, FastAPI, Pydantic AI, PostgreSQL, and OpenRouter - delivering comprehensive research capabilities with cited sources.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green.svg)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com)

---

## ✨ Features

🚀 **High Performance**
- Redis caching for 75% cost reduction ($730/year → $182/year)
- Async processing for 3x faster responses
- HTTP connection pooling for 30-50% faster API calls
- Sub-6 second response times for complex queries

🤖 **Multi-Agent Intelligence**
- **Search Agent**: Query decomposition and source discovery
- **Research Agent**: Deep analysis and synthesis
- **Synthesis Agent**: Citation-rich answer generation
- **Temporal Detection**: SpaCy-powered time-aware queries

📚 **RAG Pipeline (pgvector)**
- Document processing with Dockling
- 384-dimensional embeddings (Sentence Transformers)
- Semantic search with cosine similarity
- Full-text search (FTS) support

💰 **Cost Optimized**
- Free-tier LLM usage (DeepSeek via OpenRouter)
- Intelligent caching with configurable TTLs
- ~$15/month for moderate usage

🔍 **Privacy-Focused Search**
- SearxNG integration (no tracking)
- Crawl4AI for web crawling
- Document upload support
- Local processing

---

## 🏗️ Architecture

<div align="center">

```  

┌─────────────┐  
│   Client    │
└──────┬──────┘  
       │
       ▼
┌─────────────────────────────────────┐
│   FastAPI Server (Port 8001)        │
│   ┌──────────────────────────────┐  │
│   │  Research Pipeline           │  │
│   │  • Multi-agent orchestration │  │
│   │  • Query decomposition       │  │
│   │  • Source validation         │  │
│   │  • Temporal detection (SpaCy)│  │
│   └──────────────────────────────┘  │
│   ┌──────────────────────────────┐  │
│   │  Redis Cache Layer           │  │
│   │  • Response cache (1d TTL)   │  │
│   │  • Session storage           │  │
│   │  • 75% cost reduction        │  │
│   └──────────────────────────────┘  │
│   ┌──────────────────────────────┐  │
│   │  RAG Pipeline (pgvector)     │  │
│   │  • Document processing       │  │
│   │  • Embedding generation      │  │
│   │  • Semantic search           │  │
│   └──────────────────────────────┘  │
└─────────────────────────────────────┘
       │               │               │
       ▼               ▼               ▼
┌─────────────┐ ┌─────────────┐ ┌──────────────┐
│  SearxNG    │ │  PostgreSQL │ │    Redis     │
│  Port 4000  │ │  Port 5433  │ │  Port 6380   │
│             │ │  + pgvector │ │              │
└─────────────┘ └─────────────┘ └──────────────┘
```

</div>

**Performance Metrics:**
- First query (cold): ~5s
- Cached query: ~1s (5x faster)
- Research mode: <10s with 30+ sources
- Embedding generation: <100ms

---

## 🚀 Quick Start

### Prerequisites
- Docker and Docker Compose
- OpenRouter API key (free tier available)
- Python 3.11+ (for local development)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/chaitanyame/privachat_agents.git
cd privachat_agents
```

2. **Set up environment variables**
```bash
cp .env.example .env
# Edit .env and add your OpenRouter API key
```

Required environment variables:
```env
OPENROUTER_API_KEY=your_key_here
OPENROUTER_MODEL=openrouter/auto
POSTGRES_PASSWORD=your_secure_password
```

3. **Start the services**
```bash
docker compose up -d
```

This will start:
- **API Server** (port 8001) - Research API with FastAPI
- **Streamlit UI** (port 8503) - Testing interface
- **PostgreSQL** (port 5433) - Database with pgvector
- **Redis** (port 6380) - Cache layer
- **SearxNG** (port 4000) - Privacy-focused search

4. **Verify it's working**
```bash
curl http://localhost:8001/health
```

Access the Streamlit UI at: **http://localhost:8503**

---

## 📖 API Usage

### Basic Research Request

```bash
curl -X POST http://localhost:8001/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the latest developments in AI?",
    "mode": "search",
    "max_sources": 20
  }'
```

### Deep Research Mode

```bash
curl -X POST http://localhost:8001/api/v1/research \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Explain quantum computing and its applications",
    "mode": "research",
    "max_sources": 30
  }'
```

### Response Format

```json
{
  "session_id": "uuid-here",
  "query": "What are the latest developments in AI?",
  "answer": "Detailed answer with citations...",
  "sources": [
    {
      "title": "Article Title",
      "url": "https://example.com/article",
      "relevance_score": 0.95,
      "content_snippet": "..."
    }
  ],
  "execution_time_seconds": 5.2
}
```

---

## 🗂️ Project Structure

```
privachat_agents/
├── privachat_agents/           # Main Python package
│   ├── agents/                 # AI agents (Pydantic AI)
│   │   ├── search_agent.py     # Search orchestration
│   │   ├── research_agent.py   # Deep research
│   │   └── synthesis_agent.py  # Answer synthesis
│   ├── api/v1/                 # FastAPI routes
│   │   └── endpoints/
│   │       ├── search.py       # Search endpoints
│   │       └── research.py     # Research endpoints
│   ├── clients/                # External API clients
│   │   ├── searxng_client.py   # SearxNG integration
│   │   └── web_crawler.py      # Crawl4AI wrapper
│   ├── core/                   # Core configuration
│   │   ├── config.py           # Settings management
│   │   └── pipelines/          # Processing pipelines
│   ├── database/               # SQLAlchemy models
│   │   ├── models.py           # Database tables
│   │   └── repositories/       # Data access layer
│   ├── models/                 # Pydantic schemas
│   │   └── schemas/            # Request/response models
│   ├── rag/                    # RAG pipeline
│   │   ├── vector_store.py     # pgvector operations
│   │   └── embeddings.py       # Sentence transformers
│   ├── services/               # Business logic
│   │   ├── llm/                # LLM integrations
│   │   └── cache/              # Redis caching
│   └── main.py                 # FastAPI application
├── tests/                      # Test suite
│   ├── unit/                   # Unit tests
│   ├── integration/            # Integration tests
│   └── e2e/                    # End-to-end tests
├── alembic/                    # Database migrations
├── config/searxng/             # SearxNG settings
├── docs/                       # Documentation
├── scripts/                    # Utility scripts
├── ui/                         # User interfaces
│   └── streamlit_app.py        # Testing UI
├── docker-compose.yml          # Service orchestration
├── Dockerfile                  # API container
├── Dockerfile.streamlit        # UI container
├── pyproject.toml              # Package configuration
├── requirements.txt            # Dependencies
└── requirements-dev.txt        # Dev dependencies
```

---

## 🎛️ Configuration

### Environment Variables

```env
# Required
OPENROUTER_API_KEY=sk-or-...
OPENROUTER_MODEL=openrouter/auto
POSTGRES_PASSWORD=your_secure_password

# Database
DATABASE_URL=postgresql+asyncpg://research_user:${POSTGRES_PASSWORD}@postgres:5432/research_db
POSTGRES_USER=research_user
POSTGRES_DB=research_db

# Redis Cache
REDIS_URL=redis://redis:6379/0
REDIS_ENABLED=true

# Search
SEARXNG_API_URL=http://searxng:8080

# Server
API_HOST=0.0.0.0
API_PORT=8001
LOG_LEVEL=INFO

# Optional: Langfuse Monitoring
LANGFUSE_ENABLED=false
LANGFUSE_PUBLIC_KEY=pk-...
LANGFUSE_SECRET_KEY=sk-...
```

### Docker Compose Ports

- `8001` - Research API (FastAPI)
- `8503` - Streamlit UI
- `5433` - PostgreSQL (with pgvector)
- `6380` - Redis cache
- `4000` - SearxNG search engine

---

## 🔧 Development

### Running Locally (Without Docker)

```bash
# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Download SpaCy model
python -m spacy download en_core_web_sm

# Set environment variables
export OPENROUTER_API_KEY=your_key_here
export DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5433/research
export REDIS_URL=redis://localhost:6380

# Run database migrations
alembic upgrade head

# Run the server
uvicorn privachat_agents.main:app --reload --host 0.0.0.0 --port 8001
```

### Viewing Logs

```bash
# All services
docker compose logs -f

# Specific service
docker compose logs -f api

# Search for specific patterns
docker compose logs api | grep -i "search\|error"
```

### Rebuilding After Changes

```bash
docker compose build api
docker compose up -d api
```

---

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest --cov=privachat_agents --cov-report=html tests/

# Run specific test categories
pytest tests/unit/ -v          # Unit tests only
pytest tests/integration/ -v   # Integration tests
pytest tests/e2e/ -v           # End-to-end tests

# Run marked tests
pytest -m "unit and fast" -v
```

### Quick Test Scripts

```bash
# Windows
run_tests.bat
run_all_tests.bat

# Linux/Mac
./run_tests.sh
```

### Code Quality

```bash
# Linting
ruff check .
ruff check --fix .

# Type checking
mypy privachat_agents/

# Format code
ruff format .
```

---

## 📊 Performance Optimizations

The system includes 4 major performance optimizations:

### 1. Redis Caching (75% cost reduction)
- **Response cache**: 1-day TTL
- **Session storage**: Persistent across restarts
- **Result**: Identical queries 5x faster (5s → 1s)
- **Savings**: $730/year → $182/year

### 2. Async Processing (3x faster)
- Concurrent source fetching with `asyncio.gather()`
- **Before**: 9+ seconds for 3 URLs
- **After**: ~3 seconds (3x faster)

### 3. HTTP Connection Pooling (30-50% faster)
- Singleton `httpx.AsyncClient`
- 100 max connections, 20 keep-alive
- Connection reuse across all API calls

### 4. Optimized Embeddings
- Sentence Transformers (384D)
- Batch processing support
- CPU/GPU adaptive

See [docs/PROCESS_FLOWS.md](docs/PROCESS_FLOWS.md) for detailed architecture.

---

## 📈 Monitoring

### Health Check
```bash
curl http://localhost:8001/health
```

### API Documentation
```bash
# Interactive API docs (Swagger UI)
open http://localhost:8001/docs

# Alternative docs (ReDoc)
open http://localhost:8001/redoc
```

### Container Status
```bash
docker compose ps
```

### Database Status
```bash
docker compose exec postgres psql -U research_user -d research_db -c "SELECT count(*) FROM research_sessions;"
```

### View Metrics (if Langfuse enabled)
Access Langfuse dashboard at your configured URL

---

## 🐛 Troubleshooting

### API returns empty responses
- Check OpenRouter API key is set correctly: `echo $OPENROUTER_API_KEY`
- Check logs: `docker compose logs api`
- Verify health endpoint: `curl http://localhost:8001/health`

### Cache not working
- Verify Redis is running: `docker compose ps redis`
- Check Redis connection: `docker compose exec redis redis-cli PING`
- Review cache logs: `docker compose logs api | grep -i cache`

### Slow responses
- Check if caching is enabled (`REDIS_ENABLED=true`)
- Verify async processing is working (logs show "concurrent")
- Monitor Redis: `docker stats`

### Database errors
- Check migrations: `alembic current`
- Run migrations: `alembic upgrade head`
- Check pgvector extension: `docker compose exec postgres psql -U research_user -d research_db -c "\dx"`

### SearxNG errors
- Ensure SearxNG is running: `curl http://localhost:4000`
- Check SearxNG logs: `docker compose logs searxng`
- Verify configuration in `config/searxng/settings.yml`

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Follow TDD principles (tests first!)
4. Commit your changes (`git commit -m 'Add amazing feature'`)
5. Push to the branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

**Development Standards:**
- ✅ Write tests BEFORE implementation (TDD)
- ✅ Minimum 80% test coverage
- ✅ Type hints on all functions
- ✅ Docstrings on all public APIs
- ✅ Pass ruff and mypy checks

See [CONTRIBUTING.md](CONTRIBUTING.md) and [.github/copilot-instructions.md](.github/copilot-instructions.md) for detailed guidelines.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Built upon concepts from [Perplexica](https://github.com/ItzCrazyKns/Perplexica)
- Uses [SearxNG](https://github.com/searxng/searxng) for privacy-focused search
- Powered by [OpenRouter](https://openrouter.ai/) for affordable LLM access
- Enhanced with [Pydantic AI](https://ai.pydantic.dev/) for type-safe AI agents
- Document processing with [Dockling](https://github.com/DS4SD/docling)
- Web crawling with [Crawl4AI](https://crawl4ai.com/)

---

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/chaitanyame/privachat_agents/issues)
- **Documentation**: See the `docs/` folder or [online docs](https://github.com/chaitanyame/privachat_agents/tree/main/docs)
- **Contributing**: See [CONTRIBUTING.md](CONTRIBUTING.md)
- **Code of Conduct**: See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)
- **Security**: See [SECURITY.md](SECURITY.md)

---

## 🎯 Roadmap

See [ROADMAP.md](ROADMAP.md) for planned features and improvements.

**Upcoming:**
- [ ] Multi-modal document support (PDFs, images, videos)
- [ ] Advanced citation tracking
- [ ] Custom agent workflows
- [ ] Real-time collaboration features
- [ ] Enhanced monitoring and observability

---

**Made with ❤️ for the AI agent community**
