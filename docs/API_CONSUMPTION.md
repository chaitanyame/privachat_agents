# API Consumption Guide

A comprehensive guide for developers to integrate and use the PrivaChat Agents API in their applications.

---

## Quick Start

### 1. API is Live

Start the API with Docker:

```bash
docker-compose up -d
```

### 2. Access Interactive Docs

Open your browser:
- **Swagger UI**: http://localhost:8001/api/docs
- **ReDoc**: http://localhost:8001/api/redoc

### 3. Make Your First Request

**cURL**:
```bash
curl -X POST http://localhost:8001/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query": "What is AI?"}'
```

**Python**:
```python
import requests

response = requests.post(
    "http://localhost:8001/api/v1/search",
    json={"query": "What is AI?"}
)
print(response.json()["answer"])
```

---

## Getting Started

### Prerequisites

- **API running**: `docker-compose up -d` (see [README.md](../README.md))
- **Python 3.8+** or **Node.js 14+** (depending on your language)
- **HTTP client**: requests, httpx, fetch, axios, etc.

### Installation (Python)

```bash
# Synchronous client
pip install requests

# Asynchronous client (recommended)
pip install httpx
```

### Installation (JavaScript/Node.js)

```bash
npm install axios
# or
npm install node-fetch
```

---

## Core Concepts

### Sessions

Every request creates a **session** with a unique `session_id`. You can retrieve previous results:

```python
# Get session results
response = requests.get(
    f"http://localhost:8001/api/v1/sessions/{session_id}"
)
```

### Search Modes

**Choose based on your needs**:

| Mode | Speed | Sources | Cost | Use Case |
|------|-------|---------|------|----------|
| `speed` | ⚡ Fast | 5-10 | Low | Quick answers |
| `balanced` | ⚖️ Medium | 10-15 | Medium | **Default, best for most** |
| `deep` | 🔍 Slow | 15-20 | High | Comprehensive research |

### Search Engines

| Engine | Type | Cost | Speed | Quality |
|--------|------|------|-------|---------|
| `searxng` | Open-source | Free | ⚡ Medium | ⭐⭐⭐ |
| `serperdev` | Google API | $ | ⚡ Fast | ⭐⭐⭐⭐ |
| `perplexity` | AI-powered | $$ | 🐢 Slow | ⭐⭐⭐⭐⭐ |
| `auto` | Fallback chain | Varies | ⚖️ Balanced | ⭐⭐⭐⭐ |

---

## API Endpoints Reference

### Search Endpoint

**Fast web search with answer synthesis**

```
POST /api/v1/search
```

**Minimal example** (query only):
```json
{
  "query": "What is climate change?"
}
```

**Full example** (all options):
```json
{
  "query": "What is climate change?",
  "mode": "balanced",
  "max_sources": 20,
  "timeout": 60,
  "model": "google/gemini-2.0-flash-lite-001",
  "search_engine": "auto",
  "prompt_strategy": "auto",
  "enable_diversity": true,
  "enable_recency": false,
  "enable_query_aware": false
}
```

**Response** (key fields):
```json
{
  "session_id": "uuid",
  "answer": "AI-generated answer...",
  "sources": [
    {
      "title": "Source Title",
      "url": "https://...",
      "snippet": "Content excerpt",
      "relevance": 0.95
    }
  ],
  "execution_time": 3.45,
  "confidence": 0.92
}
```

---

## Usage Examples

### Example 1: Simple Search (Python)

```python
import requests

def search(query: str, mode: str = "balanced") -> dict:
    """Perform a simple search."""
    response = requests.post(
        "http://localhost:8001/api/v1/search",
        json={"query": query, "mode": mode}
    )
    response.raise_for_status()
    return response.json()


# Usage
result = search("What is Pydantic AI?")
print(f"Answer: {result['answer']}")
print(f"Sources: {len(result['sources'])}")
print(f"Confidence: {result['confidence']:.1%}")
```

### Example 2: Async Search (Python)

```python
import httpx
import asyncio

async def search_async(query: str) -> dict:
    """Perform an async search."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8001/api/v1/search",
            json={"query": query}
        )
        response.raise_for_status()
        return response.json()


# Usage
async def main():
    result = await search_async("What is Pydantic AI?")
    print(result["answer"])

asyncio.run(main())
```

### Example 3: Search with Session Tracking (Python)

```python
import requests
import json

response = requests.post(
    "http://localhost:8001/api/v1/search",
    json={"query": "Python best practices"}
)
result = response.json()

# Save session for later
session_id = result["session_id"]
print(f"Session ID: {session_id}")

# Retrieve later
later = requests.get(
    f"http://localhost:8001/api/v1/sessions/{session_id}"
)
cached = later.json()
print(cached["result"]["answer"])
```

### Example 4: Batch Searches (Python)

```python
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

def search(query: str) -> dict:
    """Single search."""
    response = requests.post(
        "http://localhost:8001/api/v1/search",
        json={"query": query, "mode": "speed"}  # Fast mode
    )
    return response.json()


# Batch search multiple queries
queries = [
    "What is machine learning?",
    "What is deep learning?",
    "What is neural networks?",
]

with ThreadPoolExecutor(max_workers=3) as executor:
    futures = {
        executor.submit(search, q): q for q in queries
    }

    for future in as_completed(futures):
        query = futures[future]
        result = future.result()
        print(f"Q: {query}")
        print(f"A: {result['answer'][:100]}...\n")
```

### Example 5: Deep Research (Python)

```python
import requests
import time

def research(query: str, max_iterations: int = 3) -> dict:
    """Perform deep research."""
    response = requests.post(
        "http://localhost:8001/api/v1/research",
        json={
            "query": query,
            "mode": "deep",
            "max_iterations": max_iterations
        },
        timeout=600  # 10 minutes
    )
    response.raise_for_status()
    return response.json()


# Usage
start = time.time()
result = research("Impact of AI on software development")
elapsed = time.time() - start

print(f"Research completed in {elapsed:.1f}s")
print(f"Findings: {result['findings']}")
print(f"Citations: {len(result['citations'])}")
```

### Example 6: Document Upload & Query (Python)

```python
import requests

# Upload a document
with open("technical-paper.pdf", "rb") as f:
    upload_resp = requests.post(
        "http://localhost:8001/api/v1/documents",
        files={"file": f}
    )
upload_resp.raise_for_status()

doc_id = upload_resp.json()["document_id"]
print(f"Document uploaded: {doc_id}")

# Query the document
query_resp = requests.post(
    "http://localhost:8001/api/v1/documents/query",
    json={
        "query": "What are the main conclusions?",
        "document_ids": [doc_id],
        "max_sources": 5
    }
)
result = query_resp.json()
print(f"Answer: {result['answer']}")
```

### Example 7: JavaScript/Node.js

```javascript
// Using fetch (native)
async function search(query) {
  const response = await fetch("http://localhost:8001/api/v1/search", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, mode: "balanced" })
  });

  if (!response.ok) throw new Error(`API Error: ${response.status}`);
  return response.json();
}

// Usage
const result = await search("What is Pydantic AI?");
console.log(result.answer);
console.log(`Sources: ${result.sources.length}`);
```

### Example 8: JavaScript with Axios

```javascript
import axios from "axios";

const API_BASE = "http://localhost:8001";

const client = axios.create({
  baseURL: API_BASE,
  headers: { "Content-Type": "application/json" }
});

async function search(query, mode = "balanced") {
  const { data } = await client.post("/api/v1/search", { query, mode });
  return data;
}

// Usage
async function main() {
  try {
    const result = await search("Latest AI breakthroughs");
    console.log(result.answer);
  } catch (error) {
    console.error("Search failed:", error.message);
  }
}

main();
```

---

## Error Handling

### Common Errors & Solutions

#### 422 - Validation Error
```python
import requests

try:
    response = requests.post(
        "http://localhost:8001/api/v1/search",
        json={"query": ""}  # Empty query!
    )
    response.raise_for_status()
except requests.exceptions.HTTPError as e:
    if e.response.status_code == 422:
        errors = e.response.json()["detail"]
        for error in errors:
            print(f"Field {error['loc']}: {error['msg']}")
```

#### 504 - Timeout
```python
import requests

try:
    response = requests.post(
        "http://localhost:8001/api/v1/search",
        json={"query": "Complex query", "timeout": 30},
        timeout=60
    )
except requests.exceptions.Timeout:
    print("API request timed out - try increasing timeout or using 'speed' mode")
```

#### 500 - Server Error
```python
response.json()
# {
#   "error": "search_execution_failed",
#   "trace_id": "trace-xyz123",
#   "details": "Full error traceback"  # In DEBUG mode
# }

# Use trace_id to debug with Langfuse:
# https://langfuse.com/traces/{trace_id}
```

### Retry Logic

```python
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

def create_session_with_retries():
    """Create requests session with automatic retries."""
    session = requests.Session()

    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["POST", "GET", "DELETE"]
    )

    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    return session

# Usage
session = create_session_with_retries()
response = session.post("http://localhost:8001/api/v1/search", json={...})
```

---

## Best Practices

### 1. Choose the Right Mode

```python
# Quick answers: use 'speed'
response = requests.post(..., json={"query": q, "mode": "speed"})

# Most cases: use 'balanced' (default)
response = requests.post(..., json={"query": q})

# Comprehensive: use 'deep'
response = requests.post(..., json={"query": q, "mode": "deep", "timeout": 300})
```

### 2. Handle Timeouts Gracefully

```python
DEFAULT_TIMEOUT = 60

try:
    response = requests.post(
        "http://localhost:8001/api/v1/search",
        json={"query": "..."},
        timeout=DEFAULT_TIMEOUT
    )
except requests.exceptions.Timeout:
    # Fallback to speed mode with shorter timeout
    response = requests.post(
        "http://localhost:8001/api/v1/search",
        json={"query": "...", "mode": "speed", "timeout": 30},
        timeout=40
    )
```

### 3. Use Sessions for Reproducibility

```python
# Store session IDs for later reference
def search_and_save(query: str, filename: str) -> str:
    response = requests.post(
        "http://localhost:8001/api/v1/search",
        json={"query": query}
    )
    result = response.json()

    # Save session ID
    with open(filename, "w") as f:
        json.dump({"session_id": result["session_id"]}, f)

    return result

# Later, retrieve without re-running
with open("session.json") as f:
    session_id = json.load(f)["session_id"]

# Get cached results
response = requests.get(f"http://localhost:8001/api/v1/sessions/{session_id}")
```

### 4. Validate Responses

```python
import requests
from pydantic import BaseModel, ValidationError

class SearchResponse(BaseModel):
    answer: str
    sources: list
    confidence: float

try:
    raw_response = requests.post(...).json()
    result = SearchResponse(**raw_response)
    print(result.answer)
except ValidationError as e:
    print(f"Invalid response: {e}")
```

### 5. Implement Caching

```python
import requests
import json
from pathlib import Path

class CachedSearchClient:
    def __init__(self, cache_dir: str = ".search_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

    def _get_cache_key(self, query: str) -> str:
        import hashlib
        return hashlib.md5(query.encode()).hexdigest()

    def search(self, query: str) -> dict:
        cache_key = self._get_cache_key(query)
        cache_file = self.cache_dir / f"{cache_key}.json"

        # Check cache
        if cache_file.exists():
            with open(cache_file) as f:
                return json.load(f)

        # Fetch from API
        response = requests.post(
            "http://localhost:8001/api/v1/search",
            json={"query": query}
        )
        result = response.json()

        # Cache result
        with open(cache_file, "w") as f:
            json.dump(result, f)

        return result

# Usage
client = CachedSearchClient()
result = client.search("What is AI?")  # API call
result = client.search("What is AI?")  # Cached
```

### 6. Logging & Monitoring

```python
import logging
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def search_with_logging(query: str) -> dict:
    logger.info(f"Searching: {query}")

    response = requests.post(
        "http://localhost:8001/api/v1/search",
        json={"query": query}
    )

    result = response.json()
    logger.info(
        f"Search completed",
        extra={
            "session_id": result["session_id"],
            "execution_time": result["execution_time"],
            "confidence": result["confidence"],
            "sources": len(result["sources"])
        }
    )

    return result
```

---

## Production Deployment

### Environment Variables

```bash
# .env
API_BASE_URL=https://api.yourcompany.com
API_TIMEOUT=120
API_RETRIES=3
LOG_LEVEL=INFO
```

### Configuration

```python
import os
from pathlib import Path

class Config:
    API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8001")
    API_TIMEOUT = int(os.getenv("API_TIMEOUT", "60"))
    API_RETRIES = int(os.getenv("API_RETRIES", "3"))
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

config = Config()
```

### Rate Limiting Awareness

```python
import time
from datetime import datetime, timedelta

class RateLimiter:
    """Simple rate limiter to avoid overwhelming the API."""

    def __init__(self, requests_per_minute: int = 30):
        self.requests_per_minute = requests_per_minute
        self.min_interval = 60 / requests_per_minute
        self.last_request = None

    def wait_if_needed(self):
        if self.last_request:
            elapsed = time.time() - self.last_request
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)
        self.last_request = time.time()

# Usage
limiter = RateLimiter(requests_per_minute=30)
for query in queries:
    limiter.wait_if_needed()
    response = requests.post(..., json={"query": query})
```

---

## Integration Examples

### Django Integration

```python
# django_app/search/client.py
import requests
from django.conf import settings

class PrivaChatClient:
    def __init__(self):
        self.api_url = settings.PRIVACHAT_API_URL
        self.timeout = settings.PRIVACHAT_TIMEOUT

    def search(self, query: str):
        response = requests.post(
            f"{self.api_url}/api/v1/search",
            json={"query": query},
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()

# Usage in view
from django.http import JsonResponse
from .client import PrivaChatClient

def search_view(request):
    query = request.GET.get("q")
    client = PrivaChatClient()
    result = client.search(query)
    return JsonResponse(result)
```

### FastAPI Integration

```python
# fastapi_app/clients/privachat.py
import httpx
from fastapi import FastAPI
from typing import Optional

app = FastAPI()

async def search_privachat(query: str, mode: str = "balanced") -> dict:
    """Search using PrivaChat API."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8001/api/v1/search",
            json={"query": query, "mode": mode},
            timeout=60.0
        )
        response.raise_for_status()
        return response.json()

@app.post("/search")
async def search(query: str, mode: Optional[str] = "balanced"):
    return await search_privachat(query, mode)
```

---

## Debugging & Support

### Enable Tracing

All responses include a `trace_url` for debugging:

```python
result = requests.post(...).json()
print(result.get("trace_url"))  # https://langfuse.com/trace/xyz
```

Open the trace URL to see:
- Full request/response details
- Execution timeline
- Token usage
- Errors and warnings

### Check Health

```bash
curl http://localhost:8001/api/v1/health
```

### View API Spec

- **Interactive**: http://localhost:8001/api/docs
- **Raw JSON**: http://localhost:8001/api/openapi.json

---

## FAQ

**Q: How do I use the API from a different machine?**
A: Replace `localhost` with your server's IP/hostname in all URLs.

**Q: What's the difference between /search and /research?**
A: `/search` is fast (3-5s), `/research` is deep and iterative (30-60s).

**Q: Can I use my own LLM model?**
A: Yes, pass the `model` parameter: `{"query": "...", "model": "your-model-name"}`

**Q: How long are results cached?**
A: Use `session_id` to retrieve any previous result indefinitely.

**Q: What's the rate limit?**
A: No hard limit currently. For production, implement client-side limiting.

**Q: How do I report bugs?**
A: Open an issue: https://github.com/chaitanyame/privachat_agents/issues

---

## Next Steps

1. **Read the full spec**: [API_SPECIFICATION.md](./API_SPECIFICATION.md)
2. **Explore interactive docs**: http://localhost:8001/api/docs
3. **Try examples**: See `examples/` directory
4. **Build integration**: Use the patterns above
5. **Ask questions**: GitHub Discussions

---

**Happy coding! 🚀**
