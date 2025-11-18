# PrivaChat Agents API Specification

**Version**: 0.2.0
**Last Updated**: November 2025
**Status**: Production Ready

---

## Table of Contents

1. [Base URL & Access](#base-url--access)
2. [Authentication](#authentication)
3. [Search Endpoints](#search-endpoints)
4. [Research Endpoints](#research-endpoints)
5. [Document Management](#document-management)
6. [Sessions](#sessions)
7. [Error Handling](#error-handling)
8. [API Documentation](#api-documentation)
9. [Examples](#examples)

---

## Base URL & Access

```
http://localhost:8001
```

### Interactive Documentation

- **Swagger UI**: http://localhost:8001/api/docs
- **ReDoc**: http://localhost:8001/api/redoc
- **OpenAPI JSON**: http://localhost:8001/api/openapi.json

---

## Authentication

The API is currently **open** (no authentication required). For production deployments, implement:
- API key authentication
- OAuth 2.0
- JWT tokens

See [SECURITY.md](../SECURITY.md) for security guidelines.

---

## Search Endpoints

### 1. POST /api/v1/search - Fast Search

Execute a fast search with query decomposition and parallel source retrieval.

**Summary**: Perform fast web search with AI-powered answer synthesis

**Tags**: `search`

#### Request Body

```json
{
  "query": "What is Pydantic AI?",
  "mode": "balanced",
  "max_sources": 20,
  "timeout": 60,
  "model": null,
  "search_engine": "auto",
  "prompt_strategy": "auto",
  "enable_diversity": true,
  "enable_recency": false,
  "enable_query_aware": false
}
```

#### Request Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `query` | string | ✅ | - | Search query (1-5000 characters) |
| `mode` | enum | No | `balanced` | Search mode: `speed` (fast, 5-10 sources), `balanced` (default, 10-15 sources), `deep` (comprehensive, 15-20 sources) |
| `max_sources` | integer | No | None | Maximum sources (5-50). Overrides mode default if provided |
| `timeout` | integer | No | None | Timeout in seconds (10-300). Overrides mode default if provided |
| `model` | string | No | None | LLM model to use (e.g., `google/gemini-2.0-flash-lite-001`). Uses default if not provided |
| `search_engine` | enum | No | `auto` | Search backend: `searxng` (open-source), `serperdev` (Google API), `perplexity` (AI-powered), `auto` (tries SearXNG first, fallback to SerperDev) |
| `prompt_strategy` | enum | No | `auto` | System prompt strategy: `static` (fixed prompts), `dynamic` (query-aware), `auto` (uses config) |
| `enable_diversity` | boolean | No | `true` | Enable diversity penalty to reduce duplicate results |
| `enable_recency` | boolean | No | `false` | Enable recency boost for temporal queries (experimental) |
| `enable_query_aware` | boolean | No | `false` | Enable query-aware score adaptations (experimental) |

#### Response (200 OK)

```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "query": "What is Pydantic AI?",
  "answer": "Pydantic AI is an LLM framework that makes building AI agents simple and intuitive...",
  "sub_queries": [
    {
      "query": "What is Pydantic AI used for?",
      "intent": "definition",
      "priority": 1
    }
  ],
  "sources": [
    {
      "title": "Pydantic AI Documentation",
      "url": "https://ai.pydantic.dev/",
      "snippet": "Pydantic AI is an LLM framework...",
      "relevance": 0.95,
      "semantic_score": 0.92,
      "final_score": 0.935,
      "source_type": "web"
    }
  ],
  "mode": "balanced",
  "execution_time": 3.45,
  "confidence": 0.92,
  "model_used": "google/gemini-2.0-flash-lite-001",
  "trace_url": "https://langfuse.com/trace/...",
  "grounding_score": 0.98,
  "hallucination_count": 0,
  "created_at": "2025-11-16T10:30:00Z"
}
```

#### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `session_id` | UUID | Unique session identifier for tracking |
| `query` | string | Original search query |
| `answer` | string | AI-generated answer with inline citations |
| `sub_queries` | array | Query decomposition results |
| `sources` | array | Retrieved and ranked sources |
| `mode` | string | Search mode used |
| `execution_time` | number | Total execution time in seconds |
| `confidence` | number | Confidence score (0-1, higher is better) |
| `model_used` | string | LLM model used for answer generation |
| `trace_url` | string | Langfuse trace URL for debugging (null if tracing disabled) |
| `grounding_score` | number | Hallucination detection score (0-1, higher is better) |
| `hallucination_count` | integer | Number of unsupported claims detected |
| `created_at` | string | ISO 8601 timestamp |

#### Possible Responses

- **200 OK**: Search completed successfully
- **422 Unprocessable Entity**: Invalid request parameters (e.g., empty query)
- **504 Gateway Timeout**: Search exceeded timeout limit
- **500 Internal Server Error**: Search execution failed

#### Example: cURL

```bash
curl -X POST http://localhost:8001/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is Pydantic AI?",
    "mode": "balanced"
  }'
```

#### Example: Python (requests)

```python
import requests

response = requests.post(
    "http://localhost:8001/api/v1/search",
    json={
        "query": "What is Pydantic AI?",
        "mode": "balanced"
    }
)

data = response.json()
print(data["answer"])
print(f"Sources: {len(data['sources'])}")
print(f"Confidence: {data['confidence']:.2%}")
```

#### Example: JavaScript (fetch)

```javascript
const response = await fetch("http://localhost:8001/api/v1/search", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    query: "What is Pydantic AI?",
    mode: "balanced"
  })
});

const data = await response.json();
console.log(data.answer);
console.log(`Sources: ${data.sources.length}`);
console.log(`Confidence: ${(data.confidence * 100).toFixed(1)}%`);
```

---

### 2. POST /api/v1/search/perplexity - Direct Perplexity Search

Execute search using Perplexity AI API directly (bypasses SearchAgent).

**Summary**: Perform direct search using Perplexity AI (returns ready-to-use answer with citations)

**Tags**: `search`

#### Request Body

Same as `/api/v1/search` (see above)

#### Response (200 OK)

Same structure as `/api/v1/search` but with:
- `mode`: `"perplexity"`
- `sources`: Generated from Perplexity citations
- `sub_queries`: Empty (Perplexity handles decomposition internally)
- `confidence`: Always `0.95` (Perplexity curated results)

#### Example: cURL

```bash
curl -X POST http://localhost:8001/api/v1/search/perplexity \
  -H "Content-Type: application/json" \
  -d '{"query": "Latest AI breakthroughs in 2025"}'
```

---

## Research Endpoints

### 1. POST /api/v1/research - Deep Research

Execute iterative research with planning and source synthesis.

**Summary**: Perform deep research with multi-step research plan and comprehensive findings

**Tags**: `research`

#### Request Body

```json
{
  "query": "Impact of AI on software development in 2025",
  "mode": "deep",
  "max_iterations": 3,
  "timeout": 300,
  "model": null,
  "prompt_strategy": "auto"
}
```

#### Request Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `query` | string | ✅ | - | Research query (1-5000 characters) |
| `mode` | enum | No | `deep` | Search mode for underlying searches |
| `max_iterations` | integer | No | `3` | Maximum research iterations (1-10) |
| `timeout` | integer | No | `300` | Timeout in seconds |
| `model` | string | No | None | LLM model override |
| `prompt_strategy` | enum | No | `auto` | System prompt strategy |

#### Response (200 OK)

```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440001",
  "query": "Impact of AI on software development in 2025",
  "plan": {
    "original_query": "Impact of AI on software development in 2025",
    "steps": [
      {
        "step_number": 1,
        "description": "Identify current AI tools in software development",
        "search_query": "AI tools software development 2025",
        "expected_outcome": "Overview of popular AI development tools",
        "depends_on": []
      }
    ],
    "estimated_time": 45,
    "complexity": "moderate"
  },
  "findings": "After comprehensive research, AI is fundamentally transforming software development...",
  "citations": [
    {
      "source_id": "1",
      "title": "The State of AI in Software Development 2025",
      "url": "https://example.com/ai-software-dev",
      "excerpt": "AI tools are now mainstream in software development...",
      "relevance": 0.98
    }
  ],
  "execution_time": 45.5,
  "execution_steps": [],
  "confidence": 0.94,
  "model_used": "google/gemini-2.0-flash-lite-001",
  "trace_url": "https://langfuse.com/trace/..."
}
```

#### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `session_id` | UUID | Unique session identifier |
| `query` | string | Original research query |
| `plan` | object | Research plan with steps |
| `findings` | string | Comprehensive research findings |
| `citations` | array | Sources used in research |
| `execution_time` | number | Total execution time in seconds |
| `confidence` | number | Research confidence score (0-1) |
| `model_used` | string | LLM model used |
| `trace_url` | string | Langfuse trace URL (optional) |

#### Possible Responses

- **200 OK**: Research completed successfully
- **422 Unprocessable Entity**: Invalid request parameters
- **504 Gateway Timeout**: Research exceeded timeout limit
- **500 Internal Server Error**: Research execution failed

#### Example: cURL

```bash
curl -X POST http://localhost:8001/api/v1/research \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Impact of AI on software development",
    "max_iterations": 3
  }'
```

---

## Document Management

### 1. POST /api/v1/documents - Upload Document

Upload a document (PDF, Word, Excel, etc.) for RAG-based queries.

**Tags**: `documents`

#### Request

- **Content-Type**: `multipart/form-data`
- **Body**:
  - `file` (file, required): Document file to upload
  - `collection_id` (string, optional): Collection to store document in

#### Response (200 OK)

```json
{
  "document_id": "550e8400-e29b-41d4-a716-446655440002",
  "filename": "technical-paper.pdf",
  "file_size": 2048576,
  "pages": 45,
  "chunks": 150,
  "status": "indexed",
  "indexed_at": "2025-11-16T10:30:00Z"
}
```

#### Example: cURL

```bash
curl -X POST http://localhost:8001/api/v1/documents \
  -F "file=@document.pdf"
```

#### Example: Python

```python
import requests

with open("document.pdf", "rb") as f:
    files = {"file": f}
    response = requests.post(
        "http://localhost:8001/api/v1/documents",
        files=files
    )

data = response.json()
print(f"Document ID: {data['document_id']}")
print(f"Chunks created: {data['chunks']}")
```

---

### 2. GET /api/v1/documents - List Documents

List all uploaded documents.

**Tags**: `documents`

#### Response (200 OK)

```json
{
  "documents": [
    {
      "document_id": "550e8400-e29b-41d4-a716-446655440002",
      "filename": "technical-paper.pdf",
      "file_size": 2048576,
      "pages": 45,
      "chunks": 150,
      "uploaded_at": "2025-11-16T10:30:00Z"
    }
  ],
  "total": 1
}
```

#### Example: cURL

```bash
curl http://localhost:8001/api/v1/documents
```

---

### 3. POST /api/v1/documents/query - Query Documents

Query uploaded documents using RAG (Retrieval-Augmented Generation).

**Tags**: `documents`

#### Request Body

```json
{
  "query": "What are the main conclusions?",
  "document_ids": ["550e8400-e29b-41d4-a716-446655440002"],
  "max_sources": 5
}
```

#### Response (200 OK)

```json
{
  "query": "What are the main conclusions?",
  "answer": "The research concludes that...",
  "sources": [
    {
      "document_id": "550e8400-e29b-41d4-a716-446655440002",
      "filename": "technical-paper.pdf",
      "page": 42,
      "excerpt": "In conclusion, our findings show..."
    }
  ]
}
```

#### Example: cURL

```bash
curl -X POST http://localhost:8001/api/v1/documents/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the main conclusions?",
    "document_ids": ["550e8400-e29b-41d4-a716-446655440002"]
  }'
```

---

### 4. DELETE /api/v1/documents/{document_id} - Delete Document

Delete a document and its indexed chunks.

**Tags**: `documents`

#### Path Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `document_id` | UUID | Document ID to delete |

#### Response (200 OK)

```json
{
  "success": true,
  "message": "Document deleted successfully"
}
```

#### Example: cURL

```bash
curl -X DELETE http://localhost:8001/api/v1/documents/550e8400-e29b-41d4-a716-446655440002
```

---

## Sessions

### GET /api/v1/sessions/{session_id} - Get Session

Retrieve a previous search or research session by ID.

**Tags**: `sessions`

#### Path Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `session_id` | UUID | Session ID to retrieve |

#### Response (200 OK)

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "query": "What is Pydantic AI?",
  "mode": "search",
  "status": "completed",
  "result": {
    "answer": "...",
    "sources": [...]
  },
  "created_at": "2025-11-16T10:30:00Z",
  "completed_at": "2025-11-16T10:33:00Z"
}
```

#### Example: cURL

```bash
curl http://localhost:8001/api/v1/sessions/550e8400-e29b-41d4-a716-446655440000
```

#### Possible Responses

- **200 OK**: Session found
- **404 Not Found**: Session does not exist

---

## System Endpoints

### GET /api/v1/health - Health Check

Check if the API is running and healthy.

**Tags**: `health`

#### Response (200 OK)

```json
{
  "status": "healthy",
  "service": "research-service",
  "version": "0.2.0",
  "environment": "development"
}
```

#### Example: cURL

```bash
curl http://localhost:8001/api/v1/health
```

---

### GET / - Root Endpoint

Get API information and available endpoints.

#### Response (200 OK)

```json
{
  "service": "Research Service API",
  "version": "0.2.0",
  "docs": "/api/docs",
  "health": "/api/v1/health",
  "status": "ready"
}
```

---

## Error Handling

### Common Error Responses

#### 422 Unprocessable Entity - Validation Error

```json
{
  "detail": [
    {
      "type": "string_too_short",
      "loc": ["body", "query"],
      "msg": "String should have at least 1 character",
      "input": "",
      "ctx": {"min_length": 1}
    }
  ]
}
```

**Causes**:
- Missing required fields
- Invalid field types
- Out-of-range values
- Invalid enum values

#### 504 Gateway Timeout

```json
{
  "error": "search_timeout",
  "message": "Search exceeded timeout of 60s",
  "trace_id": "trace-123456"
}
```

**Causes**:
- Search/research took longer than timeout
- Network issues with search backends
- Slow LLM response

#### 500 Internal Server Error

```json
{
  "error": "search_execution_failed",
  "message": "Search execution failed: ...",
  "trace_id": "trace-123456",
  "details": "Full error traceback (DEBUG mode only)"
}
```

**Causes**:
- API key issues
- Database connectivity
- Search backend failures
- LLM service unavailable

#### 404 Not Found

```json
{
  "error": "session_not_found",
  "message": "Session 550e8400-e29b-41d4-a716-446655440000 not found"
}
```

---

## API Documentation

### Interactive Documentation

Access interactive API documentation at these endpoints:

- **Swagger UI** (OpenAPI 3.0): http://localhost:8001/api/docs
  - Interactive request/response testing
  - Schema validation
  - Model examples

- **ReDoc** (ReadTheDocs style): http://localhost:8001/api/redoc
  - Clean, readable documentation
  - Best for reading

- **OpenAPI JSON**: http://localhost:8001/api/openapi.json
  - Raw OpenAPI 3.1.0 specification
  - For code generation tools

### API Clients & SDKs

Generate API clients from OpenAPI spec:

**OpenAPI Generator**:
```bash
openapi-generator-cli generate -i http://localhost:8001/api/openapi.json \
  -g python \
  -o ./client
```

**FastAPI Clients**:
```bash
pip install httpx
```

---

## Examples

### Example 1: Quick Search

```python
import httpx

async with httpx.AsyncClient() as client:
    response = await client.post(
        "http://localhost:8001/api/v1/search",
        json={"query": "Latest Python features"}
    )
    result = response.json()
    print(result["answer"])
```

### Example 2: Deep Research

```python
import httpx

async with httpx.AsyncClient() as client:
    response = await client.post(
        "http://localhost:8001/api/v1/research",
        json={
            "query": "State of AI in 2025",
            "max_iterations": 5
        }
    )
    result = response.json()
    print(f"Findings: {result['findings']}")
    print(f"Citations: {len(result['citations'])}")
```

### Example 3: Document Upload & Query

```python
import httpx

async with httpx.AsyncClient() as client:
    # Upload
    with open("paper.pdf", "rb") as f:
        files = {"file": f}
        upload_resp = await client.post(
            "http://localhost:8001/api/v1/documents",
            files=files
        )
    doc_id = upload_resp.json()["document_id"]

    # Query
    query_resp = await client.post(
        "http://localhost:8001/api/v1/documents/query",
        json={
            "query": "What is the conclusion?",
            "document_ids": [doc_id]
        }
    )
    print(query_resp.json()["answer"])
```

---

## Rate Limiting

No rate limiting is currently enforced. For production, implement:
- Per-IP rate limits (100 requests/minute)
- Per-API-key rate limits
- Per-endpoint rate limits

---

## Support & Issues

- **Documentation**: [API Guide](./API_CONSUMPTION.md)
- **Issues**: https://github.com/chaitanyame/privachat_agents/issues
- **Discussions**: https://github.com/chaitanyame/privachat_agents/discussions
