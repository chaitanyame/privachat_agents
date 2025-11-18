"""
PrivaChat Agents Python SDK

A simple, easy-to-use Python client for the PrivaChat Agents API.

Usage:
    from privachat_sdk import PrivaChatClient

    client = PrivaChatClient()
    result = client.search("What is AI?")
    print(result.answer)
"""

from __future__ import annotations

import httpx
import asyncio
from typing import Optional, Literal, Any
from dataclasses import dataclass
from datetime import datetime


@dataclass
class Source:
    """Represents a search source."""
    title: str
    url: str
    snippet: str
    relevance: float
    semantic_score: Optional[float] = None
    final_score: Optional[float] = None
    source_type: str = "web"


@dataclass
class SearchResult:
    """Represents a search result."""
    session_id: str
    query: str
    answer: str
    sources: list[Source]
    execution_time: float
    confidence: float
    mode: str = "balanced"
    model_used: Optional[str] = None
    trace_url: Optional[str] = None
    grounding_score: Optional[float] = None
    hallucination_count: Optional[int] = None
    created_at: Optional[str] = None

    def __str__(self) -> str:
        """Pretty print the result."""
        lines = [
            f"Query: {self.query}",
            f"\nAnswer:\n{self.answer}",
            f"\nSources ({len(self.sources)}):",
        ]
        for i, source in enumerate(self.sources, 1):
            lines.append(f"  [{i}] {source.title}")
            lines.append(f"      {source.url}")
            lines.append(f"      Relevance: {source.relevance:.1%}")
        lines.extend([
            f"\nMetrics:",
            f"  Execution time: {self.execution_time:.2f}s",
            f"  Confidence: {self.confidence:.1%}",
            f"  Mode: {self.mode}",
        ])
        if self.trace_url:
            lines.append(f"  Trace: {self.trace_url}")
        return "\n".join(lines)


@dataclass
class ResearchResult:
    """Represents a research result."""
    session_id: str
    query: str
    findings: str
    citations: list[dict]
    execution_time: float
    confidence: float
    model_used: Optional[str] = None
    trace_url: Optional[str] = None

    def __str__(self) -> str:
        """Pretty print the result."""
        lines = [
            f"Query: {self.query}",
            f"\nFindings:\n{self.findings}",
            f"\nCitations ({len(self.citations)}):",
        ]
        for i, cite in enumerate(self.citations, 1):
            lines.append(f"  [{i}] {cite.get('title', 'N/A')}")
            lines.append(f"      {cite.get('url', 'N/A')}")
        lines.extend([
            f"\nMetrics:",
            f"  Execution time: {self.execution_time:.2f}s",
            f"  Confidence: {self.confidence:.1%}",
        ])
        return "\n".join(lines)


class PrivaChatClient:
    """Client for PrivaChat Agents API (synchronous)."""

    def __init__(
        self,
        base_url: str = "http://localhost:8001",
        timeout: int = 60,
        verify_ssl: bool = True,
    ):
        """
        Initialize the client.

        Args:
            base_url: API base URL
            timeout: Request timeout in seconds
            verify_ssl: Whether to verify SSL certificates
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.verify_ssl = verify_ssl

    def search(
        self,
        query: str,
        mode: Literal["speed", "balanced", "deep"] = "balanced",
        max_sources: Optional[int] = None,
        timeout: Optional[int] = None,
        model: Optional[str] = None,
        search_engine: Literal["searxng", "serperdev", "perplexity", "auto"] = "auto",
    ) -> SearchResult:
        """
        Perform a web search.

        Args:
            query: Search query
            mode: Search mode (speed/balanced/deep)
            max_sources: Maximum sources to retrieve
            timeout: Request timeout in seconds
            model: LLM model to use
            search_engine: Search engine backend

        Returns:
            SearchResult with answer and sources

        Raises:
            httpx.HTTPError: If API request fails
            ValueError: If response is invalid
        """
        with httpx.Client(
            verify=self.verify_ssl,
            timeout=timeout or self.timeout,
        ) as client:
            payload = {
                "query": query,
                "mode": mode,
                "search_engine": search_engine,
            }
            if max_sources:
                payload["max_sources"] = max_sources
            if model:
                payload["model"] = model

            response = client.post(
                f"{self.base_url}/api/v1/search",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

            # Convert to SearchResult
            return SearchResult(
                session_id=data["session_id"],
                query=data["query"],
                answer=data["answer"],
                sources=[Source(**src) for src in data["sources"]],
                execution_time=data["execution_time"],
                confidence=data["confidence"],
                mode=data.get("mode", "balanced"),
                model_used=data.get("model_used"),
                trace_url=data.get("trace_url"),
                grounding_score=data.get("grounding_score"),
                hallucination_count=data.get("hallucination_count"),
                created_at=data.get("created_at"),
            )

    def research(
        self,
        query: str,
        mode: Literal["speed", "balanced", "deep"] = "deep",
        max_iterations: int = 3,
        timeout: Optional[int] = None,
        model: Optional[str] = None,
    ) -> ResearchResult:
        """
        Perform deep research.

        Args:
            query: Research query
            mode: Search mode for underlying searches
            max_iterations: Maximum research iterations
            timeout: Request timeout in seconds
            model: LLM model to use

        Returns:
            ResearchResult with findings and citations

        Raises:
            httpx.HTTPError: If API request fails
        """
        with httpx.Client(
            verify=self.verify_ssl,
            timeout=timeout or 600,  # 10 minutes for research
        ) as client:
            payload = {
                "query": query,
                "mode": mode,
                "max_iterations": max_iterations,
            }
            if model:
                payload["model"] = model

            response = client.post(
                f"{self.base_url}/api/v1/research",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

            return ResearchResult(
                session_id=data["session_id"],
                query=data["query"],
                findings=data["findings"],
                citations=data.get("citations", []),
                execution_time=data["execution_time"],
                confidence=data["confidence"],
                model_used=data.get("model_used"),
                trace_url=data.get("trace_url"),
            )

    def get_session(self, session_id: str) -> dict:
        """
        Retrieve a previous session.

        Args:
            session_id: Session UUID

        Returns:
            Session data with results

        Raises:
            httpx.HTTPError: If session not found or API fails
        """
        with httpx.Client(verify=self.verify_ssl, timeout=self.timeout) as client:
            response = client.get(f"{self.base_url}/api/v1/sessions/{session_id}")
            response.raise_for_status()
            return response.json()

    def upload_document(
        self,
        file_path: str,
        collection_id: Optional[str] = None,
    ) -> dict:
        """
        Upload a document for RAG.

        Args:
            file_path: Path to document file
            collection_id: Optional collection ID

        Returns:
            Upload response with document_id

        Raises:
            FileNotFoundError: If file doesn't exist
            httpx.HTTPError: If upload fails
        """
        from pathlib import Path

        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")

        with httpx.Client(verify=self.verify_ssl, timeout=self.timeout) as client:
            with open(file_path, "rb") as f:
                files = {"file": f}
                data = {}
                if collection_id:
                    data["collection_id"] = collection_id

                response = client.post(
                    f"{self.base_url}/api/v1/documents",
                    files=files,
                    data=data,
                )
                response.raise_for_status()
                return response.json()

    def query_documents(
        self,
        query: str,
        document_ids: list[str],
        max_sources: int = 5,
    ) -> dict:
        """
        Query uploaded documents.

        Args:
            query: Query text
            document_ids: List of document UUIDs
            max_sources: Maximum sources to return

        Returns:
            Query response with answer

        Raises:
            httpx.HTTPError: If query fails
        """
        with httpx.Client(verify=self.verify_ssl, timeout=self.timeout) as client:
            response = client.post(
                f"{self.base_url}/api/v1/documents/query",
                json={
                    "query": query,
                    "document_ids": document_ids,
                    "max_sources": max_sources,
                },
            )
            response.raise_for_status()
            return response.json()

    def health_check(self) -> dict:
        """
        Check API health.

        Returns:
            Health status

        Raises:
            httpx.HTTPError: If API is down
        """
        with httpx.Client(verify=self.verify_ssl, timeout=self.timeout) as client:
            response = client.get(f"{self.base_url}/api/v1/health")
            response.raise_for_status()
            return response.json()


class AsyncPrivaChatClient:
    """Client for PrivaChat Agents API (asynchronous)."""

    def __init__(
        self,
        base_url: str = "http://localhost:8001",
        timeout: int = 60,
        verify_ssl: bool = True,
    ):
        """
        Initialize the async client.

        Args:
            base_url: API base URL
            timeout: Request timeout in seconds
            verify_ssl: Whether to verify SSL certificates
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.verify_ssl = verify_ssl

    async def search(
        self,
        query: str,
        mode: Literal["speed", "balanced", "deep"] = "balanced",
        max_sources: Optional[int] = None,
        timeout: Optional[int] = None,
        model: Optional[str] = None,
        search_engine: Literal["searxng", "serperdev", "perplexity", "auto"] = "auto",
    ) -> SearchResult:
        """Perform a web search asynchronously."""
        async with httpx.AsyncClient(
            verify=self.verify_ssl,
            timeout=timeout or self.timeout,
        ) as client:
            payload = {
                "query": query,
                "mode": mode,
                "search_engine": search_engine,
            }
            if max_sources:
                payload["max_sources"] = max_sources
            if model:
                payload["model"] = model

            response = await client.post(
                f"{self.base_url}/api/v1/search",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

            return SearchResult(
                session_id=data["session_id"],
                query=data["query"],
                answer=data["answer"],
                sources=[Source(**src) for src in data["sources"]],
                execution_time=data["execution_time"],
                confidence=data["confidence"],
                mode=data.get("mode", "balanced"),
                model_used=data.get("model_used"),
                trace_url=data.get("trace_url"),
                grounding_score=data.get("grounding_score"),
                hallucination_count=data.get("hallucination_count"),
                created_at=data.get("created_at"),
            )

    async def research(
        self,
        query: str,
        mode: Literal["speed", "balanced", "deep"] = "deep",
        max_iterations: int = 3,
        timeout: Optional[int] = None,
        model: Optional[str] = None,
    ) -> ResearchResult:
        """Perform deep research asynchronously."""
        async with httpx.AsyncClient(
            verify=self.verify_ssl,
            timeout=timeout or 600,
        ) as client:
            payload = {
                "query": query,
                "mode": mode,
                "max_iterations": max_iterations,
            }
            if model:
                payload["model"] = model

            response = await client.post(
                f"{self.base_url}/api/v1/research",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

            return ResearchResult(
                session_id=data["session_id"],
                query=data["query"],
                findings=data["findings"],
                citations=data.get("citations", []),
                execution_time=data["execution_time"],
                confidence=data["confidence"],
                model_used=data.get("model_used"),
                trace_url=data.get("trace_url"),
            )

    async def health_check(self) -> dict:
        """Check API health asynchronously."""
        async with httpx.AsyncClient(verify=self.verify_ssl, timeout=self.timeout) as client:
            response = await client.get(f"{self.base_url}/api/v1/health")
            response.raise_for_status()
            return response.json()


# Example usage
if __name__ == "__main__":
    # Synchronous
    client = PrivaChatClient()
    result = client.search("What is Pydantic AI?")
    print(result)

    # Asynchronous
    async def async_example():
        async_client = AsyncPrivaChatClient()
        result = await async_client.search("What is Pydantic AI?")
        print(result)

    # asyncio.run(async_example())
