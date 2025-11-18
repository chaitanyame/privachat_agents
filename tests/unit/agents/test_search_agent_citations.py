"""Unit tests for SearchAgent citation mapping - RED phase (tests written FIRST).

Test Coverage:
- Citation extraction from answer text
- Citation-to-source mapping (1-based to 0-based conversion)
- Mention count tracking
- Simplified domain extraction
- Full URL inclusion
- Out-of-bounds citation handling
- Edge cases (no citations, invalid URLs, etc.)

Following TDD: These tests will FAIL until implementation is complete.
"""

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from privachat_agents.agents.search_agent import SearchAgent, SearchAgentDeps, SearchSource
from privachat_agents.services.llm.langfuse_tracer import LangfuseTracer
from privachat_agents.services.llm.openrouter_client import OpenRouterClient


@pytest.fixture
def mock_llm_client() -> OpenRouterClient:
    """Mock OpenRouter LLM client."""
    client = MagicMock(spec=OpenRouterClient)
    client.chat = AsyncMock()
    return client


@pytest.fixture
def mock_tracer() -> LangfuseTracer:
    """Mock Langfuse tracer."""
    tracer = MagicMock(spec=LangfuseTracer)
    tracer.trace_llm_call = AsyncMock()
    return tracer


@pytest.fixture
def mock_searxng_client() -> httpx.AsyncClient:
    """Mock SearxNG HTTP client."""
    client = AsyncMock(spec=httpx.AsyncClient)
    client.get = AsyncMock()
    return client


@pytest.fixture
def search_agent(
    mock_llm_client: OpenRouterClient,
    mock_tracer: LangfuseTracer,
    mock_searxng_client: httpx.AsyncClient,
) -> SearchAgent:
    """Create SearchAgent for testing."""
    deps = SearchAgentDeps(
        llm_client=mock_llm_client,
        tracer=mock_tracer,
        db=MagicMock(spec=AsyncSession),
        searxng_client=mock_searxng_client,
        serperdev_api_key="test_key",
        max_sources=20,
        timeout=60.0,
    )
    return SearchAgent(deps=deps)


class TestCitationExtraction:
    """Test citation marker extraction from answer text."""

    def test_extract_simple_citations(self, search_agent: SearchAgent) -> None:
        """Test extraction of simple citation markers.

        Given: An answer with citation markers [1], [2]
        When: Extracting citations
        Then: Returns correct citation numbers
        """
        answer = "AI agents are systems [1]. They use reasoning [2]."
        sources = [
            SearchSource(
                title="Source 1",
                url="https://example.com/1",
                snippet="Content 1",
                relevance=0.9,
                semantic_score=0.8,
                final_score=0.85,
                source_type="web",
            ),
            SearchSource(
                title="Source 2",
                url="https://arxiv.org/paper",
                snippet="Content 2",
                relevance=0.8,
                semantic_score=0.7,
                final_score=0.75,
                source_type="academic",
            ),
        ]

        
        citation_mapping = search_agent._extract_citation_mapping(answer, sources)

        assert len(citation_mapping) == 2
        assert citation_mapping[0]["citation_number"] == 1
        assert citation_mapping[1]["citation_number"] == 2

    def test_extract_multiple_mentions(self, search_agent: SearchAgent) -> None:
        """Test counting multiple mentions of same citation.

        Given: An answer with [1] appearing multiple times
        When: Extracting citations
        Then: Correctly counts mention_count
        """
        answer = "AI agents [1] are systems [1]. They work [2]. More on agents [1]."
        sources = [
            SearchSource(
                title="Source 1",
                url="https://openai.com/research",
                snippet="Content",
                relevance=0.9,
                semantic_score=0.8,
                final_score=0.85,
                source_type="web",
            ),
            SearchSource(
                title="Source 2",
                url="https://example.com/2",
                snippet="Content",
                relevance=0.8,
                semantic_score=0.7,
                final_score=0.75,
                source_type="web",
            ),
        ]

        
        citation_mapping = search_agent._extract_citation_mapping(answer, sources)

        assert len(citation_mapping) == 2
        assert citation_mapping[0]["citation_number"] == 1
        assert citation_mapping[0]["mention_count"] == 3
        assert citation_mapping[1]["citation_number"] == 2
        assert citation_mapping[1]["mention_count"] == 1

    def test_extract_no_citations(self, search_agent: SearchAgent) -> None:
        """Test extraction when answer has no citations.

        Given: An answer with no citation markers
        When: Extracting citations
        Then: Returns empty list
        """
        answer = "AI agents are systems that use reasoning."
        sources = [
            SearchSource(
                title="Source 1",
                url="https://example.com",
                snippet="Content",
                relevance=0.9,
                semantic_score=0.8,
                final_score=0.85,
                source_type="web",
            )
        ]

        
        citation_mapping = search_agent._extract_citation_mapping(answer, sources)

        assert citation_mapping == []


class TestCitationToSourceMapping:
    """Test mapping citation numbers to source array indices."""

    def test_map_citations_to_sources(self, search_agent: SearchAgent) -> None:
        """Test that citation numbers map to correct source indices.

        Given: Citations [1], [2] and 2 sources
        When: Extracting citation mapping
        Then: [1] maps to sources[0], [2] maps to sources[1]
        """
        answer = "First fact [1]. Second fact [2]."
        sources = [
            SearchSource(
                title="Title 1",
                url="https://example.com/1",
                snippet="Content",
                relevance=0.9,
                semantic_score=0.8,
                final_score=0.85,
                source_type="web",
            ),
            SearchSource(
                title="Title 2",
                url="https://example.com/2",
                snippet="Content",
                relevance=0.8,
                semantic_score=0.7,
                final_score=0.75,
                source_type="web",
            ),
        ]

        
        citation_mapping = search_agent._extract_citation_mapping(answer, sources)

        assert citation_mapping[0]["citation_number"] == 1
        assert citation_mapping[0]["source_index"] == 0
        assert citation_mapping[1]["citation_number"] == 2
        assert citation_mapping[1]["source_index"] == 1

    def test_out_of_bounds_citations_excluded(self, search_agent: SearchAgent) -> None:
        """Test that out-of-bounds citations are excluded.

        Given: Citations [1], [2], [99] but only 2 sources
        When: Extracting citation mapping
        Then: Only [1] and [2] are included, [99] is excluded
        """
        answer = "First fact [1]. Second fact [2]. Invalid [99]."
        sources = [
            SearchSource(
                title="Title 1",
                url="https://example.com/1",
                snippet="Content",
                relevance=0.9,
                semantic_score=0.8,
                final_score=0.85,
                source_type="web",
            ),
            SearchSource(
                title="Title 2",
                url="https://example.com/2",
                snippet="Content",
                relevance=0.8,
                semantic_score=0.7,
                final_score=0.75,
                source_type="web",
            ),
        ]

        
        citation_mapping = search_agent._extract_citation_mapping(answer, sources)

        assert len(citation_mapping) == 2
        assert all(c["citation_number"] in [1, 2] for c in citation_mapping)


class TestDomainExtraction:
    """Test simplified domain name extraction from URLs."""

    def test_extract_simple_domain(self, search_agent: SearchAgent) -> None:
        """Test extraction of simple domain name.

        Given: URL https://wikipedia.org/article
        When: Extracting domain
        Then: Returns "wikipedia"
        """
        
        domain = search_agent._extract_simplified_domain("https://wikipedia.org/article")

        assert domain == "wikipedia"

    def test_extract_domain_with_www(self, search_agent: SearchAgent) -> None:
        """Test extraction strips www prefix.

        Given: URL https://www.arxiv.org/paper
        When: Extracting domain
        Then: Returns "arxiv" (www removed)
        """
        
        domain = search_agent._extract_simplified_domain("https://www.arxiv.org/paper")

        assert domain == "arxiv"

    def test_extract_domain_with_subdomain(self, search_agent: SearchAgent) -> None:
        """Test extraction handles subdomains.

        Given: URL https://en.wikipedia.org/wiki/AI
        When: Extracting domain
        Then: Returns "wikipedia" (primary domain, not "en")
        """
        
        domain = search_agent._extract_simplified_domain("https://en.wikipedia.org/wiki/AI")

        assert domain == "wikipedia"

    def test_extract_domain_github(self, search_agent: SearchAgent) -> None:
        """Test extraction for GitHub URLs.

        Given: URL https://github.com/user/repo
        When: Extracting domain
        Then: Returns "github"
        """
        
        domain = search_agent._extract_simplified_domain("https://github.com/user/repo")

        assert domain == "github"

    def test_extract_domain_docs_subdomain(self, search_agent: SearchAgent) -> None:
        """Test extraction for docs subdomains.

        Given: URL https://docs.python.org/3/
        When: Extracting domain
        Then: Returns "python" (primary domain)
        """
        
        domain = search_agent._extract_simplified_domain("https://docs.python.org/3/")

        assert domain == "python"

    def test_extract_domain_invalid_url(self, search_agent: SearchAgent) -> None:
        """Test extraction handles invalid URLs.

        Given: Invalid URL string
        When: Extracting domain
        Then: Returns "unknown"
        """
        
        domain = search_agent._extract_simplified_domain("not-a-url")

        assert domain == "unknown"

    def test_extract_domain_empty_url(self, search_agent: SearchAgent) -> None:
        """Test extraction handles empty URLs.

        Given: Empty URL string
        When: Extracting domain
        Then: Returns "unknown"
        """
        
        domain = search_agent._extract_simplified_domain("")

        assert domain == "unknown"


class TestFullCitationMapping:
    """Test complete citation mapping with all fields."""

    def test_citation_mapping_complete_fields(self, search_agent: SearchAgent) -> None:
        """Test that citation mapping includes all required fields.

        Given: Answer with citations and sources
        When: Extracting citation mapping
        Then: Each mapping contains all fields (citation_number, source_index,
              mention_count, source_title, source_url, domain)
        """
        answer = "AI agents [1] use reasoning [2]."
        sources = [
            SearchSource(
                title="OpenAI Research",
                url="https://openai.com/research/agents",
                snippet="Content",
                relevance=0.9,
                semantic_score=0.8,
                final_score=0.85,
                source_type="web",
            ),
            SearchSource(
                title="arXiv Paper",
                url="https://arxiv.org/abs/2024.12345",
                snippet="Content",
                relevance=0.8,
                semantic_score=0.7,
                final_score=0.75,
                source_type="academic",
            ),
        ]

        
        citation_mapping = search_agent._extract_citation_mapping(answer, sources)

        # Check first citation
        assert citation_mapping[0]["citation_number"] == 1
        assert citation_mapping[0]["source_index"] == 0
        assert citation_mapping[0]["mention_count"] == 1
        assert citation_mapping[0]["source_title"] == "OpenAI Research"
        assert citation_mapping[0]["source_url"] == "https://openai.com/research/agents"
        assert citation_mapping[0]["domain"] == "openai"

        # Check second citation
        assert citation_mapping[1]["citation_number"] == 2
        assert citation_mapping[1]["source_index"] == 1
        assert citation_mapping[1]["mention_count"] == 1
        assert citation_mapping[1]["source_title"] == "arXiv Paper"
        assert citation_mapping[1]["source_url"] == "https://arxiv.org/abs/2024.12345"
        assert citation_mapping[1]["domain"] == "arxiv"

    def test_citation_mapping_preserves_url(self, search_agent: SearchAgent) -> None:
        """Test that full URLs are preserved in mapping.

        Given: Sources with complex URLs
        When: Extracting citation mapping
        Then: Full URLs are included unchanged
        """
        answer = "Reference [1]."
        sources = [
            SearchSource(
                title="Complex URL",
                url="https://example.com/path/to/resource?param=value&other=123#section",
                snippet="Content",
                relevance=0.9,
                semantic_score=0.8,
                final_score=0.85,
                source_type="web",
            )
        ]

        
        citation_mapping = search_agent._extract_citation_mapping(answer, sources)

        assert (
            citation_mapping[0]["source_url"]
            == "https://example.com/path/to/resource?param=value&other=123#section"
        )


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_sources_list(self, search_agent: SearchAgent) -> None:
        """Test handling of empty sources list.

        Given: Answer with citations but no sources
        When: Extracting citation mapping
        Then: Returns empty list (all citations are out of bounds)
        """
        answer = "Some fact [1]."
        sources = []

        
        citation_mapping = search_agent._extract_citation_mapping(answer, sources)

        assert citation_mapping == []

    def test_citation_zero_excluded(self, search_agent: SearchAgent) -> None:
        """Test that citation [0] is excluded (invalid index).

        Given: Answer with [0] citation marker
        When: Extracting citation mapping
        Then: [0] is excluded (citations start at 1)
        """
        answer = "Invalid citation [0]. Valid [1]."
        sources = [
            SearchSource(
                title="Source",
                url="https://example.com",
                snippet="Content",
                relevance=0.9,
                semantic_score=0.8,
                final_score=0.85,
                source_type="web",
            )
        ]

        
        citation_mapping = search_agent._extract_citation_mapping(answer, sources)

        assert len(citation_mapping) == 1
        assert citation_mapping[0]["citation_number"] == 1

    def test_non_sequential_citations(self, search_agent: SearchAgent) -> None:
        """Test handling of non-sequential citation numbers.

        Given: Answer with [1], [3], [2] (non-sequential)
        When: Extracting citation mapping
        Then: All valid citations are included, sorted by number
        """
        answer = "First [1]. Third [3]. Second [2]."
        sources = [
            SearchSource(title="S1", url="https://a.com", snippet="", relevance=0.9, semantic_score=0.8, final_score=0.85, source_type="web"),
            SearchSource(title="S2", url="https://b.com", snippet="", relevance=0.8, semantic_score=0.7, final_score=0.75, source_type="web"),
            SearchSource(title="S3", url="https://c.com", snippet="", relevance=0.7, semantic_score=0.6, final_score=0.65, source_type="web"),
        ]

        
        citation_mapping = search_agent._extract_citation_mapping(answer, sources)

        assert len(citation_mapping) == 3
        # Should be sorted by citation number
        assert citation_mapping[0]["citation_number"] == 1
        assert citation_mapping[1]["citation_number"] == 2
        assert citation_mapping[2]["citation_number"] == 3
