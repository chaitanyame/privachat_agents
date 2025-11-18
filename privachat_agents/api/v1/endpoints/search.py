"""Search endpoint for fast web search using SearchAgent.

This module provides the /v1/search endpoint that integrates with SearchAgent
to perform query decomposition, parallel search, and result aggregation.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime
from typing import TYPE_CHECKING
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession

from privachat_agents.agents.search_agent import SearchAgent, SearchAgentDeps
from privachat_agents.api.v1.schemas import (
    CitationInfo,
    SearchRequest,
    SearchResponse,
    SearchSourceResponse,
    SubQueryResponse,
)
from privachat_agents.core.config import settings
from privachat_agents.database.models import ResearchSession
from privachat_agents.database.session import get_db
from privachat_agents.services.crawl.crawl4ai_client import Crawl4AIClient
from privachat_agents.services.document.dockling_processor import DocklingProcessor
from privachat_agents.services.embedding.embedding_service import EmbeddingService
from privachat_agents.services.llm.langfuse_tracer import LangfuseTracer
from privachat_agents.services.llm.openrouter_client import OpenRouterClient
from privachat_agents.services.redis_client import RedisClient
from privachat_agents.services.search.searxng_client import SearxNGClient

if TYPE_CHECKING:
    from privachat_agents.agents.search_agent import SearchOutput

router = APIRouter(prefix="/v1", tags=["search"])


async def create_search_agent(
    db: AsyncSession,
    model: str | None = None,
    max_sources: int = 20,
    timeout: float = 60.0,
    min_sources: int = 5,
    min_confidence: float = 0.5,
    enable_diversity: bool = True,  # ENABLED BY DEFAULT for better diversity
    enable_recency: bool = False,
    enable_query_aware: bool = False,
    search_engine: str = "auto",
) -> SearchAgent:
    """Create SearchAgent with dependencies.

    Args:
        db: Database session
        model: Optional LLM model override
        max_sources: Maximum sources to retrieve
        timeout: Search timeout in seconds
        min_sources: Minimum required sources for valid output
        min_confidence: Minimum confidence threshold
        enable_diversity: Enable diversity penalty for deduplication
        enable_recency: Enable recency boost for temporal queries
        enable_query_aware: Enable query-aware score adaptations
        search_engine: Search engine backend ('searxng', 'serperdev', 'auto')

    Returns:
        Configured SearchAgent instance
    """
    # Initialize tracer first (needed by LLM client)
    tracer = LangfuseTracer(
        public_key=settings.LANGFUSE_PUBLIC_KEY,
        secret_key=settings.LANGFUSE_SECRET_KEY,
        host=settings.LANGFUSE_BASE_URL,
    )

    # Initialize LLM client with DeepSeek R1 for search
    llm_client = OpenRouterClient(
        api_key=settings.OPENROUTER_API_KEY,
        model=model or settings.RESEARCH_LLM_MODEL,  # Use DeepSeek R1 for search
        tracer=tracer,
    )

    # Initialize SearxNG client (wrapper) - only if needed
    if search_engine in ["searxng", "auto"]:
        searxng_client = SearxNGClient(
            base_url=settings.SEARXNG_BASE_URL,
            timeout=timeout,
        )
    else:
        # For serperdev-only mode, create a minimal client that will be bypassed
        searxng_client = None  # SearchAgent will handle None gracefully

    # Adjust SerperDev key based on search engine preference
    if search_engine == "searxng":
        # Disable SerperDev fallback when using SearXNG only
        serperdev_key = ""
    elif search_engine == "serperdev":
        # Use SerperDev as primary (SearchAgent will handle this)
        serperdev_key = settings.SERPER_API_KEY or ""
        if not serperdev_key:
            raise HTTPException(
                status_code=400,
                detail="SerperDev API key not configured. Cannot use 'serperdev' mode.",
            )
    else:  # auto
        # Use both: SearXNG primary, SerperDev fallback
        serperdev_key = settings.SERPER_API_KEY or ""

    # Initialize Crawl4AI client for content extraction
    crawl_client = Crawl4AIClient(
        headless=True,
        browser_type="chromium",
        max_concurrent=3,
        timeout=30,
    )

    # Initialize Dockling processor for PDF/document extraction
    document_processor = DocklingProcessor(
        max_file_size=50_000_000,  # 50MB
        max_pages=100,
        chunk_size=1000,
        chunk_overlap=200,
    )

    # Initialize embedding service for semantic reranking
    embedding_service = EmbeddingService(
        model_name="all-MiniLM-L6-v2",
        reranker_model_name=settings.RERANKER_MODEL,
        device="cpu",
    )

    # Create agent dependencies
    deps = SearchAgentDeps(
        llm_client=llm_client,
        tracer=tracer,
        db=db,
        searxng_client=searxng_client,
        serperdev_api_key=serperdev_key,
        crawl_client=crawl_client,
        document_processor=document_processor,
        embedding_service=embedding_service,
        max_sources=max_sources,
        timeout=timeout,
        min_sources=min_sources,
        min_confidence=min_confidence,
        enable_crawling=True,  # Enable URL crawling
        max_crawl_urls=5,  # Crawl top 5 URLs
        enable_reranking=settings.ENABLE_RERANKING,  # Enable semantic reranking
        rerank_weight=settings.RERANK_WEIGHT,  # Weight for semantic score
        # Enhanced reranking features (experimental)
        enable_diversity_penalty=enable_diversity,
        enable_recency_boost=enable_recency,
        enable_query_aware=enable_query_aware,
    )

    return SearchAgent(deps=deps)


async def store_search_session(
    db: AsyncSession,
    session_id: uuid.UUID,
    query: str,
    result: dict,
) -> None:
    """Store search session in database.

    Args:
        db: Database session
        session_id: Session identifier
        query: Search query
        result: Search result data
    """
    session = ResearchSession(
        id=session_id,
        query=query,
        mode="search",
        status="completed",
        result=result,
        created_at=datetime.utcnow(),
        completed_at=datetime.utcnow(),
    )

    db.add(session)
    await db.commit()


async def execute_search_background(
    session_id: uuid.UUID,
    request: SearchRequest,
    redis_client: RedisClient,
) -> None:
    """Execute search in background and store result in Redis.

    Args:
        session_id: Session identifier
        request: Search request parameters
        redis_client: Redis client for job state storage
    """
    # Import here to avoid circular dependencies
    from privachat_agents.database.session import AsyncSessionLocal

    try:
        # Update status to 'processing'
        await redis_client.set_job_status(
            session_id=str(session_id),
            status="processing",
        )

        # Create new DB session for background task
        async with AsyncSessionLocal() as db:
            # Get mode configuration
            from privachat_agents.core.search_modes import get_mode_from_string

            search_mode = get_mode_from_string(request.mode)
            config = search_mode.config

            # Use request parameters or mode defaults
            max_sources = request.max_sources if request.max_sources is not None else config.max_sources
            timeout = request.timeout if request.timeout is not None else config.timeout

            # Create SearchAgent
            agent = await create_search_agent(
                db=db,
                model=request.model,
                max_sources=max_sources,
                timeout=float(timeout),
                enable_diversity=request.enable_diversity,
                enable_recency=request.enable_recency,
                enable_query_aware=request.enable_query_aware,
                search_engine=request.search_engine or "auto",
            )

            # Execute search
            output = await asyncio.wait_for(
                agent.run(request.query, mode=search_mode),
                timeout=float(timeout),
            )

            # Convert to response
            response = convert_search_output_to_response(
                session_id=session_id,
                query=request.query,
                output=output,
                model_used=request.model or settings.LLM_MODEL,
                mode=request.mode or "balanced",
                trace_url=None,
            )

            # Store in Redis with 'completed' status
            await redis_client.set_job_status(
                session_id=str(session_id),
                status="completed",
                result=response.model_dump(mode="json"),
            )

            # Store session in database
            await store_search_session(
                db=db,
                session_id=session_id,
                query=request.query,
                result=response.model_dump(mode="json"),
            )

            # Flush traces
            if agent.deps.tracer:
                agent.deps.tracer.flush()

    except TimeoutError:
        await redis_client.set_job_status(
            session_id=str(session_id),
            status="failed",
            error=f"Search exceeded timeout of {timeout}s",
        )

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        await redis_client.set_job_status(
            session_id=str(session_id),
            status="failed",
            error=f"Search execution failed: {str(e)}\n{error_details}",
        )


def convert_search_output_to_response(
    session_id: uuid.UUID,
    query: str,
    output: SearchOutput,
    model_used: str,
    mode: str = "balanced",
    trace_url: str | None = None,
) -> SearchResponse:
    """Convert SearchOutput to SearchResponse.

    Args:
        session_id: Session identifier
        query: Original query
        output: SearchAgent output
        model_used: LLM model used
        mode: Search mode used (speed/balanced/deep)
        trace_url: Langfuse trace URL

    Returns:
        SearchResponse model
    """
    return SearchResponse(
        session_id=session_id,
        query=query,
        answer=output.answer,
        sub_queries=[
            SubQueryResponse(
                query=sq.query,
                intent=sq.intent,
                priority=sq.priority,
            )
            for sq in output.sub_queries
        ],
        sources=[
            SearchSourceResponse(
                title=src.title,
                url=src.url,
                snippet=src.snippet,
                relevance=src.relevance,
                semantic_score=src.semantic_score,
                final_score=src.final_score,
                source_type=src.source_type,
            )
            for src in output.sources
        ],
        citation_mapping=(
            [CitationInfo(**cm) for cm in output.citation_mapping]
            if output.citation_mapping
            else None
        ),
        mode=mode,
        execution_time=output.execution_time,
        confidence=output.confidence,
        model_used=model_used,
        trace_url=trace_url,
        grounding_score=output.grounding_score,
        hallucination_count=output.hallucination_count,
        created_at=datetime.utcnow(),
    )


@router.post(
    "/search",
    response_model=SearchResponse,
    status_code=status.HTTP_200_OK,
    summary="Execute fast search",
    description="Perform fast web search with query decomposition and parallel execution",
    responses={
        200: {"description": "Search completed successfully"},
        422: {"description": "Validation error"},
        504: {"description": "Search timeout"},
        500: {"description": "Internal server error"},
    },
)
async def search(
    request: SearchRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
) -> SearchResponse:
    """Execute search query using SearchAgent.

    Supports two modes:
    1. Sync mode (default): Returns complete result immediately
    2. Async mode (async_mode=true): Returns session_id immediately, processes in background

    Args:
        request: Search request with query and parameters
        background_tasks: FastAPI background tasks
        db: Database session

    Returns:
        SearchResponse with results (sync) or status='pending' (async)

    Raises:
        HTTPException: On timeout or execution error
    """
    session_id = uuid.uuid4()

    # ASYNC MODE: Return session_id immediately, process in background
    if request.async_mode:
        redis_client = RedisClient()

        # Set initial status to 'pending'
        await redis_client.set_job_status(
            session_id=str(session_id),
            status="pending",
        )

        # Queue background task
        background_tasks.add_task(
            execute_search_background,
            session_id=session_id,
            request=request,
            redis_client=redis_client,
        )

        # Return immediate response with pending status
        return SearchResponse(
            session_id=session_id,
            status="pending",
            query=request.query,
            mode=request.mode or "balanced",
            created_at=datetime.utcnow(),
        )

    # SYNC MODE: Execute search and return complete result
    try:
        # Get mode configuration
        from privachat_agents.core.search_modes import get_mode_from_string

        search_mode = get_mode_from_string(request.mode)
        config = search_mode.config

        # Use request parameters or mode defaults
        max_sources = request.max_sources if request.max_sources is not None else config.max_sources
        timeout = request.timeout if request.timeout is not None else config.timeout

        # Create SearchAgent
        agent = await create_search_agent(
            db=db,
            model=request.model,
            max_sources=max_sources,
            timeout=float(timeout),
            enable_diversity=request.enable_diversity,
            enable_recency=request.enable_recency,
            enable_query_aware=request.enable_query_aware,
            search_engine=request.search_engine or "auto",
        )

        # Execute search with timeout and mode, traced in Langfuse
        trace = None
        try:
            with agent.deps.tracer.trace_context(
                name="search",
                session_id=str(session_id),
                metadata={
                    "mode": request.mode or "balanced",
                    "max_sources": max_sources,
                    "timeout": float(timeout),
                },
            ) as _trace:
                trace = _trace
                output = await asyncio.wait_for(
                    agent.run(request.query, mode=search_mode),
                    timeout=float(timeout),
                )
        except TimeoutError:
            # Ensure any partial traces are flushed and return trace id for debugging
            try:
                agent.deps.tracer.track_error(TimeoutError(f"search timeout {timeout}s"))
                agent.deps.tracer.flush()
            except Exception:
                pass
            return JSONResponse(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                content={
                    "error": "search_timeout",
                    "message": f"Search exceeded timeout of {timeout}s",
                    "trace_id": agent.deps.tracer.get_trace_id(trace)
                    if agent.deps.tracer
                    else None,
                },
            )

        # Convert to response
        response = convert_search_output_to_response(
            session_id=session_id,
            query=request.query,
            output=output,
            model_used=request.model or settings.LLM_MODEL,
            mode=request.mode or "balanced",
            trace_url=(agent.deps.tracer.get_trace_url(trace) if agent.deps.tracer else None),
        )

        # Store session in database
        await store_search_session(
            db=db,
            session_id=session_id,
            query=request.query,
            result=response.model_dump(mode="json"),
        )

        # Flush traces to Langfuse before returning
        if agent.deps.tracer:
            agent.deps.tracer.flush()

        return response

    except HTTPException:
        # Re-raise HTTP exceptions
        raise

    except Exception as e:
        # Handle unexpected errors
        import traceback

        error_details = traceback.format_exc()
        print(f"❌ Search Error: {str(e)}")
        print(f"Traceback:\n{error_details}")

        # Try to flush any pending traces
        try:
            # agent may not exist if failure happened earlier
            if "agent" in locals() and agent.deps.tracer:
                agent.deps.tracer.track_error(e)
                agent.deps.tracer.flush()
        except Exception:
            pass

        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "search_execution_failed",
                "message": f"Search execution failed: {str(e)}",
                "details": error_details if settings.LOG_LEVEL == "DEBUG" else None,
                "trace_id": (
                    agent.deps.tracer.get_trace_id()
                    if "agent" in locals() and agent.deps.tracer
                    else None
                ),
            },
        )


@router.get(
    "/search/status/{session_id}",
    response_model=SearchResponse,
    status_code=status.HTTP_200_OK,
    summary="Get async search job status",
    description="Poll for async search job status and results (24-hour TTL)",
    responses={
        200: {"description": "Job status retrieved successfully"},
        404: {"description": "Job not found (expired or invalid session_id)"},
        500: {"description": "Internal server error"},
    },
)
async def get_search_status(
    session_id: uuid.UUID,
) -> SearchResponse:
    """Get async search job status from Redis.

    Args:
        session_id: Session identifier from async search request

    Returns:
        SearchResponse with status field:
        - 'pending': Job queued but not started
        - 'processing': Job currently executing
        - 'completed': Job finished, includes full result
        - 'failed': Job failed, includes error message

    Raises:
        HTTPException: If job not found or Redis error
    """
    redis_client = RedisClient()

    try:
        job_data = await redis_client.get_job_status(str(session_id))

        if job_data is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job not found: session_id={session_id} (expired or invalid)",
            )

        job_status = job_data.get("status")
        result = job_data.get("result")
        error = job_data.get("error")

        # For 'completed' status, return full result
        if job_status == "completed" and result:
            return SearchResponse(**result)

        # For 'pending'/'processing'/'failed' status, return minimal response
        return SearchResponse(
            session_id=session_id,
            status=job_status,
            query=result.get("query") if result else "Processing...",
            error=error,
            mode=result.get("mode", "balanced") if result else "balanced",
            created_at=datetime.utcnow(),
        )

    except HTTPException:
        raise

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"❌ Status Poll Error: {str(e)}")
        print(f"Traceback:\n{error_details}")

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve job status: {str(e)}",
        )

    finally:
        await redis_client.close()


@router.post(
    "/search/perplexity",
    response_model=SearchResponse,
    status_code=status.HTTP_200_OK,
    summary="Execute Perplexity AI search",
    description="Perform direct search using Perplexity AI (returns ready-to-use answer with citations)",
    responses={
        200: {"description": "Perplexity search completed successfully"},
        422: {"description": "Validation error"},
        500: {"description": "Perplexity API error"},
    },
)
async def search_with_perplexity(
    request: SearchRequest,
    db: AsyncSession = Depends(get_db),
) -> SearchResponse:
    """Execute search query using Perplexity AI directly.

    This endpoint bypasses the SearchAgent and uses Perplexity AI's search API
    directly, which returns a complete answer with citations that requires no
    further processing.

    Args:
        request: Search request with query and parameters
        db: Database session

    Returns:
        SearchResponse with Perplexity's answer and citations

    Raises:
        HTTPException: On API error or execution failure
    """
    from privachat_agents.services.search.perplexity_search import perplexity_search

    session_id = uuid.uuid4()

    try:
        import time

        start_time = time.time()

        # Call Perplexity directly (no model parameter - uses config default)
        result = await perplexity_search(
            query=request.query,
            temperature=0.2,  # Precision-focused
        )

        execution_time = time.time() - start_time

        # Build sources from citations
        sources = []
        for citation in result.get("citations", []):
            sources.append(
                SearchSourceResponse(
                    url=citation["url"],
                    title=citation.get("title", f"Citation [{citation['index']}]"),
                    snippet="",  # Perplexity doesn't provide snippets
                    relevance=0.95,  # High relevance - Perplexity's curated results
                    semantic_score=None,  # No reranking needed
                    final_score=0.95,  # Same as relevance
                    source_type="web",  # Default to web
                )
            )

        response = SearchResponse(
            session_id=session_id,
            query=request.query,
            answer=result["content"],
            sources=sources,
            sub_queries=[],  # Perplexity handles decomposition internally
            execution_time=execution_time,
            model_used=result.get("model", settings.PERPLEXITY_MODEL),
            mode="perplexity",
            confidence=0.95,  # High confidence - Perplexity curated results
            trace_url=None,
        )

        # Store session in database
        await store_search_session(
            db=db,
            session_id=session_id,
            query=request.query,
            result=response.model_dump(mode="json"),
        )

        return response

    except HTTPException:
        raise

    except Exception as e:
        import traceback

        error_details = traceback.format_exc()
        print(f"❌ Perplexity Search Error: {str(e)}")
        print(f"Traceback:\n{error_details}")

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Perplexity search execution failed: {str(e)}",
        ) from e
