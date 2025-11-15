"""Research Service - FastAPI Application Entry Point."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .api.v1.endpoints.documents import router as documents_router
from .api.v1.endpoints.research import router as research_router
from .api.v1.endpoints.search import router as search_router
from .api.v1.endpoints.sessions import router as sessions_router
from .core.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager."""
    # Startup
    print(f"🚀 Research Service starting on port {settings.API_PORT}...")
    print(f"📊 Database: {settings.DATABASE_URL}")
    print(f"💾 Redis: {settings.REDIS_URL}")
    print(f"🤖 LLM Model: {settings.LLM_MODEL}")
    print(f"📝 Log Level: {settings.LOG_LEVEL}")

    yield

    # Shutdown
    print("👋 PrivaChat Agents shutting down...")


# Create FastAPI app
app = FastAPI(
    title="PrivaChat Agents API",
    description="Privacy-focused AI agent system with advanced search and research capabilities",
    version="0.1.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Health check endpoint
@app.get("/api/v1/health", tags=["Health"])
async def health_check() -> JSONResponse:
    """Health check endpoint.

    Returns:
        JSONResponse: Service health status
    """
    return JSONResponse(
        content={
            "status": "healthy",
            "service": "research-service",
            "version": "0.1.0",
            "environment": "development" if settings.DEBUG else "production",
        }
    )


# Root endpoint
@app.get("/", tags=["Root"])
async def root() -> JSONResponse:
    """Root endpoint with service information.

    Returns:
        JSONResponse: Service information and available endpoints
    """
    return JSONResponse(
        content={
            "service": "Research Service API",
            "version": "0.1.0",
            "docs": "/api/docs",
            "health": "/api/v1/health",
            "status": "ready",
        }
    )


# Include API routers
app.include_router(search_router, prefix="/api")
app.include_router(research_router, prefix="/api")
app.include_router(sessions_router, prefix="/api")
app.include_router(documents_router, prefix="/api/v1")  # Document upload and query
