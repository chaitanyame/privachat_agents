# syntax=docker/dockerfile:1.7
FROM python:3.11-slim

WORKDIR /app

# Speed up pip and keep output lean
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_COLOR=1 \
    HF_HOME=/home/research/.cache/huggingface \
    TRANSFORMERS_CACHE=/home/research/.cache/huggingface \
    HUGGINGFACE_HUB_CACHE=/home/research/.cache/huggingface

# Install system dependencies (minimal set) - use cache mount for apt
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user early
RUN useradd -m -u 1000 research

# Install Python dependencies first for better layer caching
COPY requirements.txt requirements-dev.txt ./
# Use BuildKit cache mount when available to accelerate pip installs
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install -r requirements-dev.txt

# Playwright installation (browsers and OS deps)
# Create cache directory and set permissions before switching users
RUN mkdir -p /opt/app/.cache/ms-playwright && chown -R research:research /opt/app
ENV PLAYWRIGHT_BROWSERS_PATH=/opt/app/.cache/ms-playwright
USER research
# Only install chromium (skip webkit/firefox) to save ~1GB and 2+ minutes
RUN playwright install chromium
USER root
RUN playwright install-deps chromium

# NOTE: Hugging Face models are downloaded at RUNTIME on first use
# This saves ~5 minutes of build time and ~400MB of image size
# Models are cached in /home/research/.cache/huggingface (persisted via volume)

# Copy application code with correct ownership (avoid chown step)
COPY --chown=research:research privachat_agents/ ./privachat_agents/
COPY --chown=research:research tests/ ./tests/
COPY --chown=research:research pyproject.toml .
COPY --chown=research:research alembic/ ./alembic/
COPY --chown=research:research alembic.ini .

# Switch to research user for runtime
USER research

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Run application
CMD ["uvicorn", "privachat_agents.main:app", "--host", "0.0.0.0", "--port", "8000"]
