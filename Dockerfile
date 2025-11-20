# syntax=docker/dockerfile:1.7
# Stage 1: Builder
FROM python:3.11-slim-bookworm AS builder

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_COLOR=1

# Install build dependencies
RUN rm -f /etc/apt/apt.conf.d/docker-clean; echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' > /etc/apt/apt.conf.d/keep-cache
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt requirements-dev.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --prefix=/install -r requirements.txt && \
    pip install --prefix=/install -r requirements-dev.txt

# Stage 2: Runtime
FROM python:3.11-slim-bookworm AS runtime

# Use /opt/app as the application directory
WORKDIR /opt/app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    APP_HOME=/opt/app \
    APP_DATA=/opt/app/data \
    HF_HOME=/opt/app/data/.cache/huggingface \
    NLTK_DATA=/opt/app/data/nltk_data \
    PATH=/install/bin:$PATH \
    PYTHONPATH=/install/lib/python3.11/site-packages

# Install runtime dependencies
RUN rm -f /etc/apt/apt.conf.d/docker-clean; echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' > /etc/apt/apt.conf.d/keep-cache
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user 'app' and necessary directories with proper permissions
RUN useradd -m -u 1000 app && \
    mkdir -p /opt/app/data/.crawl4ai /opt/app/data/.cache /opt/app/data/nltk_data /opt/app/data/.streamlit && \
    chown -R app:app /opt/app && \
    chmod -R 755 /opt/app

# Copy installed python packages from builder
COPY --from=builder /install /install

# Playwright installation
# We need to install browsers as the app user, but deps as root
# Note: playwright install-deps might install many packages. 
# Ideally we should list them explicitly to keep image small, but for now we rely on the tool.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    playwright install-deps chromium

USER app

# Install Playwright browsers WITHOUT cache mount (they need to persist in the image)
RUN playwright install chromium

# Pre-download Hugging Face models (without cache mount so they persist in image)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')" || true
RUN python -c "from transformers import AutoTokenizer, AutoModelForSequenceClassification; name='cross-encoder/ms-marco-MiniLM-L-6-v2'; AutoTokenizer.from_pretrained(name); AutoModelForSequenceClassification.from_pretrained(name)" || true

# Copy application code to /opt/app
COPY --chown=app:app privachat_agents/ /opt/app/privachat_agents/
COPY --chown=app:app tests/ /opt/app/tests/
COPY --chown=app:app pyproject.toml /opt/app/
COPY --chown=app:app alembic/ /opt/app/alembic/
COPY --chown=app:app alembic.ini /opt/app/

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Run application
CMD ["uvicorn", "privachat_agents.main:app", "--host", "0.0.0.0", "--port", "8000"]
