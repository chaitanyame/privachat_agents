# Docker Build Quick Reference

## 📦 Build Commands

### Standard Build
```bash
docker-compose build api
```

### Build with Progress Details
```bash
docker-compose build api --progress=plain
```

### Build with No Cache (Force Rebuild)
```bash
docker-compose build api --no-cache
```

### Build with Timing
```bash
time docker-compose build api
```

---

## 📊 Performance Metrics

### Build Context Size
- **Before optimization**: 122MB
- **After optimization**: 10.29KB
- **Reduction**: 99.99%

### Build Times
| Scenario | Time | Notes |
|----------|------|-------|
| Fresh build (no cache) | ~3-4 min | First time or after `--no-cache` |
| Rebuild (with cache) | ~1.5 min | Code changes only |
| Rebuild (deps changed) | ~2-3 min | requirements.txt modified |

---

## 🔍 What's Being Built?

### Layers Overview
1. **Base Image**: `python:3.11-slim` (~120MB)
2. **System Dependencies**: gcc, g++, libpq-dev, curl (~50MB)
3. **Python Packages**: FastAPI, SQLAlchemy, etc. (~300MB)
4. **Playwright Browser**: chromium only (~130MB)
5. **Application Code**: ~2MB

**Total Image Size**: ~600MB (before HF models)

### Excluded from Build Context
- `.git/` (117MB)
- `docs/` (320KB)
- `tests/` (2.5MB)
- `examples/`, `scripts/`
- All `*.md` files
- Config files not needed in container

---

## 🚀 Runtime Model Loading

### HuggingFace Models (Lazy Loaded)
**First Query After Fresh Deploy:**
```
Downloading sentence-transformers/all-MiniLM-L6-v2 (~90MB)
Downloading answerdotai/ModernBERT-base (~310MB)
Time: ~10-15 seconds (one-time)
```

**Subsequent Queries:**
- Models cached in `/home/research/.cache/huggingface`
- No download needed (instant)

### Pre-warming Models (Production)
```bash
# After deployment, trigger one search to download models
curl -X POST http://localhost:8000/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query": "test warmup"}'
```

---

## 🛠️ Troubleshooting

### Build Fails: "permission denied"
**Symptom**: Playwright install fails with permission errors

**Fix**: Ensure cache directory exists and has correct ownership
```dockerfile
RUN mkdir -p /opt/app/.cache/ms-playwright && \
    chown -R research:research /opt/app
```

### Build Context Too Large
**Check current size:**
```bash
docker-compose build api --progress=plain 2>&1 | grep "transferring context"
```

**Expected output:**
```
#7 transferring context: 10.29KB 0.0s done
```

**If > 15KB:** Check if `.dockerignore` is working
```bash
# List what would be sent to Docker daemon
tar -czf - . --exclude-from=.dockerignore | wc -c
```

### Playwright Browser Not Found
**Symptom**: Runtime error "Browser executable not found"

**Fix**: Check browser path environment variable
```dockerfile
ENV PLAYWRIGHT_BROWSERS_PATH=/opt/app/.cache/ms-playwright
```

**Verify in running container:**
```bash
docker exec -it <container> ls -la /opt/app/.cache/ms-playwright/chromium-*/
```

### HuggingFace Models Download Slow
**Symptom**: First query takes 30+ seconds

**Causes:**
- Slow network connection
- Large models being downloaded
- No cache volume mounted

**Fix**: Ensure volume is mounted in `docker-compose.yml`
```yaml
volumes:
  - huggingface-cache:/home/research/.cache/huggingface
```

---

## 📝 Key Files

### `.dockerignore` (60+ patterns)
Located at: `.dockerignore`

**Most Important Exclusions:**
```
.git
.github
docs
examples
scripts
tests
*.md
config/searxng
```

### `Dockerfile` (69 lines)
Located at: `Dockerfile`

**Key Optimizations:**
- BuildKit cache mounts for apt and pip
- Playwright chromium only (no webkit/firefox)
- HF models loaded at runtime (not build time)
- Non-root user for runtime security

### `docker-compose.yml`
**API Service Definition:**
```yaml
api:
  build:
    context: .
    dockerfile: Dockerfile
  volumes:
    - huggingface-cache:/home/research/.cache/huggingface
  environment:
    - HF_HOME=/home/research/.cache/huggingface
```

---

## 🔧 Maintenance

### Updating Python Dependencies
```bash
# 1. Update requirements.txt or requirements-dev.txt
# 2. Rebuild (will re-run pip install layer)
docker-compose build api

# 3. Restart services
docker-compose up -d api
```

### Clearing Build Cache
```bash
# Clear all build cache (use sparingly!)
docker builder prune -af

# Clear only dangling cache
docker builder prune -f
```

### Checking Image Size
```bash
docker images | grep privachat_agents-api
```

**Expected size**: ~600MB (excluding HF models in volume)

---

## 📚 Related Documentation

- **Build Optimization Details**: `docs/DOCKER_BUILD_OPTIMIZATION.md`
- **Dockerfile Best Practices**: https://docs.docker.com/develop/dev-best-practices/
- **BuildKit Reference**: https://docs.docker.com/build/buildkit/

---

## ✅ Quick Health Check

**After Build:**
1. Check build completed: `docker images | grep privachat_agents-api`
2. Check size is reasonable: ~600MB
3. Start container: `docker-compose up -d api`
4. Check health: `curl http://localhost:8000/api/v1/health`
5. Trigger model download: Send first search query
6. Verify models cached: `docker exec -it api ls /home/research/.cache/huggingface/`

**Expected Results:**
- ✅ Image exists with recent timestamp
- ✅ Container starts within 5 seconds
- ✅ Health endpoint returns 200 OK
- ✅ First search takes ~15 seconds (model download)
- ✅ Second search takes < 2 seconds (cached models)
