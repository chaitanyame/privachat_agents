# Repository Restructuring Plan - PrivaChat Agents

**Date**: November 15, 2025  
**Status**: PLANNING PHASE  
**Risk Level**: HIGH (Breaking changes to imports, paths, Docker)

---

## 🎯 Goals

1. **Flatten structure**: Remove `research-service/` nesting
2. **Standard Python project**: Follow Python packaging best practices
3. **Clean repository root**: Professional open-source appearance
4. **Maintain functionality**: All services, tests, Docker must work
5. **Easy contribution**: Clear structure for new contributors

---

## 📊 Current Structure (BEFORE)

```
privachat_agents/                    # Repository root
├── .github/                         # GitHub config (copilot instructions)
├── research-service/                # ❌ UNNECESSARY NESTING
│   ├── src/                         # Actual Python package
│   │   ├── agents/                  # AI agents
│   │   ├── api/                     # FastAPI routes
│   │   ├── clients/                 # External API clients
│   │   ├── core/                    # Config, settings
│   │   ├── database/                # SQLAlchemy models
│   │   ├── models/                  # Pydantic schemas
│   │   ├── rag/                     # Vector store, retrieval
│   │   ├── services/                # Business logic
│   │   ├── utils/                   # Utilities
│   │   └── main.py                  # FastAPI app
│   ├── tests/                       # Test suite
│   ├── alembic/                     # DB migrations
│   ├── docs/                        # Technical docs
│   ├── scripts/                     # Utility scripts
│   ├── streamlit_ui.py             # Testing UI
│   ├── pyproject.toml              # Package config
│   ├── requirements.txt            # Dependencies
│   ├── requirements-dev.txt        # Dev dependencies
│   ├── docker-compose.yml          # Service orchestration
│   ├── Dockerfile                  # API container
│   ├── Dockerfile.streamlit        # UI container
│   └── alembic.ini                 # Migration config
├── searxng/                         # SearxNG config
├── docs/                            # ❌ DUPLICATE: General docs
├── docker-compose.yaml              # ❌ DUPLICATE: Root compose
├── LICENSE                          # MIT license
├── README.md                        # Main readme
├── CONTRIBUTING.md                  # Contribution guide
└── CODE_OF_CONDUCT.md              # Community standards
```

**Issues:**
- ❌ `research-service/` adds unnecessary nesting
- ❌ Two `docs/` folders (root + research-service)
- ❌ Two docker-compose files (root + research-service)
- ❌ Confusing for contributors (where to start?)
- ❌ Import paths: `from src.agents...` instead of `from privachat_agents.agents...`

---

## 🎨 Proposed Structure (AFTER)

```
privachat_agents/                    # Repository root
├── .github/                         # GitHub Actions, templates
│   ├── workflows/                   # CI/CD pipelines
│   │   ├── tests.yml               # Run tests on PR
│   │   ├── docker-build.yml        # Build Docker images
│   │   └── release.yml             # Automated releases
│   ├── ISSUE_TEMPLATE/             # Issue templates
│   └── copilot-instructions.md     # Copilot guidance
│
├── privachat_agents/                # ✅ Main Python package (renamed from src/)
│   ├── agents/                      # AI agents (search, research, synthesis)
│   ├── api/                         # FastAPI application
│   │   ├── v1/                     # API v1 endpoints
│   │   │   ├── endpoints/          # Route handlers
│   │   │   ├── dependencies.py     # FastAPI dependencies
│   │   │   └── router.py           # API router
│   │   └── middleware/             # CORS, logging, etc.
│   ├── clients/                     # External API clients
│   │   ├── openrouter.py           # OpenRouter LLM client
│   │   └── searxng.py              # SearxNG search client
│   ├── core/                        # Core configuration
│   │   ├── config.py               # Settings management
│   │   ├── logging.py              # Logging setup
│   │   └── exceptions.py           # Custom exceptions
│   ├── database/                    # Database layer
│   │   ├── models.py               # SQLAlchemy ORM models
│   │   ├── repositories/           # Repository pattern
│   │   └── session.py              # DB session management
│   ├── models/                      # Pydantic schemas
│   │   ├── requests.py             # API request models
│   │   ├── responses.py            # API response models
│   │   └── documents.py            # Document models
│   ├── rag/                         # RAG system
│   │   ├── embeddings.py           # Embedding generation
│   │   ├── retrieval.py            # Document retrieval
│   │   └── vectorstore.py          # pgvector integration
│   ├── services/                    # Business logic
│   │   ├── llm/                    # LLM service
│   │   ├── search/                 # Search service
│   │   └── crawler/                # Web crawling
│   ├── utils/                       # Shared utilities
│   ├── main.py                      # FastAPI app entry
│   └── __init__.py                  # Package initialization
│
├── tests/                           # ✅ Test suite
│   ├── unit/                        # Unit tests
│   ├── integration/                 # Integration tests
│   ├── e2e/                         # End-to-end tests
│   ├── conftest.py                  # Pytest fixtures
│   └── __init__.py
│
├── alembic/                         # ✅ Database migrations
│   ├── versions/                    # Migration files
│   └── env.py                       # Alembic environment
│
├── docs/                            # ✅ Comprehensive documentation
│   ├── api/                         # API documentation
│   ├── architecture/                # System architecture
│   ├── development/                 # Development guides
│   ├── deployment/                  # Deployment guides
│   └── contributing/                # Contribution guides
│
├── scripts/                         # ✅ Utility scripts
│   ├── setup_db.py                 # Database initialization
│   ├── run_tests.sh                # Test runner
│   └── fix_source_type.py          # Migration scripts
│
├── config/                          # ✅ Configuration files
│   ├── searxng/                    # SearxNG settings
│   │   ├── settings.yml
│   │   ├── limiter.toml
│   │   └── uwsgi.ini
│   └── docker/                     # Docker configs (if needed)
│
├── ui/                              # ✅ User interfaces
│   ├── streamlit_app.py            # Streamlit testing UI
│   └── requirements.txt            # UI-specific deps
│
├── .github/                         # GitHub configuration
├── .dockerignore                    # Docker ignore patterns
├── .env.example                     # Example environment file
├── .gitignore                       # Git ignore patterns
├── alembic.ini                      # Alembic configuration
├── docker-compose.yml               # ✅ Service orchestration (single file)
├── Dockerfile                       # ✅ API container
├── Dockerfile.streamlit             # ✅ UI container
├── pyproject.toml                   # ✅ Package configuration
├── requirements.txt                 # ✅ Production dependencies
├── requirements-dev.txt             # ✅ Development dependencies
├── LICENSE                          # MIT License
├── README.md                        # Main documentation
├── CONTRIBUTING.md                  # Contribution guidelines
├── CODE_OF_CONDUCT.md              # Community standards
├── SECURITY.md                      # Security policy
├── ACKNOWLEDGMENTS.md              # Credits
├── CHANGELOG.md                     # Version history
└── ROADMAP.md                       # Future plans
```

**Benefits:**
- ✅ Standard Python package: `pip install privachat-agents`
- ✅ Clear imports: `from privachat_agents.agents import SearchAgent`
- ✅ Professional structure: Matches popular open-source projects
- ✅ Easy navigation: Everything at root level
- ✅ CI/CD ready: GitHub Actions in `.github/workflows/`

---

## 📋 Migration Steps

### Phase 1: Prepare New Structure (1-2 hours, LOW RISK)

1. **Create branch**
   ```bash
   git checkout -b feature/flatten-structure
   ```

2. **Create new directories at root**
   ```bash
   mkdir -p privachat_agents tests alembic config/searxng ui .github/workflows
   ```

3. **Copy files to new locations** (DON'T delete originals yet)
   ```bash
   # Python package
   cp -r research-service/src/* privachat_agents/
   
   # Tests
   cp -r research-service/tests/* tests/
   
   # Migrations
   cp -r research-service/alembic/* alembic/
   
   # UI
   cp research-service/streamlit_ui.py ui/streamlit_app.py
   
   # SearxNG config
   cp -r searxng/* config/searxng/
   
   # Scripts
   cp -r research-service/scripts/* scripts/
   
   # Root files
   cp research-service/pyproject.toml .
   cp research-service/requirements.txt .
   cp research-service/requirements-dev.txt .
   cp research-service/alembic.ini .
   cp research-service/Dockerfile .
   cp research-service/Dockerfile.streamlit .
   cp research-service/docker-compose.yml docker-compose.yml
   ```

### Phase 2: Update Import Paths (2-3 hours, MEDIUM RISK)

4. **Find all imports**
   ```bash
   grep -r "from src\." privachat_agents/
   grep -r "import src\." privachat_agents/
   ```

5. **Update imports globally**
   ```bash
   # In all Python files: src. → privachat_agents.
   find privachat_agents/ -name "*.py" -exec sed -i 's/from src\./from privachat_agents./g' {} +
   find tests/ -name "*.py" -exec sed -i 's/from src\./from privachat_agents./g' {} +
   ```

6. **Update __init__.py**
   - Update `privachat_agents/__init__.py` to be proper package init

### Phase 3: Update Configuration Files (1-2 hours, MEDIUM RISK)

7. **Update pyproject.toml**
   ```toml
   [project]
   name = "privachat-agents"
   packages = ["privachat_agents"]  # Add this
   ```

8. **Update docker-compose.yml**
   - Remove `research-service/` path prefixes
   - Update volume mounts
   - Update build contexts: `./` instead of `./research-service/`

9. **Update Dockerfiles**
   - Change `COPY src/` → `COPY privachat_agents/`
   - Update PYTHONPATH if needed

10. **Update alembic.ini**
    - Update script_location: `alembic` instead of `research-service/alembic`

11. **Update alembic/env.py**
    - Change imports from `src.` to `privachat_agents.`

### Phase 4: Update Documentation (1 hour, LOW RISK)

12. **Merge docs/ folders**
    ```bash
    # Move research-service docs to main docs
    cp -r research-service/docs/* docs/
    ```

13. **Update all documentation**
    - Update file paths in markdown files
    - Update import examples
    - Update installation instructions

14. **Update README.md**
    - New structure diagram
    - Updated installation steps
    - Updated contribution guide references

### Phase 5: Testing (2-3 hours, HIGH RISK)

15. **Test imports**
    ```bash
    python -c "from privachat_agents.agents import SearchAgent"
    python -c "from privachat_agents import __version__"
    ```

16. **Run unit tests**
    ```bash
    pytest tests/unit/ -v
    ```

17. **Run integration tests**
    ```bash
    pytest tests/integration/ -v
    ```

18. **Test Docker build**
    ```bash
    docker compose build
    ```

19. **Test Docker deployment**
    ```bash
    docker compose up -d
    curl http://localhost:8001/health
    curl http://localhost:8503
    ```

20. **Test Streamlit UI**
    - Open http://localhost:8503
    - Run a search query
    - Verify results

### Phase 6: Cleanup (30 mins, LOW RISK)

21. **Remove old structure** (ONLY after everything works!)
    ```bash
    git rm -rf research-service/
    git rm -rf searxng/
    git rm docs/API_SPECIFICATION.md  # If moved to docs/api/
    ```

22. **Update .gitignore**
    - Remove `research-service/` references
    - Add proper Python package ignores

23. **Commit changes**
    ```bash
    git add .
    git commit -m "refactor: Flatten repository structure to standard Python package layout"
    ```

### Phase 7: CI/CD Setup (2 hours, OPTIONAL)

24. **Create GitHub Actions workflows**
    - `.github/workflows/tests.yml` - Run tests on PR
    - `.github/workflows/docker-build.yml` - Build Docker images
    - `.github/workflows/release.yml` - Automated releases

25. **Test CI/CD**
    - Push branch and create PR
    - Verify workflows run successfully

### Phase 8: Release (1 hour, LOW RISK)

26. **Merge to main**
    ```bash
    git checkout main
    git merge feature/flatten-structure
    git push origin main
    ```

27. **Create release tag**
    ```bash
    git tag -a v0.1.0 -m "Release v0.1.0: Restructured repository"
    git push origin v0.1.0
    ```

28. **Update GitHub repository**
    - Add description
    - Add topics: `ai`, `agents`, `rag`, `fastapi`, `pydantic`, `python`, `llm`, `search`
    - Create GitHub release with changelog

---

## ⚠️ Risk Assessment

### HIGH RISK (Breaking Changes)
- ✅ **Import path changes**: Every Python file affected
- ✅ **Docker builds**: Paths change, must rebuild
- ✅ **Database migrations**: alembic.ini and env.py updates
- ✅ **Existing deployments**: Will break without update

### MEDIUM RISK (Configuration)
- ⚠️ **docker-compose.yml**: Volume mounts, build contexts
- ⚠️ **Environment variables**: May need updates
- ⚠️ **Documentation**: All file paths change

### LOW RISK (Cosmetic)
- ℹ️ **README updates**: Clear improvements
- ℹ️ **Directory structure**: Better organization
- ℹ️ **CI/CD setup**: New addition

---

## 🛡️ Mitigation Strategy

### Backup Plan
1. **Keep feature branch**: Don't delete `feature/flatten-structure`
2. **Tag before merge**: `git tag pre-restructure`
3. **Test everything**: Full test suite + manual testing
4. **Gradual rollout**: Deploy to test environment first

### Rollback Plan
If restructuring fails:
```bash
git checkout main
git reset --hard pre-restructure
git push origin main --force
```

### Testing Checklist
- [ ] All unit tests pass
- [ ] All integration tests pass
- [ ] Docker builds successfully
- [ ] Docker containers start healthy
- [ ] API health check works
- [ ] Streamlit UI loads
- [ ] Search functionality works
- [ ] Research functionality works
- [ ] Database migrations work
- [ ] Redis caching works

---

## 📅 Timeline Estimate

| Phase | Duration | Risk | Dependencies |
|-------|----------|------|--------------|
| Phase 1: Prepare | 1-2 hours | LOW | None |
| Phase 2: Imports | 2-3 hours | MEDIUM | Phase 1 |
| Phase 3: Config | 1-2 hours | MEDIUM | Phase 2 |
| Phase 4: Docs | 1 hour | LOW | Phase 3 |
| Phase 5: Testing | 2-3 hours | HIGH | Phase 4 |
| Phase 6: Cleanup | 30 mins | LOW | Phase 5 |
| Phase 7: CI/CD | 2 hours | LOW | Phase 6 |
| Phase 8: Release | 1 hour | LOW | Phase 7 |
| **TOTAL** | **11-14.5 hours** | **MEDIUM-HIGH** | Sequential |

**Recommended**: Allocate 2 full working days for safety.

---

## 🤔 Alternative: Minimal Restructure

If full restructuring is too risky, consider minimal changes:

### Option B: Keep research-service/ but improve
```
privachat_agents/
├── research-service/              # Keep existing structure
│   └── ... (all existing code)
├── LICENSE, README.md, etc.       # Root files stay
└── .github/workflows/             # Add CI/CD only
```

**Changes:**
1. Add `.github/workflows/` for CI/CD
2. Improve documentation
3. Keep all paths as-is
4. Update README with clear structure explanation

**Pros:**
- ✅ Zero breaking changes
- ✅ Minimal risk
- ✅ Quick to implement (2-3 hours)

**Cons:**
- ❌ Not standard Python package structure
- ❌ Confusing for contributors
- ❌ Import paths not ideal

---

## 🎯 Recommendation

**PROCEED WITH FULL RESTRUCTURE** (Option A)

**Reasons:**
1. **Long-term benefit**: Proper structure attracts more contributors
2. **Professional appearance**: Matches major open-source projects
3. **Package installability**: Can publish to PyPI later
4. **One-time pain**: Better to do it early before more users
5. **Current state**: Repository is new, minimal external dependencies

**When to do it:**
- ✅ **NOW** - Repository just published, no external users yet
- ✅ Tests are working (85% coverage)
- ✅ Docker setup is stable
- ✅ All features working

**When NOT to do it:**
- ❌ Active users depend on current structure
- ❌ Many open PRs with conflicts
- ❌ Time pressure for new features
- ❌ Unstable codebase

---

## 📝 Next Steps

1. **Review this plan** - Discuss any concerns
2. **Set aside time** - Block 2 days for this work
3. **Create backup** - Tag current state
4. **Execute phases sequentially** - Don't skip testing
5. **Communicate** - If anyone is using it, notify them

---

## 📞 Questions to Resolve

Before starting:
- [ ] Is anyone currently using this repository?
- [ ] Are there any active external dependencies?
- [ ] Do we have a test environment for validation?
- [ ] Should we add CI/CD in this PR or separate?
- [ ] Do we want to publish to PyPI eventually?

---

**Status**: ⏸️ AWAITING APPROVAL TO PROCEED

**Author**: GitHub Copilot  
**Reviewed By**: [Pending]  
**Approved**: [Pending]
