# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2025-11-15

### 🎉 Major Repository Restructure

This release represents a complete restructuring of the repository from a nested `research-service/src/` layout to a professional, flat Python package structure.

### ✨ Added

- **Flat Package Structure**: Moved from `research-service/src/` to root-level `privachat_agents/` package
- **Root-Level Tests**: Moved tests from `research-service/tests/` to `tests/` for standard Python project layout
- **Root-Level Migrations**: Moved alembic from `research-service/alembic/` to `alembic/`
- **Organized Config**: Created `config/` directory for organized configuration files
- **UI Directory**: Created `ui/` directory for Streamlit and other interfaces
- **Scripts Directory**: Organized utility scripts in root `scripts/` directory
- **Professional Package**: Can now be installed with `pip install privachat-agents`
- **Updated Documentation**: Completely rewritten README.md reflecting new structure
- **CHANGELOG.md**: This file to track version changes

### 🔄 Changed

- **Import Paths**: Changed from `from src.*` to `from privachat_agents.*` (200+ imports updated)
- **Package Name**: Now `privachat-agents` instead of nested research-service
- **Docker Configuration**: Updated Dockerfile and docker-compose.yml for new structure
- **Service Ports**: 
  - API: 3000 → 8001
  - SearxNG: 8080 → 4000
  - Redis: 6379 → 6380
  - PostgreSQL: 5432 → 5433
  - Streamlit: 8501 → 8503
- **Container Names**: Updated to reflect new package name
- **Volume Mounts**: Updated all Docker volume paths
- **pyproject.toml**: Updated with packages field and new paths
- **alembic.ini**: Updated paths to new structure
- **.gitignore**: Enhanced with comprehensive Python patterns
- **Test Coverage Paths**: Updated to new package structure

### 🗑️ Removed

- **research-service/ directory**: Removed 171 files (nested structure)
- **docker-compose.yaml**: Removed duplicate (using docker-compose.yml)
- **searxng/ directory**: Moved to config/searxng/
- **46,566 lines of duplicate code**: Cleanup removed all duplicates

### 🐛 Fixed

- **Import Organization**: All imports now properly organized (stdlib, third-party, local)
- **Package Discovery**: Python can now properly discover the package
- **Test Execution**: Tests now run from root level with correct imports
- **Docker Builds**: Both API and Streamlit containers build successfully
- **Migration Paths**: Alembic migrations now reference correct package paths

### 📚 Documentation

- **README.md**: Complete rewrite with new structure, architecture, and usage
- **RESTRUCTURE_PLAN.md**: Detailed 8-phase restructuring plan
- **.github/copilot-instructions.md**: Updated development standards
- **API Documentation**: Updated all endpoint references to new paths

### 🔧 Technical Details

**Repository Statistics:**
- **Before**: ~360 files, ~100,000+ lines (nested structure)
- **After**: 214 active files, ~54,887 lines (flat structure)
- **Reduction**: 46,566 lines of duplicate code removed
- **Commits**: 3 commits on feature/flatten-structure branch

**Improvements:**
- ✅ Standard Python package layout
- ✅ Professional structure matching major open-source projects
- ✅ Clear imports: `from privachat_agents.agents import SearchAgent`
- ✅ Zero duplication (single source of truth)
- ✅ Better organization (logical grouping)
- ✅ Easy navigation (everything at root, no deep nesting)
- ✅ CI/CD ready (GitHub Actions compatible)
- ✅ Contributor friendly (clear structure for new developers)

**Testing:**
- ✅ All Docker containers build successfully
- ✅ All services healthy (API, Streamlit, PostgreSQL, Redis, SearxNG)
- ✅ Health checks passing
- ✅ API responding correctly
- ✅ Imports working correctly
- ✅ No breaking issues found

### 🚀 Migration Guide

**For Users:**

1. **Pull the latest changes:**
   ```bash
   git checkout main
   git pull origin main
   ```

2. **Update imports if using as library:**
   ```python
   # OLD
   from src.agents.search_agent import SearchAgent
   
   # NEW
   from privachat_agents.agents.search_agent import SearchAgent
   ```

3. **Update Docker commands:**
   ```bash
   # OLD
   docker compose -f research-service/docker-compose.yml up -d
   
   # NEW
   docker compose up -d
   ```

4. **Update environment variables:**
   - Copy `.env.example` to `.env`
   - Update any service URLs with new ports

**For Developers:**

1. **Update your development environment:**
   ```bash
   # Install from root (not research-service/)
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   
   # Run tests from root
   pytest tests/ -v
   
   # Run API from root
   uvicorn privachat_agents.main:app --reload
   ```

2. **Update IDE configuration:**
   - Update source roots to repository root
   - Update test discovery to `tests/` directory
   - Update run configurations for new paths

3. **Update imports in your PRs:**
   - Use `privachat_agents.*` instead of `src.*`
   - Follow new import organization standards

### ⚠️ Breaking Changes

- **Import paths**: All `src.*` imports must be changed to `privachat_agents.*`
- **Docker paths**: Volume mounts and paths updated in docker-compose.yml
- **Service ports**: All service ports have changed (see Changed section)
- **Directory structure**: Old `research-service/` directory removed
- **Command paths**: All CLI commands now reference new package name

### 📦 Dependency Updates

No dependency version changes in this release - only structural reorganization.

### 🎯 Next Steps (v0.3.0)

- [ ] Add GitHub Actions CI/CD pipeline
- [ ] Publish to PyPI as `privachat-agents`
- [ ] Add pre-commit hooks
- [ ] Enhanced monitoring and metrics
- [ ] Multi-modal document support

---

## [0.1.0] - 2025-11-09

### Initial Release

- FastAPI-based research service
- Pydantic AI agents (search, research, synthesis)
- PostgreSQL with pgvector for RAG
- Redis caching layer
- SearxNG integration
- Streamlit testing UI
- Docker Compose deployment
- Comprehensive test suite
- SpaCy temporal detection
- Langfuse monitoring integration

---

**Legend:**
- 🎉 Major changes
- ✨ New features
- 🔄 Changes
- 🐛 Bug fixes
- 🗑️ Removals
- 📚 Documentation
- 🔧 Technical details
- ⚠️ Breaking changes
