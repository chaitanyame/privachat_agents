# Contributing to PrivaChat Agents

Thank you for your interest in contributing to PrivaChat Agents! This document provides guidelines for contributing to the project.

## Code of Conduct

By participating in this project, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md).

## How to Contribute

### Reporting Bugs

Before creating bug reports, please check existing issues. When creating a bug report, include:

- **Clear title and description**
- **Steps to reproduce** the behavior
- **Expected behavior**
- **Actual behavior**
- **Environment details** (OS, Python version, Docker version)
- **Logs or error messages**

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, include:

- **Clear title and description**
- **Use case** - Why is this enhancement useful?
- **Proposed solution**
- **Alternatives considered**

### Pull Requests

1. **Fork the repository** and create your branch from `main`
2. **Follow coding standards**:
   - Python: PEP 8 style guide
   - Type hints required for all functions
   - Docstrings for all public functions/classes
   - Follow existing code structure
3. **Write tests**:
   - Follow TDD (Test-Driven Development)
   - Write tests BEFORE implementation
   - Maintain 80%+ coverage
   - Unit, integration, and e2e tests as appropriate
4. **Update documentation**:
   - Update README if adding features
   - Add docstrings to new functions
   - Update API documentation if changing endpoints
5. **Commit messages**:
   - Use present tense ("Add feature" not "Added feature")
   - Use imperative mood ("Move cursor to..." not "Moves cursor to...")
   - Limit first line to 72 characters
   - Reference issues: "Fix #123: Description"

### Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/privachat_agents.git
cd privachat_agents

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r research-service/requirements.txt
pip install -r research-service/requirements-dev.txt

# Set up environment
cp research-service/.env.example research-service/.env
# Edit .env with your API keys

# Start services
docker-compose up -d

# Run tests
cd research-service
pytest tests/ -v

# Run with coverage
pytest --cov=src --cov-report=html tests/
```

### Coding Standards

**Python Code:**
- Use type hints for all function parameters and return types
- Write docstrings for all public functions/classes (Google style)
- Use meaningful variable names
- Keep functions small and focused (single responsibility)
- Follow async/await patterns consistently

**Tests:**
- Use AAA pattern (Arrange, Act, Assert)
- One assertion per test when possible
- Use descriptive test names: `test_<function>_<scenario>_<expected_result>`
- Use fixtures for reusable test data
- Mock external dependencies

**Example:**
```python
def calculate_similarity(text1: str, text2: str) -> float:
    """Calculate cosine similarity between two text strings.
    
    Args:
        text1: First text string to compare
        text2: Second text string to compare
        
    Returns:
        Similarity score between 0.0 and 1.0
        
    Raises:
        ValueError: If either text is empty
    """
    if not text1 or not text2:
        raise ValueError("Text strings cannot be empty")
    # Implementation...
```

### Testing Guidelines

**Test Structure:**
```
tests/
├── unit/           # Fast, isolated tests (no I/O)
├── integration/    # Tests with DB, external services
└── e2e/           # Full pipeline tests
```

**Before submitting:**
```bash
# Run all tests
pytest tests/ -v

# Check coverage
pytest --cov=src --cov-report=term-missing tests/

# Lint code
ruff check .
mypy src/

# Format code
ruff format .
```

### Documentation

- Use Markdown for all documentation
- Update README.md for user-facing changes
- Update ARCHITECTURE.md for structural changes
- Add API examples for new endpoints
- Include type information in examples

### Git Workflow

1. Create feature branch: `git checkout -b feature/my-new-feature`
2. Make changes and commit: `git commit -m "Add feature"`
3. Push to fork: `git push origin feature/my-new-feature`
4. Create Pull Request from your fork to `main` branch
5. Address review feedback
6. Squash commits if requested
7. Wait for approval and merge

### Review Process

- All PRs require at least one approval
- CI/CD must pass (tests, linting, type checking)
- Code coverage should not decrease
- Documentation must be updated
- Breaking changes require major version bump

## Architecture Decisions

For significant changes, please:
1. Open an issue to discuss the proposal
2. Wait for maintainer feedback
3. Create RFC (Request for Comments) if needed
4. Implement after approval

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Questions?

Open an issue with the `question` label or join our community discussions.

---

Thank you for contributing to PrivaChat Agents! 🎉
