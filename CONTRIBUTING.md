# Contributing to MiRAGE

Thank you for your interest in contributing to MiRAGE! This document provides guidelines and instructions for contributing.

## Getting Started

### Development Setup

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/YOUR-USERNAME/mirage-bench.git
   cd mirage-bench
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install development dependencies**
   ```bash
   pip install -e ".[dev]"
   ```

4. **Set up your configuration**
   ```bash
   cp config.yaml.example config.yaml
   # Edit config.yaml with your API keys
   ```

5. **Run preflight checks**
   ```bash
   python preflight_check.py
   ```

## Development Guidelines

### Code Style

- Follow [PEP 8](https://pep8.org/) style guidelines
- Use meaningful variable and function names
- Add docstrings for all public functions and classes
- Keep functions focused and under 50 lines when possible

### Formatting

We use `black` and `isort` for code formatting:

```bash
# Format code
black .
isort .

# Check formatting without modifying
black --check .
isort --check-only .
```

### Type Hints

Use type hints for function parameters and return values:

```python
def process_chunk(chunk: dict, max_depth: int = 10) -> tuple[list, dict]:
    """Process a single chunk for QA generation.
    
    Args:
        chunk: Dictionary containing chunk content and metadata
        max_depth: Maximum retrieval depth
        
    Returns:
        Tuple of (qa_pairs, context_data)
    """
    pass
```

### Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test file
pytest tests/test_qa_generation.py
```

## Making Changes

### Branching Strategy

1. Create a feature branch from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes with clear, atomic commits:
   ```bash
   git commit -m "Add: new feature description"
   git commit -m "Fix: bug description"
   git commit -m "Update: documentation changes"
   ```

3. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

### Commit Messages

Use clear, descriptive commit messages:

- `Add:` for new features
- `Fix:` for bug fixes
- `Update:` for updates to existing functionality
- `Remove:` for removed features
- `Docs:` for documentation changes
- `Refactor:` for code refactoring
- `Test:` for test additions/modifications

### Pull Request Process

1. **Update documentation** if your changes affect the API or user-facing features
2. **Add tests** for new functionality
3. **Ensure all tests pass** before submitting
4. **Update the README** if you're adding new features
5. **Fill out the PR template** completely

## Project Structure

```
MiRAGE/
├── src/mirage/                    # Main package
│   ├── __init__.py               # Package initialization
│   ├── main.py                   # Pipeline orchestration
│   ├── cli.py                    # Command-line interface
│   ├── core/                     # Core functionality
│   │   ├── config.py             # Configuration management
│   │   ├── llm.py                # LLM/VLM API interfaces
│   │   └── prompts.py            # Prompt templates
│   ├── embeddings/               # Embedding models
│   │   ├── models.py             # Embedding model selection
│   │   ├── rerankers_multimodal.py  # VLM-based reranking
│   │   └── rerankers_text.py     # Text-based reranking
│   ├── pipeline/                 # Processing pipeline
│   │   ├── pdf_processor.py      # PDF to Markdown conversion
│   │   ├── chunker.py            # Semantic chunking
│   │   ├── context.py            # Multi-hop context retrieval
│   │   ├── qa_generator.py       # QA generation and verification
│   │   ├── domain.py             # Domain/expert extraction
│   │   └── deduplication.py      # QA deduplication
│   ├── evaluation/               # Evaluation metrics
│   │   ├── metrics.py            # Standard RAGAS metrics
│   │   └── metrics_optimized.py  # Optimized metrics (faster)
│   └── utils/                    # Utilities
│       ├── preflight.py          # System checks
│       ├── stats.py              # Dataset statistics
│       └── ablation.py           # Ablation studies
├── data/documents/               # Input documents folder
├── output/                       # Generated results
├── assets/                       # Documentation images
├── config.yaml.example           # Example configuration
├── run_mirage.py                 # Main entry point script
├── setup.py                      # Package installation
├── pyproject.toml                # Package configuration
├── requirements.txt              # Dependencies
├── README.md                     # This file
├── CONTRIBUTING.md               # Contribution guidelines
└── LICENSE                       # Apache 2.0 License
```

## Reporting Issues

When reporting issues, please include:

1. **Environment info**: Python version, OS, GPU (if applicable)
2. **Steps to reproduce**: Clear, minimal reproduction steps
3. **Expected behavior**: What you expected to happen
4. **Actual behavior**: What actually happened
5. **Error messages**: Full traceback if applicable
6. **Configuration**: Relevant config.yaml settings (without API keys!)

## Feature Requests

We welcome feature requests! Please:

1. Check existing issues to avoid duplicates
2. Clearly describe the use case
3. Explain why existing features don't meet your needs
4. If possible, suggest an implementation approach

## License

By contributing to MiRAGE, you agree that your contributions will be licensed under the Apache License 2.0.

## Thank You!

We appreciate your contributions to making MiRAGE better for everyone!
