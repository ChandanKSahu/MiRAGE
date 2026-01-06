# MiRAGE: Multimodal Multihop RAG Evaluation Dataset Generator

<p align="center">
  <img src="https://img.shields.io/badge/python-3.9+-blue.svg" alt="Python 3.9+">
  <img src="https://img.shields.io/badge/license-Apache%202.0-green.svg" alt="License">
  <img src="https://img.shields.io/pypi/v/mirage-bench.svg" alt="PyPI">
</p>

**MiRAGE** is a multi-agent framework for generating high-quality, multimodal, multihop question-answer datasets for evaluating Retrieval-Augmented Generation (RAG) systems. It automatically extracts domain expertise, builds complete context through iterative retrieval, and generates verified QA pairs from technical documents.

<p align="center">
  <img src="assets/mirage_framework.png" alt="MiRAGE Framework Architecture" width="100%">
</p>

## Key Features

- **Multi-hop Context Completion**: Iteratively expands incomplete chunks with relevant context across documents
- **Domain and Expert Role Detection**: Automatic domain identification using BERTopic + LLM
- **Multi-stage QA Pipeline**: Generate, Select, Verify, Correct for quality assurance
- **Multimodal Support**: Handles text, tables, figures, and images in documents
- **Cross-Document Retrieval**: Unified FAISS index enables retrieval across all documents
- **Hierarchical Deduplication**: Two-stage clustering with LLM-based merging
- **Multiple Backend Support**: Gemini, OpenAI, and local Ollama models
- **Optimized Evaluation**: 3-5x faster metrics with harmonized RAGAS implementation
- **Fully Parallelized**: Thread and process pools for maximum throughput

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Pipeline Overview](#pipeline-overview)
- [Usage](#usage)
- [Output Format](#output-format)
- [Evaluation Metrics](#evaluation-metrics)
- [Hyperparameter Guide](#hyperparameter-guide)
- [API Keys Setup](#api-keys-setup)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)

## Installation

### From PyPI (Recommended)

```bash
pip install mirage-bench
```

### From Source (Development)

```bash
# Clone the repository
git clone https://github.com/ChandanKSahu/MiRAGE.git
cd MiRAGE

# Install in development mode
pip install -e .

# Or install with all optional dependencies
pip install -e ".[all]"
```

### With Optional Dependencies

```bash
# GPU support (CUDA-enabled embeddings and FAISS)
pip install mirage-bench[gpu]

# PDF processing (Docling for PDF to Markdown conversion)
pip install mirage-bench[pdf]

# Evaluation metrics (RAGAS and LangChain)
pip install mirage-bench[eval]

# Development tools (testing, linting)
pip install mirage-bench[dev]

# All dependencies
pip install mirage-bench[all]
```

## Quick Start

### 1. Clone and Install

```bash
git clone https://github.com/ChandanKSahu/MiRAGE.git
cd MiRAGE
pip install -e .
```

### 2. Add Your Documents

Place your PDF, HTML, or other documents in the `data/documents/` folder:

```bash
# The folder structure should look like:
data/
└── documents/
    ├── document1.pdf
    ├── document2.pdf
    └── ...
```

A sample dataset (`data/FinanceAnnualReports.zip`) is included for testing.

### 3. Configure API Keys

Create your configuration file:

```bash
cp config.yaml.example config.yaml
```

Edit `config.yaml` to add your API keys:

```yaml
backend:
  active: GEMINI  # Options: GEMINI, OPENAI, OLLAMA
  
  gemini:
    api_key_path: ~/.config/gemini/api_key.txt
    # Or use environment variable: export GEMINI_API_KEY="your-key"
    
  openai:
    api_key_path: ~/.config/openai/api_key.txt
    # Or use environment variable: export OPENAI_API_KEY="your-key"
    
  ollama:
    base_url: http://localhost:11434
    # No API key needed for local Ollama

paths:
  input_pdf_dir: data/documents
  output_dir: output/my_dataset
```

### 4. Run Preflight Checks

```bash
python run_mirage.py --preflight
```

### 5. Generate QA Dataset

```bash
python run_mirage.py
```

## Project Structure

```
MiRAGE/
├── src/mirage/                 # Main package
│   ├── __init__.py            # Package exports
│   ├── cli.py                 # Command line interface
│   ├── core/                  # Core functionality
│   │   ├── llm.py            # LLM/VLM API interfaces
│   │   ├── prompts.py        # Prompt templates
│   │   └── config.py         # Configuration management
│   ├── embeddings/            # Embedding models
│   │   ├── models.py         # Embedding model classes
│   │   ├── rerankers_multimodal.py
│   │   └── rerankers_text.py
│   ├── pipeline/              # Processing pipeline
│   │   ├── pdf_processor.py  # PDF to Markdown
│   │   ├── chunker.py        # Semantic chunking
│   │   ├── context.py        # Multi-hop retrieval
│   │   ├── qa_generator.py   # QA generation
│   │   ├── domain.py         # Domain extraction
│   │   └── deduplication.py  # Deduplication
│   ├── evaluation/            # Metrics
│   │   ├── metrics.py
│   │   └── metrics_optimized.py
│   └── utils/                 # Utilities
│       ├── preflight.py      # Preflight checks
│       ├── stats.py          # Dataset statistics
│       └── ablation.py       # Ablation studies
├── data/                      # Your documents go here
│   └── documents/            # Input PDFs/HTMLs
├── output/                    # Generated results
├── assets/                    # Documentation images
├── config.yaml.example        # Example configuration
├── run_mirage.py             # Main entry point
├── setup.py                   # Package setup
└── README.md
```

## Configuration

MiRAGE uses a YAML configuration file. Key sections:

| Section | Description |
|---------|-------------|
| `backend` | LLM/VLM provider settings (Gemini, OpenAI, Ollama) |
| `paths` | Input documents and output directory |
| `qa_generation` | Target QA pairs and type (multihop/multimodal/text) |
| `embedding` | Embedding model and batch size |
| `retrieval` | Multi-hop retrieval parameters |
| `deduplication` | Similarity thresholds for deduplication |
| `evaluation` | Metrics and evaluation settings |

See [`config.yaml.example`](config.yaml.example) for full documentation.

## API Keys Setup

### Google Gemini

```bash
# Option 1: Environment variable
export GEMINI_API_KEY="your-gemini-api-key"

# Option 2: File (create the directory first)
mkdir -p ~/.config/gemini
echo "your-gemini-api-key" > ~/.config/gemini/api_key.txt
```

### OpenAI

```bash
# Option 1: Environment variable
export OPENAI_API_KEY="your-openai-api-key"

# Option 2: File
mkdir -p ~/.config/openai
echo "your-openai-api-key" > ~/.config/openai/api_key.txt
```

### Ollama (Local)

No API key needed. Just install and start Ollama:

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull models
ollama pull llama3
ollama pull llava

# Ollama runs on http://localhost:11434 by default
```

## Pipeline Overview

The MiRAGE framework operates through a multi-stage pipeline:

```
+------------------------------------------------------------------+
|  STEP 1: Document Processing                                      |
|  PDF/HTML -> Markdown -> Semantic Chunks                          |
+--------------------------------+---------------------------------+
                                 |
                                 v
+------------------------------------------------------------------+
|  STEP 2: Embedding and Indexing                                   |
|  Embed all chunks -> Build unified FAISS index                    |
+--------------------------------+---------------------------------+
                                 |
                                 v
+------------------------------------------------------------------+
|  STEP 3: Domain and Expert Extraction                             |
|  BERTopic analysis -> LLM domain/role identification              |
+--------------------------------+---------------------------------+
                                 |
                                 v
+------------------------------------------------------------------+
|  STEP 4: QA Generation (per chunk, parallel)                      |
|  +--------------------------------------------------------------+ |
|  | 4.1 Verify chunk completeness                                | |
|  | 4.2 Multi-hop retrieval for incomplete chunks                | |
|  | 4.3 Generate QA pairs from complete context                  | |
|  | 4.4 Select high-quality pairs                                | |
|  | 4.5 Verify correctness and context necessity                 | |
|  | 4.6 Correct failed pairs (optional)                          | |
|  +--------------------------------------------------------------+ |
+--------------------------------+---------------------------------+
                                 |
                                 v
+------------------------------------------------------------------+
|  STEP 5: Hierarchical Deduplication                               |
|  Question clustering -> Answer sub-clustering -> LLM merging      |
+--------------------------------+---------------------------------+
                                 |
                                 v
+------------------------------------------------------------------+
|  STEP 6: Evaluation                                               |
|  RAGAS metrics + Custom metrics (faithfulness, relevancy, etc)    |
+------------------------------------------------------------------+
```

## Usage

### Full Pipeline

```bash
# Using the entry script
python run_mirage.py

# Or using the CLI (after pip install -e .)
mirage --config config.yaml
```

### Individual Components

```bash
# Preflight checks only
python run_mirage.py --preflight

# With custom config
python run_mirage.py --config my_config.yaml

# Skip preflight checks
python run_mirage.py --skip-preflight

# Verbose output
python run_mirage.py --verbose
```

### Programmatic Usage

```python
from mirage.core import call_llm_simple, setup_logging
from mirage.pipeline import build_complete_context, generate_qa_for_chunk
from mirage.embeddings import get_best_embedding_model

# Setup logging
setup_logging()

# Load embedding model
embedder = get_best_embedding_model()

# Generate QA for a chunk
qa_pairs = generate_qa_for_chunk(chunk, domain="Finance", expert="Financial Analyst")
```

## Output Format

### Sample Generated Question-Answer Pair

<p align="center">
  <img src="assets/ample question-answer pair generated.png" alt="Sample Generated QA Pair" width="100%">
</p>

### QA Dataset Structure (qa_deduplicated.json)

```json
[
  {
    "chunk_id": 1,
    "question": "What efficiency must a 75kW IE4 motor achieve?",
    "answer": "A 75kW IE4 motor must achieve 96.0% efficiency at 50Hz...",
    "context_chunks": [...],
    "hop_count": 2,
    "relevance_score": "9",
    "difficulty_score": "7",
    "expert_persona": "Motor Design Engineer",
    "domain": "Electrical Engineering"
  }
]
```

### Output Directory Structure

```
output/my_dataset/
├── markdown/              # Converted markdown files
├── chunks.json           # Semantic chunks
├── embeddings/           # FAISS index and embeddings
├── qa_dataset.json       # Raw QA pairs
├── qa_deduplicated.json  # Final deduplicated QA pairs
└── evaluation_report.json # Metrics and statistics
```

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Faithfulness** | Answer grounded in context |
| **Answer Relevancy** | Answer addresses the question |
| **Context Precision** | Retrieved chunks are relevant |
| **Context Recall** | Context contains reference info |
| **Multi-hop Reasoning** | Quality of multi-step reasoning |
| **Visual Dependency** | Requires image to answer |
| **Context Necessity** | Requires context (anti-parametric bias) |
| **Domain Coverage** | Corpus coverage |

## Hyperparameter Guide

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_depth` | 10 | Maximum retrieval iterations |
| `max_breadth` | 5 | Search queries per iteration |
| `chunks_per_search` | 2 | Chunks retrieved per query |
| `qa_max_workers` | 6 | Parallel workers for QA gen |
| `question_similarity_threshold` | 0.75 | Question clustering threshold |

### Recommended Settings

| Use Case | max_depth | max_breadth | chunks_per_search |
|----------|-----------|-------------|-------------------|
| **Quick Testing** | 2 | 2 | 1 |
| **Balanced (Default)** | 10 | 5 | 2 |
| **Thorough** | 20 | 10 | 3 |

## Contributing

Contributions are welcome! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Citation

If you use MiRAGE in your research, please cite:

```bibtex
@software{mirage2024,
  title = {MiRAGE: A Multiagent Framework for Generating Multimodal Multihop QA Datasets for RAG Evaluation},
  author = {MiRAGE Authors},
  year = {2024},
  url = {https://github.com/ChandanKSahu/MiRAGE}
}
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [RAGAS](https://github.com/explodinggradients/ragas) for evaluation metrics
- [BERTopic](https://github.com/MaartenGr/BERTopic) for topic modeling
- [Sentence Transformers](https://www.sbert.net/) for embeddings
- [FAISS](https://github.com/facebookresearch/faiss) for similarity search
- [Docling](https://github.com/DS4SD/docling) for PDF processing
