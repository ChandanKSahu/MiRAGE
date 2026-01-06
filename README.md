# MiRAGE: A Multiagent Framework for Generating Multimodal Multihop Question-Answer Dataset for RAG Evaluation

<p align="center">
  <img src="https://img.shields.io/badge/python-3.9+-blue.svg" alt="Python 3.9+">
  <img src="https://img.shields.io/badge/license-Apache%202.0-green.svg" alt="License">
  <img src="https://img.shields.io/pypi/v/mirage-benchmark.svg" alt="PyPI">
</p>

**MiRAGE** is a multi-agent framework for generating high-quality, multimodal, multihop question-answer datasets for evaluating Retrieval-Augmented Generation (RAG) systems.

<p align="center">
  <img src="assets/mirage_framework.png" alt="MiRAGE Framework Architecture" width="100%">
</p>

## Key Features

- **Multi-hop Context Completion**: Iteratively expands incomplete chunks with relevant context
- **Domain and Expert Role Detection**: Automatic domain identification using BERTopic + LLM
- **Multi-stage QA Pipeline**: Generate, Select, Verify, Correct for quality assurance
- **Multimodal Support**: Handles text, tables, figures, and images
- **Multiple Backend Support**: Gemini, OpenAI, and local Ollama models
- **Fully Parallelized**: Thread and process pools for maximum throughput

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [API Keys Setup](#api-keys-setup)
- [Configuration](#configuration)
- [Command Line Options](#command-line-options)
- [Output Format](#output-format)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Installation

### From PyPI

```bash
pip install mirage-benchmark
```

### From Source

```bash
git clone https://github.com/ChandanKSahu/MiRAGE.git
cd MiRAGE
pip install -e .
```

### With Optional Dependencies

```bash
pip install mirage-benchmark[gpu]   # GPU support
pip install mirage-benchmark[pdf]   # PDF processing
pip install mirage-benchmark[all]   # All dependencies
```

## Quick Start

### Step 1: Set Up API Key

Choose one of the following backends:

**Option A: Google Gemini (Recommended)**
```bash
export GEMINI_API_KEY="your-gemini-api-key"
```

**Option B: OpenAI**
```bash
export OPENAI_API_KEY="your-openai-api-key"
```

**Option C: Local Ollama (No API key needed)**
```bash
# Install and start Ollama
ollama serve
ollama pull llama3
```

### Step 2: Prepare Your Data

Place your documents in a folder:
```bash
mkdir -p data/my_documents
cp /path/to/your/*.pdf data/my_documents/
```

### Step 3: Run MiRAGE

```bash
# Basic usage
python run_mirage.py --input data/my_documents --output output/my_dataset

# With API key as argument
python run_mirage.py -i data/my_documents -o output/my_dataset --api-key YOUR_API_KEY

# Using OpenAI
python run_mirage.py -i data/my_documents -o output/my_dataset --backend openai

# Using local Ollama
python run_mirage.py -i data/my_documents -o output/my_dataset --backend ollama
```

### Step 4: Check Results

```bash
ls output/my_dataset/
# qa_deduplicated.json  - Final QA dataset
# chunks.json           - Semantic chunks
# evaluation_report.json - Quality metrics
```

## Usage

### Basic Usage

```bash
python run_mirage.py --input <INPUT_DIR> --output <OUTPUT_DIR>
```

### With All Options

```bash
python run_mirage.py \
    --input data/documents \
    --output output/results \
    --backend gemini \
    --api-key YOUR_API_KEY \
    --num-qa-pairs 100 \
    --max-workers 4 \
    --verbose
```

### Run Preflight Checks

Before running the full pipeline, verify your setup:

```bash
python run_mirage.py --preflight
```

### Using Sample Dataset

A sample dataset is included for testing:

```bash
# Unzip sample data
unzip data/FinanceAnnualReports.zip -d data/sample/

# Run on sample
python run_mirage.py -i data/sample -o output/sample_results
```

## API Keys Setup

### Google Gemini

1. Get API key from: https://makersuite.google.com/app/apikey
2. Set environment variable:
```bash
export GEMINI_API_KEY="your-key-here"
```

Or create a file:
```bash
mkdir -p ~/.config/gemini
echo "your-key-here" > ~/.config/gemini/api_key.txt
```

### OpenAI

1. Get API key from: https://platform.openai.com/api-keys
2. Set environment variable:
```bash
export OPENAI_API_KEY="your-key-here"
```

### Ollama (Local - Free)

No API key needed! Just install Ollama:

```bash
# Install
curl -fsSL https://ollama.com/install.sh | sh

# Start server
ollama serve

# Pull models
ollama pull llama3      # For text
ollama pull llava       # For vision
```

## Configuration

### Using config.yaml

Copy the example config and customize:

```bash
cp config.yaml.example config.yaml
```

Edit `config.yaml`:

```yaml
backend:
  active: GEMINI  # GEMINI, OPENAI, or OLLAMA
  
  gemini:
    api_key_path: ~/.config/gemini/api_key.txt
    llm_model: gemini-2.0-flash
    vlm_model: gemini-2.0-flash
    
  openai:
    api_key_path: ~/.config/openai/api_key.txt
    llm_model: gpt-4o
    vlm_model: gpt-4o
    
  ollama:
    base_url: http://localhost:11434
    llm_model: llama3
    vlm_model: llava

paths:
  input_pdf_dir: data/documents
  output_dir: output/results

qa_generation:
  target_qa_pairs: 100
  max_workers: 4
```

Then run:
```bash
python run_mirage.py --config config.yaml
```

## Command Line Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--input` | `-i` | Input directory with documents | Required |
| `--output` | `-o` | Output directory for results | Required |
| `--api-key` | `-k` | API key for LLM backend | From env |
| `--backend` | `-b` | Backend: gemini, openai, ollama | gemini |
| `--model` | | Model name | Auto |
| `--config` | `-c` | Config file path | config.yaml |
| `--num-qa-pairs` | | Target QA pairs to generate | 100 |
| `--max-workers` | | Parallel workers | 4 |
| `--preflight` | | Run preflight checks only | - |
| `--skip-preflight` | | Skip preflight checks | - |
| `--skip-pdf-processing` | | Skip PDF conversion | - |
| `--skip-chunking` | | Skip chunking step | - |
| `--verbose` | `-v` | Verbose output | - |
| `--version` | | Show version | - |
| `--help` | `-h` | Show help | - |

## Output Format

### Generated Files

```
output/my_dataset/
├── markdown/              # Converted markdown files
├── chunks.json           # Semantic chunks
├── qa_dataset.json       # Raw QA pairs
├── qa_deduplicated.json  # Final deduplicated QA pairs
├── evaluation_report.json # Quality metrics
└── run_config.json       # Run configuration
```

### QA Dataset Structure

```json
{
  "chunk_id": 1,
  "question": "What is the company's revenue growth?",
  "answer": "The company achieved 15% revenue growth...",
  "context_chunks": [...],
  "hop_count": 2,
  "relevance_score": "9",
  "difficulty_score": "7",
  "expert_persona": "Financial Analyst",
  "domain": "Finance"
}
```

<p align="center">
  <img src="assets/ample question-answer pair generated.png" alt="Sample QA Pair" width="100%">
</p>

## Project Structure

```
MiRAGE/
├── src/mirage/              # Main package
│   ├── core/               # LLM interfaces, prompts, config
│   ├── embeddings/         # Embedding models, rerankers
│   ├── pipeline/           # PDF processing, QA generation
│   ├── evaluation/         # Metrics
│   └── utils/              # Utilities
├── data/                   # Your documents
│   └── documents/         # Input folder
├── output/                 # Generated results
├── config.yaml.example     # Example configuration
├── run_mirage.py          # Main entry point
└── README.md
```

## Examples

### Generate QA from PDFs

```bash
# Using Gemini
export GEMINI_API_KEY="your-key"
python run_mirage.py -i data/pdfs -o output/qa_dataset

# Using OpenAI  
export OPENAI_API_KEY="your-key"
python run_mirage.py -i data/pdfs -o output/qa_dataset --backend openai

# Using Ollama (local, free)
python run_mirage.py -i data/pdfs -o output/qa_dataset --backend ollama
```

### Generate More QA Pairs

```bash
python run_mirage.py -i data/documents -o output/large_dataset --num-qa-pairs 500
```

### Use More Workers

```bash
python run_mirage.py -i data/documents -o output/fast_run --max-workers 8
```

### Skip Already Processed Steps

```bash
# If you already have markdown files
python run_mirage.py -i data/documents -o output/results --skip-pdf-processing

# If you already have chunks
python run_mirage.py -i data/documents -o output/results --skip-chunking
```

## Troubleshooting

### API Key Issues

```bash
# Check if API key is set
echo $GEMINI_API_KEY

# Set it if missing
export GEMINI_API_KEY="your-key"
```

### Import Errors

```bash
# Reinstall package
pip install -e .
```

### Preflight Check Failures

```bash
# Run verbose preflight
python run_mirage.py --preflight --verbose
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## Citation

```bibtex
@software{mirage2024,
  title = {MiRAGE: A Multiagent Framework for Generating Multimodal Multihop Question-Answer Dataset for RAG Evaluation},
  author = {MiRAGE Authors},
  year = {2026},
  url = {https://github.com/ChandanKSahu/MiRAGE}
}
```

## License

Apache License 2.0 - see [LICENSE](LICENSE)

## Acknowledgments

- [RAGAS](https://github.com/explodinggradients/ragas) for evaluation metrics
- [BERTopic](https://github.com/MaartenGr/BERTopic) for topic modeling
- [FAISS](https://github.com/facebookresearch/faiss) for similarity search
- [Docling](https://github.com/DS4SD/docling) for PDF processing


