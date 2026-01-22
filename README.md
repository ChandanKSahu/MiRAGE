# MiRAGE: A Multiagent Framework for Generating Multimodal Multihop Question-Answer Dataset for RAG Evaluation

<p align="center">
  <img src="https://img.shields.io/badge/python-3.9+-blue.svg" alt="Python 3.9+">
  <img src="https://img.shields.io/badge/license-Apache%202.0-green.svg" alt="License">
  <img src="https://img.shields.io/pypi/v/mirage-benchmark.svg" alt="PyPI">
</p>

**MiRAGE** is a multi-agent framework for generating high-quality, multimodal, multihop question-answer datasets for evaluating Retrieval-Augmented Generation (RAG) systems.

### Multiagent Architecture

<p align="center">
  <img src="assets/mirage_framework.png" alt="MiRAGE Framework Architecture" width="100%">
</p>

### Sample QA Pair

<p align="center">
  <img src="assets/ample question-answer pair generated.png" alt="Sample QA Pair Generated" width="100%">
</p>

### Interactive Process Flow

Explore the step-by-step multihop QA generation process:

**[🔗 View Interactive Visualization](https://htmlpreview.github.io/?https://github.com/ChandanKSahu/MiRAGE/blob/main/assets/mirage_qa_gen.html)**

## Key Features

- **Modular Pipeline Architecture**: Configurable modules with simple `process()` interface - use individual components or chain them together
- **Multi-hop Context Completion**: Iteratively expands incomplete chunks with relevant context
- **Domain and Expert Role Detection**: Automatic domain identification using BERTopic + LLM
- **Multi-stage QA Pipeline**: Generate, Select, Verify, Correct for quality assurance
- **Multimodal Support**: Handles text, tables, figures, and images
- **Multiple Backend Support**: Gemini, OpenAI, and local Ollama models
- **Fully Parallelized**: Thread and process pools for maximum throughput
- **Token Usage Tracking**: Automatic tracking of input/output tokens across all LLM calls
- **Checkpoint & Resume**: Interrupt and resume long-running pipelines without losing progress
- **Comprehensive Hyperparameters**: All documented parameters exposed with sensible defaults

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [API Keys Setup](#api-keys-setup)
- [Configuration](#configuration)
- [Command Line Options](#command-line-options)
- [Output Format](#output-format)
- [Project Structure](#project-structure)
- [Python API](#python-api)
  - [Modular Pipeline](#modular-pipeline-recommended)
  - [Using the Pipeline Class](#using-the-pipeline-class)
  - [Module Reference](#module-reference)
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
pip install mirage-benchmark[pdf]   # PDF processing (docling, matplotlib)
pip install mirage-benchmark[eval]  # Evaluation metrics (ragas)
pip install mirage-benchmark[all]   # All pip dependencies
```

### GPU Support (FAISS-GPU)

For GPU-accelerated similarity search, install FAISS-GPU via conda:

```bash
# Create conda environment (recommended)
conda create -n mirage python=3.11
conda activate mirage

# Install FAISS-GPU
conda install -c pytorch faiss-gpu

# Then install MiRAGE
pip install mirage-benchmark[gpu]
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
# Using Gemini (default backend) - API key from environment
export GEMINI_API_KEY="your-gemini-key"
python run_mirage.py --input data/my_documents --output output/my_dataset

# Using Gemini with API key as argument
python run_mirage.py -i data/my_documents -o output/my_dataset --backend gemini --api-key YOUR_GEMINI_KEY

# Using OpenAI
python run_mirage.py -i data/my_documents -o output/my_dataset --backend openai --api-key YOUR_OPENAI_KEY

# Using local Ollama (no API key needed)
python run_mirage.py -i data/my_documents -o output/my_dataset --backend ollama
```

**Note**: When using `--api-key`, always specify `--backend` to indicate which service the key is for.

### Step 4: Check Results

```bash
ls output/my_dataset/
# qa_multihop_pass.json  - Generated QA pairs (always created)
# chunks.json            - Semantic chunks (always created)

# Optional outputs (if --deduplication and --evaluation flags used):
# qa_deduplicated.json   - Deduplicated QA pairs (with --deduplication)
# evaluation_report.json - Quality metrics (with --evaluation)
```

## Usage

### Basic Usage (QA Generation Only)

By default, MiRAGE runs the core pipeline: document processing, chunking, embedding, and QA generation/verification. **Deduplication and evaluation are OFF by default.**

```bash
# Default: Generates QA pairs without deduplication or evaluation
python run_mirage.py --input <INPUT_DIR> --output <OUTPUT_DIR>
```

### With Deduplication

To merge similar QA pairs and remove duplicates:

```bash
python run_mirage.py -i data/documents -o output/results --deduplication
```

### With Evaluation Metrics

To compute quality metrics (faithfulness, relevancy, etc.):

```bash
python run_mirage.py -i data/documents -o output/results --evaluation
```

### Full Pipeline (Deduplication + Evaluation)

```bash
python run_mirage.py -i data/documents -o output/results --deduplication --evaluation
```

### With All Options

```bash
python run_mirage.py \
    --input data/documents \
    --output output/results \
    --backend gemini \
    --api-key YOUR_GEMINI_KEY \
    --num-qa-pairs 100 \
    --max-workers 4 \
    --deduplication \
    --evaluation \
    --verbose
```

**Backend Options:**
- `gemini` (default) - Requires `GEMINI_API_KEY` or `--api-key`
- `openai` - Requires `OPENAI_API_KEY` or `--api-key`
- `ollama` - No API key needed (runs locally)

**Pipeline Steps:**
| Step | Description | Default |
|------|-------------|---------|
| 1. Document Processing | PDF/HTML to Markdown | **Mandatory** |
| 2. Chunking | Semantic chunking | **Mandatory** |
| 3. Embedding | FAISS index creation | **Mandatory** |
| 4. Domain Detection | Expert persona extraction | **Mandatory** |
| 5. QA Generation | Multi-hop QA with verification | **Mandatory** |
| 6. Deduplication | Merge similar QA pairs | OFF (use `--deduplication`) |
| 7. Evaluation | Quality metrics | OFF (use `--evaluation`) |

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

## ⚠️ Cost Optimization

MiRAGE uses LLM/VLM APIs extensively. Two operations consume the most tokens:

### 1. Document Processing (PDF/HTML → Markdown → Chunks)

**Cost:** High (processes every page with VLM for image/table extraction)

**Recommendation:**
- Only process documents **once** on a curated set of relevant files
- Use `--skip-pdf-processing` and `--skip-chunking` on subsequent runs
- Pre-filter documents to remove irrelevant content before running MiRAGE

```bash
# First run: Process and chunk documents
python run_mirage.py -i data/documents -o output/results

# Subsequent runs: Skip processing, only generate QA
python run_mirage.py -i data/documents -o output/results --skip-pdf-processing --skip-chunking
```

### 2. Multi-hop Context Building

**Cost:** High (recursive LLM calls to expand context at each depth level)

**Recommendation:**
- Default is now `max_depth: 2` (previously 5)
- Higher depths exponentially increase token usage with diminishing returns
- Depth 2 captures most meaningful cross-document relationships

```yaml
# config.yaml
context:
  max_depth: 2  # Recommended: 2 (default: 5)
```

Use `print_token_stats()` or check the pipeline summary to monitor actual token consumption.

## Command Line Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--input` | `-i` | Input directory with documents | Required |
| `--output` | `-o` | Output directory for results | Required |
| `--api-key` | `-k` | API key for LLM backend | From env |
| `--backend` | `-b` | Backend: gemini, openai, ollama | gemini |
| `--model` | | Model name | Auto |
| `--config` | `-c` | Config file path | config.yaml |
| `--num-qa-pairs` | | Target QA pairs to generate | 10 |
| `--max-workers` | | Parallel workers | 4 |
| `--preflight` | | Run preflight checks only | - |
| `--skip-preflight` | | Skip preflight checks | - |
| `--skip-pdf-processing` | | Skip PDF conversion | - |
| `--skip-chunking` | | Skip chunking step | - |
| `--verbose` | `-v` | Verbose output | - |
| `--version` | | Show version | - |
| `--help` | `-h` | Show help | - |

### Multihop QA Visualization

Explore an interactive visualization of the multihop QA generation process, showing how context chunks are linked through keywords to generate complex questions:

**[🔗 View Interactive Multihop QA Visualization](https://htmlpreview.github.io/?https://github.com/ChandanKSahu/MiRAGE/blob/main/assets/mirage_qa_gen.html)**

The visualization demonstrates:
- Context chunk retrieval and keyword extraction
- Keyword chain relationships across chunks
- Iterative retrieval depth progression
- Final question-answer generation with highlighted concepts

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

### Multihop QA Visualization

See the [Interactive Process Flow](#interactive-process-flow) at the top of this page for a step-by-step visualization showing:
- Context chunk retrieval and keyword extraction
- Keyword chain relationships across chunks
- Iterative retrieval depth progression
- Final question-answer generation with highlighted concepts

## Project Structure

```
MiRAGE/
├── src/mirage/                    # Main package
│   ├── __init__.py               # Package initialization
│   ├── main.py                   # Pipeline orchestration
│   ├── cli.py                    # Command-line interface
│   ├── core/                     # Core functionality
│   │   ├── base.py               # Modular pipeline base classes (BaseModule, Pipeline)
│   │   ├── config.py             # Configuration management + model registries
│   │   ├── llm.py                # LLM/VLM API interfaces + token tracking
│   │   └── prompts.py            # Prompt templates
│   ├── embeddings/               # Embedding models
│   │   ├── models.py             # EmbeddingModule + multimodal embedders
│   │   ├── rerankers_multimodal.py  # RerankerModule + VLM-based reranking
│   │   └── rerankers_text.py     # Text-based reranking
│   ├── pipeline/                 # Processing pipeline modules
│   │   ├── pdf_processor.py      # DocumentProcessor - PDF/HTML to Markdown
│   │   ├── chunker.py            # SemanticChunker - Semantic chunking
│   │   ├── context.py            # ContextBuilder - Multi-hop context retrieval
│   │   ├── qa_generator.py       # QAGenerator - QA generation and verification
│   │   ├── domain.py             # DomainExtractor - Domain/expert extraction
│   │   └── deduplication.py      # Deduplicator - QA deduplication
│   ├── evaluation/               # Evaluation metrics
│   │   ├── metrics.py            # Standard RAGAS metrics
│   │   └── metrics_optimized.py  # Evaluator - Optimized metrics module
│   └── utils/                    # Utilities
│       ├── preflight.py          # System checks
│       ├── stats.py              # Dataset statistics
│       ├── ablation.py           # Ablation studies
│       ├── checkpoint.py         # Checkpoint/resume support
│       ├── llm_cache.py          # LLM response caching
│       ├── visualize_multihop.py # Multihop QA visualization
│       └── visualize_pipeline.py # Pipeline flow visualization
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

## Python API

MiRAGE provides a modular pipeline architecture where each component can be used independently or chained together.

### Modular Pipeline (Recommended)

Each module has a simple `process()` interface with comprehensive hyperparameters:

```python
from mirage.pipeline.pdf_processor import DocumentProcessor
from mirage.pipeline.chunker import SemanticChunker
from mirage.embeddings.models import EmbeddingModule
from mirage.pipeline.domain import DomainExtractor
from mirage.pipeline.context import ContextBuilder
from mirage.pipeline.qa_generator import QAGenerator
from mirage.pipeline.deduplication import Deduplicator
from mirage.evaluation.metrics_optimized import Evaluator

# Step 1: Process documents to markdown
doc_processor = DocumentProcessor(
    annotation_model='gemini-2.5-flash',
    image_resolution_scale=2.0,
    do_ocr=True,
    cuda_device_id=0
)
markdown_files = doc_processor.process('data/documents/')

# Step 2: Chunk markdown into semantic units
chunker = SemanticChunker(
    window_size=20000,      # ~5000 tokens
    overlap_size=2000,      # ~500 tokens
    llm_model='gpt-oss-120b'
)
chunks = chunker.process(markdown_files)

# Step 3: Generate embeddings
embedder = EmbeddingModule(
    model='nomic',          # nomic, bge_m3, bge_large, minilm
    batch_size=16,
    normalize=True
)
embeddings = embedder.process(chunks)
faiss_index = embedder.build_index()

# Step 4: Extract domain and expert persona
domain_extractor = DomainExtractor(
    use_multimodal_embeddings=True,
    llm_model='gpt-oss-120b'
)
domain_result = domain_extractor.process(chunks, embeddings=embeddings)
print(f"Domain: {domain_result['domain']}")
print(f"Expert: {domain_result['expert_role']}")

# Step 5: Generate QA pairs
qa_generator = QAGenerator(
    vlm_model='qwen2.5vl:32b',
    max_depth=3,
    max_breadth=3,
    enable_correction=True
)
qa_result = qa_generator.process(
    chunks, 
    domain=domain_result['domain'],
    expert_persona=domain_result['expert_role']
)
print(f"Generated {len(qa_result['successful'])} QA pairs")

# Step 6: Deduplicate
deduplicator = Deduplicator(
    question_similarity_threshold=0.75,
    alpha=0.6  # weight for semantic vs chunk overlap
)
dedup_result = deduplicator.process(qa_result['successful'])
print(f"After dedup: {len(dedup_result['deduplicated'])} QA pairs")

# Step 7: Evaluate quality
evaluator = Evaluator(
    sample_size=50,
    use_gemini=True,
    run_faithfulness=True,
    run_context_necessity=True
)
eval_result = evaluator.process(dedup_result['deduplicated'])
print(f"Faithfulness: {eval_result['summary'].get('faithfulness_mean', 'N/A')}")
```

### Using the Pipeline Class

Chain modules together for cleaner code:

```python
from mirage.core.base import Pipeline
from mirage.pipeline.pdf_processor import DocumentProcessor
from mirage.pipeline.chunker import SemanticChunker
from mirage.embeddings.models import EmbeddingModule

# Create pipeline
pipeline = Pipeline([
    ('docs', DocumentProcessor(annotation_model='gemini-2.5-flash')),
    ('chunks', SemanticChunker(window_size=20000)),
    ('embed', EmbeddingModule(model='nomic', batch_size=16)),
])

# Run entire pipeline
result = pipeline.run('data/documents/')

# Or run partial pipeline
chunks = pipeline.run('data/documents/', stop_after='chunks')

# Access intermediate outputs
embeddings = pipeline.get_output('embed')
```

### Module Reference

| Module | Key Parameters | Description |
|--------|---------------|-------------|
| `DocumentProcessor` | `annotation_model`, `image_resolution_scale`, `do_ocr` | PDF/HTML → Markdown |
| `SemanticChunker` | `window_size`, `overlap_size`, `llm_model` | Semantic chunking |
| `EmbeddingModule` | `model`, `batch_size`, `normalize`, `load_in_4bit` | Generate embeddings |
| `RerankerModule` | `model`, `top_k` | Chunk reranking |
| `DomainExtractor` | `use_multimodal_embeddings`, `llm_model` | Domain/expert extraction |
| `ContextBuilder` | `max_depth`, `max_breadth`, `retrieval_k` | Multi-hop context |
| `QAGenerator` | `vlm_model`, `enable_correction`, `max_depth` | QA generation |
| `Deduplicator` | `question_similarity_threshold`, `alpha` | Remove duplicates |
| `Evaluator` | `sample_size`, `use_gemini`, `run_*` flags | Quality metrics |

### Legacy API (Still Supported)

```python
# Import the main pipeline
from mirage import run_pipeline

# Or import specific components
from mirage.core.llm import call_llm_simple, call_vlm_interweaved
from mirage.pipeline.context import build_complete_context
from mirage.pipeline.qa_generator import generate_qa, verify_qa
from mirage.pipeline.domain import fetch_domain_and_role
from mirage.embeddings.models import NomicVLEmbed, get_best_embedding_model
from mirage.utils.preflight import run_preflight_checks

# Example: Run preflight checks
success, results = run_preflight_checks()

# Example: Call LLM
response = call_llm_simple("What is 2+2?")

# Example: Use embedding model
embedder = NomicVLEmbed()
embedding = embedder.encode("Sample text")

# Example: Track token usage
from mirage.core.llm import get_token_stats, print_token_stats, reset_token_stats

# After running LLM calls, check token usage
stats = get_token_stats()
print(f"Input tokens: {stats['total_input_tokens']}")
print(f"Output tokens: {stats['total_output_tokens']}")

# Print formatted summary
print_token_stats()

# Reset counters for a new run
reset_token_stats()
```

See the module docstrings for detailed API documentation.

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




