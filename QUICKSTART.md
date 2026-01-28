# MiRAGE Quick Start Guide

## Step 1: Install MiRAGE

```bash
pip install mirage-benchmark[pdf]
```

## Step 2: Set Up API Key

```bash
export GEMINI_API_KEY="your-gemini-api-key-here"
```

Get your API key from: https://makersuite.google.com/app/apikey

## Step 3: Prepare Your Documents

```bash
mkdir -p input output
# Copy your PDF files to the input directory
cp /path/to/your/*.pdf input/
```

## Step 4: Run MiRAGE

### Basic Usage (Generate 1 QA Pair)

```bash
run_mirage \
    --input input \
    --output output \
    --num-qa-pairs 1 \
    --max-depth 2
```

### With All Options

```bash
run_mirage \
    --input input \
    --output output \
    --backend gemini \
    --api-key YOUR_GEMINI_API_KEY \
    --num-qa-pairs 1 \
    --embedding-model auto \
    --reranker-model gemini_vlm \
    --max-depth 2 \
    --max-workers 4 \
    --verbose
```

### With Deduplication and Evaluation

```bash
run_mirage \
    --input input \
    --output output \
    --num-qa-pairs 10 \
    --deduplication \
    --evaluation
```

## Step 5: View Results

```bash
# List generated files
ls -lh output/

# Open visualization (auto-generated for first QA pair)
open output/multihop_visualization.html  # macOS
# xdg-open output/multihop_visualization.html  # Linux
# start output/multihop_visualization.html  # Windows
```

## Complete Example (Copy-Paste Ready)

```bash
# 1. Install
pip install mirage-benchmark[pdf]

# 2. Set API key
export GEMINI_API_KEY="your-gemini-api-key-here"

# 3. Create directories
mkdir -p input output

# 4. Add your PDFs to input/ directory
# cp /path/to/your/*.pdf input/

# 5. Run MiRAGE
run_mirage \
    --input input \
    --output output \
    --num-qa-pairs 1 \
    --max-depth 2 \
    --verbose

# 6. View results
ls -lh output/
open output/multihop_visualization.html
```

## Command Reference

| Flag | Description | Default |
|------|-------------|---------|
| `--input`, `-i` | Input directory with documents | Required |
| `--output`, `-o` | Output directory for results | Required |
| `--backend`, `-b` | Backend: gemini, openai, ollama | gemini |
| `--api-key`, `-k` | API key (or use env var) | From env |
| `--num-qa-pairs` | Number of QA pairs to generate | 100 |
| `--embedding-model` | Embedding model: auto, qwen3_vl, nomic, bge_m3 | auto |
| `--reranker-model` | Reranker: gemini_vlm, qwen3_vl, text_embedding | gemini_vlm |
| `--max-depth` | Max retrieval depth | 2 |
| `--max-workers` | Parallel workers | 4 |
| `--deduplication` | Enable deduplication | Off |
| `--evaluation` | Enable evaluation metrics | Off |
| `--verbose`, `-v` | Verbose output | Off |

## Troubleshooting

```bash
# Check installation
pip show mirage-benchmark

# Run preflight checks
run_mirage --preflight

# Check API key
echo $GEMINI_API_KEY
```

## Output Files

After running, you'll find in `output/`:
- `chunks.json` - Semantic chunks from your documents
- `qa_multihop_pass.json` - Generated QA pairs
- `multihop_visualization.html` - Interactive visualization (auto-generated)
- `run_config.json` - Run configuration
