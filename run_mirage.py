#!/usr/bin/env python3
"""
MiRAGE - Multimodal Multihop RAG Evaluation Dataset Generator

This script runs the complete MiRAGE pipeline for generating multimodal
multihop QA datasets from your documents.

Usage:
    # Basic usage with default config
    python run_mirage.py --input data/documents --output output/my_dataset

    # With API key
    python run_mirage.py --input data/documents --output output/my_dataset --api-key YOUR_KEY

    # With custom config file
    python run_mirage.py --input data/documents --output output/my_dataset --config my_config.yaml

    # Preflight checks only
    python run_mirage.py --preflight

Examples:
    # Using Gemini (default)
    export GEMINI_API_KEY="your-gemini-key"
    python run_mirage.py -i data/documents -o output/results

    # Using OpenAI
    export OPENAI_API_KEY="your-openai-key"
    python run_mirage.py -i data/documents -o output/results --backend openai

    # Using local Ollama
    python run_mirage.py -i data/documents -o output/results --backend ollama
"""

import os
import sys
import argparse
import logging
import json
from pathlib import Path
from typing import Optional
import multiprocessing as mp

# Add src to path for package imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="MiRAGE: Multimodal Multihop RAG Evaluation Dataset Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (uses GEMINI_API_KEY from environment by default)
  python run_mirage.py -i data/documents -o output/my_dataset

  # With Gemini API key (explicitly specify backend)
  python run_mirage.py -i data/documents -o output/my_dataset --backend gemini --api-key YOUR_GEMINI_KEY

  # With OpenAI (MUST specify --backend and --api-key)
  python run_mirage.py -i data/documents -o output/my_dataset --backend openai --api-key YOUR_OPENAI_KEY

  # With local Ollama (MUST specify --backend, no API key needed)
  python run_mirage.py -i data/documents -o output/my_dataset --backend ollama

  # Run preflight checks
  python run_mirage.py --preflight

  # Enable optional steps (deduplication and evaluation)
  python run_mirage.py -i data/documents -o output/my_dataset --deduplication --evaluation

Pipeline Steps:
  1. PDF/HTML to Markdown conversion (mandatory)
  2. Semantic chunking (mandatory)
  3. Embedding and indexing (mandatory)
  4. Domain/expert detection (mandatory)
  5. QA generation and verification (mandatory)
  6. Deduplication (OFF by default - enable with --deduplication)
  7. Evaluation metrics (OFF by default - enable with --evaluation)

Environment Variables:
  GEMINI_API_KEY    - Google Gemini API key
  OPENAI_API_KEY    - OpenAI API key
  MIRAGE_CONFIG     - Path to config file (default: config.yaml)
"""
    )
    
    # Required arguments
    parser.add_argument(
        "--input", "-i",
        type=str,
        help="Input directory containing documents (PDF, HTML, etc.)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output directory for generated QA dataset"
    )
    
    # API and backend configuration
    parser.add_argument(
        "--backend", "-b",
        type=str,
        choices=["gemini", "openai", "ollama"],
        default="gemini",
        help="LLM backend to use: gemini (default), openai, or ollama"
    )
    parser.add_argument(
        "--api-key", "-k",
        type=str,
        help="API key for the selected backend. Use with --backend to specify which service."
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Model name to use (default depends on backend)"
    )
    
    # Configuration
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config.yaml",
        help="Path to configuration file (default: config.yaml)"
    )
    
    # Pipeline options
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Run preflight checks only"
    )
    parser.add_argument(
        "--skip-pdf-processing",
        action="store_true",
        help="Skip PDF to Markdown conversion (use existing markdown)"
    )
    parser.add_argument(
        "--skip-chunking",
        action="store_true",
        help="Skip chunking (use existing chunks.json)"
    )
    parser.add_argument(
        "--deduplication",
        action="store_true",
        help="Enable QA deduplication step (off by default)"
    )
    parser.add_argument(
        "--evaluation",
        action="store_true",
        help="Enable evaluation metrics computation (off by default)"
    )
    
    # QA generation options
    parser.add_argument(
        "--num-qa-pairs",
        type=int,
        default=100,
        help="Target number of QA pairs to generate (default: 100)"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)"
    )
    
    # Output options
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 1.2.0"
    )
    
    return parser.parse_args()


def setup_environment(args):
    """Set up environment variables based on arguments."""
    # Set API key if provided
    if args.api_key:
        if args.backend == "gemini":
            os.environ["GEMINI_API_KEY"] = args.api_key
        elif args.backend == "openai":
            os.environ["OPENAI_API_KEY"] = args.api_key
    
    # Set backend
    os.environ["MIRAGE_BACKEND"] = args.backend.upper()
    
    # Set model if provided
    if args.model:
        os.environ["MIRAGE_MODEL"] = args.model
    
    # Set input/output directories (override config.yaml)
    if args.input:
        os.environ["MIRAGE_INPUT_DIR"] = args.input
    if args.output:
        os.environ["MIRAGE_OUTPUT_DIR"] = args.output
    
    # Set QA generation parameters
    if args.num_qa_pairs:
        os.environ["MIRAGE_NUM_QA_PAIRS"] = str(args.num_qa_pairs)
    if args.max_workers:
        os.environ["MIRAGE_MAX_WORKERS"] = str(args.max_workers)
    
    # Set optional pipeline flags (off by default, enable with flags)
    if args.deduplication:
        os.environ["MIRAGE_RUN_DEDUPLICATION"] = "1"
    if args.evaluation:
        os.environ["MIRAGE_RUN_EVALUATION"] = "1"


def validate_args(args):
    """Validate command line arguments."""
    errors = []
    
    # Check input directory
    if args.input and not os.path.exists(args.input):
        errors.append(f"Input directory does not exist: {args.input}")
    
    # Check API key for non-Ollama backends
    if not args.preflight and args.backend != "ollama":
        api_key = args.api_key
        if args.backend == "gemini":
            api_key = api_key or os.environ.get("GEMINI_API_KEY")
        elif args.backend == "openai":
            api_key = api_key or os.environ.get("OPENAI_API_KEY")
        
        if not api_key:
            env_var = "GEMINI_API_KEY" if args.backend == "gemini" else "OPENAI_API_KEY"
            errors.append(
                f"API key required for {args.backend} backend. "
                f"Use --api-key or set {env_var} environment variable."
            )
    
    return errors


def main():
    """Main entry point for MiRAGE pipeline."""
    args = parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Print banner
    logger.info("=" * 70)
    logger.info("  MiRAGE: Multimodal Multihop RAG Evaluation Dataset Generator")
    logger.info("=" * 70)
    
    # Setup environment
    setup_environment(args)
    
    # Import after environment setup
    try:
        from mirage.utils.preflight import run_preflight_checks
        from mirage.main import run_pipeline
    except ImportError as e:
        logger.error(f"Failed to import mirage package: {e}")
        logger.info("Make sure to install the package: pip install -e .")
        sys.exit(1)
    
    # Run preflight checks only
    if args.preflight:
        logger.info("\nRunning preflight checks...")
        success = run_preflight_checks()
        sys.exit(0 if success else 1)
    
    # Validate arguments
    if not args.input or not args.output:
        logger.error("Both --input and --output are required.")
        logger.info("Usage: python run_mirage.py --input <input_dir> --output <output_dir>")
        logger.info("Run 'python run_mirage.py --help' for more information.")
        sys.exit(1)
    
    errors = validate_args(args)
    if errors:
        for error in errors:
            logger.error(error)
        sys.exit(1)
    
    # Log configuration
    logger.info(f"Backend: {args.backend.upper()}")
    logger.info(f"Input directory: {args.input}")
    logger.info(f"Output directory: {args.output}")
    logger.info(f"Target QA pairs: {args.num_qa_pairs}")
    logger.info(f"Workers: {args.max_workers}")
    logger.info(f"Deduplication: {'ENABLED' if args.deduplication else 'OFF (use --deduplication to enable)'}")
    logger.info(f"Evaluation: {'ENABLED' if args.evaluation else 'OFF (use --evaluation to enable)'}")
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Save run configuration
    config = {
        "input_dir": args.input,
        "output_dir": args.output,
        "backend": args.backend,
        "model": args.model,
        "num_qa_pairs": args.num_qa_pairs,
        "max_workers": args.max_workers,
    }
    config_path = os.path.join(args.output, "run_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    logger.info(f"Configuration saved to: {config_path}")
    
    # Run the full pipeline (preflight checks are mandatory and run first)
    logger.info("\nStarting MiRAGE pipeline...")
    run_pipeline()  # Preflight runs automatically, completed steps are skipped
    
    logger.info("\nPipeline completed!")


if __name__ == "__main__":
    # Use spawn method for multiprocessing (required for CUDA)
    mp.set_start_method('spawn', force=True)
    main()
