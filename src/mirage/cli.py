#!/usr/bin/env python3
"""
MiRAGE Command Line Interface

Usage:
    mirage                      # Run full pipeline
    mirage --preflight          # Run preflight checks only
    mirage --config my.yaml     # Use custom config file
    mirage-preflight            # Run preflight checks (shortcut)
"""

import os
import sys
import argparse
import logging
import multiprocessing as mp


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="MiRAGE: Multimodal Multihop RAG Evaluation Dataset Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config.yaml",
        help="Path to configuration file (default: config.yaml)"
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Run preflight checks only"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        help="Input directory with documents (overrides config)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output directory for results (overrides config)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 1.3.0"
    )
    return parser.parse_args()


def main():
    """Main entry point for MiRAGE CLI."""
    args = parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Setup device environment (auto-detect GPU/CPU, suppress spurious warnings)
    from mirage.utils.device import setup_device_environment
    setup_device_environment()
    
    # Import after parsing to speed up --help
    from mirage.core.llm import setup_logging, BACKEND, LLM_MODEL_NAME, VLM_MODEL_NAME
    from mirage.utils.preflight import run_preflight_checks
    from mirage.core.config import load_config
    
    logger.info("=" * 60)
    logger.info("MiRAGE: Multimodal Multihop RAG Evaluation Dataset Generator")
    logger.info("=" * 60)
    logger.info(f"Backend: {BACKEND}")
    logger.info(f"LLM Model: {LLM_MODEL_NAME}")
    logger.info(f"VLM Model: {VLM_MODEL_NAME}")
    
    # Run preflight checks only mode
    if args.preflight:
        logger.info("\nRunning preflight checks...")
        all_passed, _ = run_preflight_checks()
        sys.exit(0 if all_passed else 1)
    
    # Note: Preflight checks run automatically in run_pipeline() and cannot be skipped
    
    # Load configuration
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {args.config}")
        logger.info("Create config.yaml from config.yaml.example:")
        logger.info("  cp config.yaml.example config.yaml")
        sys.exit(1)
    
    paths = config.get('paths', {})
    input_dir = args.input or paths.get('input_pdf_dir', 'data/documents')
    output_dir = args.output or paths.get('output_dir', 'output/results')
    
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    
    # Validate input directory
    if not os.path.exists(input_dir):
        logger.error(f"Input directory does not exist: {input_dir}")
        logger.info("Add your documents to the data/documents/ folder")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Run pipeline
    logger.info("\nStarting MiRAGE pipeline...")
    logger.info("See README.md for detailed pipeline documentation.\n")
    
    # Import pipeline modules
    from mirage.pipeline.pdf_processor import process_directory as process_pdfs
    from mirage.pipeline.chunker import process_markdown_directory
    from mirage.pipeline.domain import fetch_domain_and_role
    from mirage.pipeline.qa_generator import run_qa_generation
    from mirage.pipeline.deduplication import deduplicate_qa_dataset
    
    # Execute pipeline steps
    # (The actual implementation would go here)
    
    logger.info("\n" + "=" * 60)
    logger.info("Pipeline complete!")
    logger.info("=" * 60)
    logger.info(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    # Use spawn method for multiprocessing (required for CUDA)
    mp.set_start_method('spawn', force=True)
    main()
