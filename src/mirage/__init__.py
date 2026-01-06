"""
MiRAGE: Multimodal Multihop RAG Evaluation Dataset Generator

A multi-agent framework for generating high-quality, multimodal, multihop
question-answer datasets for evaluating Retrieval-Augmented Generation (RAG) systems.
"""

__version__ = "1.0.2"
__author__ = "MiRAGE Authors"

# Core exports for easy access
from mirage.core.llm import (
    call_llm_simple,
    call_vlm_interweaved,
    call_vlm_with_multiple_images,
    batch_call_vlm_interweaved,
    setup_logging,
    BACKEND,
    LLM_MODEL_NAME,
    VLM_MODEL_NAME,
)

from mirage.core.config import load_config, get_config_value

from mirage.embeddings.models import (
    get_best_embedding_model,
    NomicVLEmbed,
)

from mirage.pipeline.qa_generator import generate_qa_for_chunk
from mirage.pipeline.context import build_complete_context
from mirage.pipeline.domain import fetch_domain_and_role
from mirage.pipeline.deduplication import deduplicate_qa_pairs

from mirage.utils.preflight import run_preflight_checks

__all__ = [
    # Version info
    "__version__",
    "__author__",
    # Core LLM functions
    "call_llm_simple",
    "call_vlm_interweaved",
    "call_vlm_with_multiple_images",
    "batch_call_vlm_interweaved",
    "setup_logging",
    "BACKEND",
    "LLM_MODEL_NAME",
    "VLM_MODEL_NAME",
    # Config
    "load_config",
    "get_config_value",
    # Embeddings
    "get_best_embedding_model",
    "NomicVLEmbed",
    # Pipeline
    "generate_qa_for_chunk",
    "build_complete_context",
    "fetch_domain_and_role",
    "deduplicate_qa_pairs",
    # Utils
    "run_preflight_checks",
]
