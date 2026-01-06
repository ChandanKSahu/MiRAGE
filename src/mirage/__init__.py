"""
MiRAGE: Multimodal Multihop RAG Evaluation Dataset Generator

A multi-agent framework for generating high-quality, multimodal, multihop
question-answer datasets for evaluating Retrieval-Augmented Generation (RAG) systems.
"""

__version__ = "1.0.2"
__author__ = "MiRAGE Authors"


def __getattr__(name):
    """Lazy import of submodules to avoid import-time config loading.
    
    This allows `from mirage import __version__` to work without a config file,
    while still providing convenient access to submodules when needed.
    """
    # Core LLM functions - lazy import
    if name in ("call_llm_simple", "call_vlm_interweaved", "call_vlm_with_multiple_images",
                "batch_call_vlm_interweaved", "setup_logging", "BACKEND", 
                "LLM_MODEL_NAME", "VLM_MODEL_NAME"):
        from mirage.core import llm
        return getattr(llm, name)
    
    # Config functions
    if name in ("load_config", "get_config_value"):
        from mirage.core import config
        return getattr(config, name)
    
    # Embeddings
    if name in ("get_best_embedding_model", "NomicVLEmbed"):
        from mirage.embeddings import models
        return getattr(models, name)
    
    # Pipeline functions
    if name == "generate_qa_for_chunk":
        from mirage.pipeline import qa_generator
        return qa_generator.generate_qa_for_chunk
    if name == "build_complete_context":
        from mirage.pipeline import context
        return context.build_complete_context
    if name == "fetch_domain_and_role":
        from mirage.pipeline import domain
        return domain.fetch_domain_and_role
    if name == "deduplicate_qa_pairs":
        from mirage.pipeline import deduplication
        return deduplication.deduplicate_qa_pairs
    
    # Utils
    if name == "run_preflight_checks":
        from mirage.utils import preflight
        return preflight.run_preflight_checks
    
    raise AttributeError(f"module 'mirage' has no attribute '{name}'")


__all__ = [
    # Version info
    "__version__",
    "__author__",
    # Core LLM functions (lazy loaded)
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
