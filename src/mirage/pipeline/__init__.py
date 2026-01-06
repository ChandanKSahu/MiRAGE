"""
Pipeline module for MiRAGE - Document processing, QA generation, and deduplication.

Imports are lazy to avoid loading optional dependencies at import time.
"""

# Mapping of attribute names to (module_name, attr_name)
_LAZY_IMPORTS = {
    # PDF Processing (requires docling, matplotlib - optional)
    "process_pdf_to_markdown": ("pdf_processor", "process_pdf_to_markdown"),
    "process_directory": ("pdf_processor", "process_directory"),
    # Chunking
    "chunk_markdown_to_semantic": ("chunker", "chunk_markdown_to_semantic"),
    "process_markdown_file": ("chunker", "process_markdown_file"),
    # Context
    "build_complete_context": ("context", "build_complete_context"),
    "retrieve_similar_chunks": ("context", "retrieve_similar_chunks"),
    "ContextBuilder": ("context", "ContextBuilder"),
    # QA Generation
    "generate_qa_for_chunk": ("qa_generator", "generate_qa_for_chunk"),
    "verify_qa_pair": ("qa_generator", "verify_qa_pair"),
    "select_best_qa_pairs": ("qa_generator", "select_best_qa_pairs"),
    # Domain
    "fetch_domain_and_role": ("domain", "fetch_domain_and_role"),
    "load_domain_expert_from_env": ("domain", "load_domain_expert_from_env"),
    "save_domain_expert_to_env": ("domain", "save_domain_expert_to_env"),
    "DomainExtractor": ("domain", "DomainExtractor"),
    # Deduplication
    "deduplicate_qa_pairs": ("deduplication", "deduplicate_qa_pairs"),
    "cluster_questions": ("deduplication", "cluster_questions"),
    "merge_similar_qa": ("deduplication", "merge_similar_qa"),
}


def __getattr__(name):
    """Lazy import to avoid loading optional dependencies at import time."""
    if name in _LAZY_IMPORTS:
        module_name, attr_name = _LAZY_IMPORTS[name]
        import importlib
        module = importlib.import_module(f"mirage.pipeline.{module_name}")
        return getattr(module, attr_name)
    raise AttributeError(f"module 'mirage.pipeline' has no attribute '{name}'")


__all__ = list(_LAZY_IMPORTS.keys())
