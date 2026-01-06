"""
Embeddings module for MiRAGE - Embedding models and rerankers.

Imports are lazy to avoid loading heavy dependencies at import time.
"""

_LAZY_IMPORTS = {
    # Embedding models
    "BaseEmbeddingModel": ("models", "BaseEmbeddingModel"),
    "NomicVLEmbed": ("models", "NomicVLEmbed"),
    "SentenceTransformerEmbedder": ("models", "SentenceTransformerEmbedder"),
    "HuggingFaceAPIEmbedder": ("models", "HuggingFaceAPIEmbedder"),
    "get_best_embedding_model": ("models", "get_best_embedding_model"),
    # Multimodal rerankers
    "BaseReranker": ("rerankers_multimodal", "BaseReranker"),
    "MonoVLMReranker": ("rerankers_multimodal", "MonoVLMReranker"),
    "VLMReranker": ("rerankers_multimodal", "VLMReranker"),
    "TextEmbeddingReranker": ("rerankers_multimodal", "TextEmbeddingReranker"),
    # Text rerankers
    "LLMReranker": ("rerankers_text", "LLMReranker"),
}


def __getattr__(name):
    """Lazy import to avoid loading heavy dependencies at import time."""
    if name in _LAZY_IMPORTS:
        module_name, attr_name = _LAZY_IMPORTS[name]
        import importlib
        module = importlib.import_module(f"mirage.embeddings.{module_name}")
        return getattr(module, attr_name)
    raise AttributeError(f"module 'mirage.embeddings' has no attribute '{name}'")
