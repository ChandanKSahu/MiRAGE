"""
Evaluation module for MiRAGE - Metrics and dataset evaluation.

Imports are lazy to avoid loading optional dependencies at import time.
"""

_LAZY_IMPORTS = {
    # Basic metrics
    "evaluate_qa_dataset": ("metrics", "evaluate_qa_dataset"),
    "compute_faithfulness": ("metrics", "compute_faithfulness"),
    "compute_relevancy": ("metrics", "compute_relevancy"),
    # Optimized evaluation
    "OptimizedEvaluator": ("metrics_optimized", "OptimizedEvaluator"),
    "evaluate_subset": ("metrics_optimized", "evaluate_subset"),
    "generate_evaluation_report": ("metrics_optimized", "generate_evaluation_report"),
}


def __getattr__(name):
    """Lazy import to avoid loading optional dependencies at import time."""
    if name in _LAZY_IMPORTS:
        module_name, attr_name = _LAZY_IMPORTS[name]
        import importlib
        module = importlib.import_module(f"mirage.evaluation.{module_name}")
        return getattr(module, attr_name)
    raise AttributeError(f"module 'mirage.evaluation' has no attribute '{name}'")
