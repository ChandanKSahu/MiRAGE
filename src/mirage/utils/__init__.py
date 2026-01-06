"""
Utilities module for MiRAGE - Preflight checks, statistics, and ablation studies.

Imports are lazy to avoid loading optional dependencies at import time.
"""

_LAZY_IMPORTS = {
    # Preflight
    "run_preflight_checks": ("preflight", "run_preflight_checks"),
    "check_gpu_availability": ("preflight", "check_gpu_availability"),
    "check_api_connectivity": ("preflight", "check_api_connectivity"),
    "PreflightChecker": ("preflight", "PreflightChecker"),
    # Statistics
    "compute_dataset_stats": ("stats", "compute_dataset_stats"),
    "print_dataset_stats": ("stats", "print_dataset_stats"),
    "compute_qa_category_stats": ("stats", "compute_qa_category_stats"),
    "print_qa_category_stats": ("stats", "print_qa_category_stats"),
    # Ablation
    "run_ablation_study": ("ablation", "run_ablation_study"),
    "AblationConfig": ("ablation", "AblationConfig"),
}


def __getattr__(name):
    """Lazy import to avoid loading optional dependencies at import time."""
    if name in _LAZY_IMPORTS:
        module_name, attr_name = _LAZY_IMPORTS[name]
        import importlib
        module = importlib.import_module(f"mirage.utils.{module_name}")
        return getattr(module, attr_name)
    raise AttributeError(f"module 'mirage.utils' has no attribute '{name}'")
