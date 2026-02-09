"""
Centralized device management for MiRAGE.

Provides automatic GPU/CPU detection and switching.
GPU is always preferred when available; falls back to CPU gracefully.
"""

import os
import warnings
import torch
from typing import Optional, List


def get_device(gpus: Optional[List[int]] = None) -> str:
    """
    Determine the best available device.
    
    Priority:
        1. Specific GPU from `gpus` list (e.g., cuda:1)
        2. Default CUDA device (cuda:0) if GPU is available
        3. CPU as fallback
    
    Args:
        gpus: Optional list of GPU IDs to use. First GPU in list is selected.
    
    Returns:
        Device string: "cuda:N" or "cpu"
    """
    if gpus and len(gpus) > 0:
        gpu_id = gpus[0]
        if torch.cuda.is_available() and gpu_id < torch.cuda.device_count():
            return f"cuda:{gpu_id}"
        # Requested GPU not available, fall back
        if torch.cuda.is_available():
            return "cuda:0"
        return "cpu"
    
    if torch.cuda.is_available():
        return "cuda:0"
    
    return "cpu"


def is_gpu_available() -> bool:
    """Check if any CUDA GPU is available."""
    return torch.cuda.is_available() and torch.cuda.device_count() > 0


def get_device_map(gpus: Optional[List[int]] = None) -> str:
    """
    Get device_map string for model loading (HuggingFace-compatible).
    
    Args:
        gpus: Optional list of GPU IDs.
    
    Returns:
        Device map string for from_pretrained() calls.
    """
    device = get_device(gpus)
    if device.startswith("cuda"):
        return device
    return "cpu"


def should_pin_memory() -> bool:
    """
    Determine whether to use pinned memory for DataLoaders.
    
    pin_memory=True speeds up CPU-to-GPU transfers but causes warnings
    (and is wasteful) when no GPU is available.
    
    Returns:
        True only if CUDA is available.
    """
    return torch.cuda.is_available()


def clear_gpu_cache():
    """Clear GPU memory cache if CUDA is available."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def gpu_sync():
    """Synchronize GPU operations if CUDA is available."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def suppress_pin_memory_warnings():
    """
    Suppress the 'pin_memory' UserWarning from PyTorch DataLoader.
    
    This is useful when third-party libraries (easyocr, docling) use
    pin_memory=True without checking for GPU availability.
    Call this early in the pipeline before importing those libraries.
    """
    warnings.filterwarnings(
        "ignore",
        message=".*pin_memory.*argument is set as true but no accelerator.*",
        category=UserWarning,
    )


def setup_device_environment():
    """
    Configure the runtime environment for optimal device usage.
    
    - Detects GPU availability
    - Suppresses spurious pin_memory warnings when on CPU
    - Logs the device configuration
    
    Call this once at pipeline startup.
    """
    device = get_device()
    
    # Suppress pin_memory warnings when no GPU is present
    if not is_gpu_available():
        suppress_pin_memory_warnings()
    
    return device
