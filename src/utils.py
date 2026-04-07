"""Utility functions for reproducibility and common operations."""

import os
import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility.

    Sets seeds for Python random, NumPy, and PyTorch (CPU and CUDA).

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set environment variable for Python hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device(prefer_cuda: bool = True) -> str:
    """Get the best available device.

    Args:
        prefer_cuda: Whether to prefer CUDA if available

    Returns:
        Device string ('cuda' or 'cpu')
    """
    if prefer_cuda and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def ensure_dir(path: str | Path) -> Path:
    """Ensure directory exists, create if necessary.

    Args:
        path: Directory path

    Returns:
        Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters in a model.

    Args:
        model: PyTorch model

    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
