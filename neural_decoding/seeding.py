"""Deterministic seeding helpers."""

from __future__ import annotations

import os
import random

import numpy as np

__all__ = ["new_rng", "set_global_seed"]


def set_global_seed(seed: int, *, torch_deterministic: bool = True) -> int:
    """Seed the Python, NumPy, and PyTorch global generators.

    Args:
        seed: Seed applied to every generator.
        torch_deterministic: Whether to ask cuDNN for deterministic kernels instead
            of autotuned ones.

    Returns:
        The seed that was applied.
    """
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ImportError:
        return seed

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return seed


def new_rng(seed: int) -> np.random.Generator:
    """Return an independent NumPy generator, preferred over the global one."""
    return np.random.default_rng(seed)
