# Seeds the Python, NumPy and PyTorch (CPU and CUDA) random generators.
# Call set_seed(seed) once at the start of a run.
"""Reproducibility seeding utilities."""

from __future__ import annotations

import random

import numpy as np


def set_seed(seed: int) -> None:
    """Set random seeds for Python, NumPy and PyTorch (if available)."""
    random.seed(seed)
    np.random.seed(seed)
    # torch is an optional dependency here; seed it only if importable.
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
