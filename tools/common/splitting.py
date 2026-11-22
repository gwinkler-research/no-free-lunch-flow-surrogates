# Ratio-based train/val/test split of case indices from one seeded permutation
#   (the same n_runs, ratios and seed give the same split).
# Pipeline role: KarmanH5Dataset (tools/data/karman_h5_dataset.py) uses it for
#   its default split. Benchmark runs replace that split with the frozen IDs in
#   benchmark/splits/<dataset>_<kind>.json (benchmark/runner.py, _make_views()),
#   because a ratio-based split changes whenever the number of cases changes.
# The three parts are disjoint by construction (non-overlapping slices of one
#   shuffled index array). No data are touched here.
"""Random train/val/test splitting for case-indexed datasets."""

from __future__ import annotations

import math

import numpy as np


def split_runs(
    n_runs: int,
    train_split: float,
    val_split: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic random train/val/test split of run indices.

    Parameters
    ----------
    n_runs : int
        Total number of runs (cases).
    train_split, val_split : float
        Fractions in (0, 1) with ``train_split + val_split < 1``.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    train_idx, val_idx, test_idx : int64 arrays
        Disjoint index arrays whose union covers ``[0, n_runs)``.

    Raises
    ------
    ValueError
        If split fractions are outside valid range.
    """
    if not (0.0 < train_split < 1.0):
        raise ValueError("train_split must be in (0, 1)")
    if not (0.0 < val_split < 1.0):
        raise ValueError("val_split must be in (0, 1)")
    if train_split + val_split >= 1.0:
        raise ValueError("train_split + val_split must be < 1")

    # one seeded permutation; the three slices below are disjoint by construction
    rng = np.random.default_rng(seed)
    idx = np.arange(n_runs, dtype=np.int64)
    rng.shuffle(idx)
    n_train = int(math.floor(train_split * n_runs))
    n_train = max(1, min(n_runs - 2, n_train))
    n_val = int(math.floor(val_split * n_runs))
    n_val = max(1, min(n_runs - n_train - 1, n_val))
    train_idx = idx[:n_train]
    val_idx = idx[n_train : n_train + n_val]
    test_idx = idx[n_train + n_val :]
    return train_idx, val_idx, test_idx
