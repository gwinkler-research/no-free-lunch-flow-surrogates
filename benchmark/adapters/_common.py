# benchmark/adapters/_common.py
# ============================================================================
# Purpose: shared numerics of the KVS adapters. The piecewise-linear inlet ramp
#   s(t) and its derivative ds/dt reconstructed from theta = [U_max, tau1..tau5],
#   its per-case float32 wrapper control_features(), the
#   (T,H,W,C) <-> (T,C,H,W) reshapes for the raster cache, the latent mean/std
#   estimator, the epoch logger and save_ckpt(). Not an adapter itself.
# Pipeline role: imported by grid_convae, latent_dynamics, latent_oneshot,
#   kvs_field_deeponet and dmdc. assemble_kvs_field_arrays() turns a predicted
#   (n_case,201,H,W,3) grid into the eval-node `arrays` dict stored in
#   predictions.npz, so every grid-based KVS row passes through this file.
#   Paper: Sec. 3 (control features s(t), ds/dt).
# Eval-consistency: two fixed constants define the excitation every KVS row is
#   trained and scored on: LEVELS = [0.2,0.4,0.6,0.8,1.0] and T_RAMP = 1.0.
#   save_ckpt() only writes the payload it is given; it takes no part in
#   checkpoint selection.
# ============================================================================
"""Shared helpers for the benchmark adapters.

Includes: the KVS piecewise ramp s(t)/ds/dt feature builder (theta = [U_max, tau1..tau5],
fixed levels [0.2,0.4,0.6,0.8,1.0], T_ramp=1.0), grid<->channel reshaping for the
fp16 raster cache, latent-norm stats, the epoch logger, and a grid-to-eval-nodes
assembler that turns a (n_test,201,H,W,3) prediction grid into the kvs/field
`arrays` dict via benchmark.rasterizer.grid_to_nodes.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from benchmark.rasterizer import grid_to_nodes

LEVELS = [0.2, 0.4, 0.6, 0.8, 1.0]
T_RAMP = 1.0


def ramp_s_dsdt(tau, t_grid):
    """Piecewise-linear ramp value s(t) and its time-derivative ds/dt.

    Knots: (0, 0) then (tau_i * T_RAMP, LEVELS[i]) for i in 0..4, i.e. at time
    tau_i (a fractional time in [0,1] scaled by T_RAMP) the inlet has ramped to
    LEVELS[i] of its target. After the last knot s stays at its final level (ds = 0);
    before t=0 it is 0. tau need not be pre-sorted; knots are sorted by time.
    """
    tau = np.asarray(tau, dtype=np.float64).ravel()
    t_grid = np.asarray(t_grid, dtype=np.float64)
    knot_t = np.concatenate([[0.0], tau * T_RAMP])
    knot_s = np.concatenate([[0.0], np.asarray(LEVELS, dtype=np.float64)])
    order = np.argsort(knot_t, kind="stable")
    knot_t, knot_s = knot_t[order], knot_s[order]
    # ensure strictly increasing time for interp (collapse duplicates, keep last s)
    keep = np.concatenate([[True], np.diff(knot_t) > 1e-12])
    knot_t, knot_s = knot_t[keep], knot_s[keep]
    s = np.interp(t_grid, knot_t, knot_s, left=0.0, right=knot_s[-1])
    # derivative: slope of the segment t falls into; 0 outside [knot_t[0], knot_t[-1]]
    seg_slope = np.diff(knot_s) / np.maximum(np.diff(knot_t), 1e-12)   # (K-1,)
    ds = np.zeros_like(s)
    inside = (t_grid > knot_t[0]) & (t_grid < knot_t[-1])
    seg_idx = np.clip(np.searchsorted(knot_t, t_grid[inside], side="right") - 1, 0, len(seg_slope) - 1)
    ds[inside] = seg_slope[seg_idx]
    return s.astype(np.float64), ds.astype(np.float64)


# Per-case wrapper of ramp_s_dsdt (used by latent_dynamics); the other KVS adapters
# call ramp_s_dsdt directly.
def control_features(theta_row, t_grid):
    """(s(t), ds/dt) of one case as float32, reconstructed from theta = [U_max, tau_1..tau_5]."""
    s, ds = ramp_s_dsdt(np.asarray(theta_row)[1:], np.asarray(t_grid, dtype=np.float64))
    return s.astype(np.float32), ds.astype(np.float32)


def grid_to_channels_first(grid_thwc: np.ndarray) -> np.ndarray:
    """(..., H, W, 3) -> (..., 3, H, W)."""
    g = np.asarray(grid_thwc)
    return np.moveaxis(g, -1, -3)


def channels_first_to_grid(g_cfhw: np.ndarray) -> np.ndarray:
    """(..., 3, H, W) -> (..., H, W, 3)."""
    g = np.asarray(g_cfhw)
    return np.moveaxis(g, -3, -1)


def latent_norm_stats(z: np.ndarray):
    """z: (..., d) -> (mean (d,), std (d,)) over all but the last axis."""
    z = np.asarray(z, dtype=np.float64).reshape(-1, np.asarray(z).shape[-1])
    mean = z.mean(axis=0).astype(np.float32)
    std = np.maximum(z.std(axis=0), 1e-8).astype(np.float32)
    return mean, std


class EpochLogger:
    """Print every `every` epochs (default 10). After epoch 1, print a total-time
    estimate from the epoch-1 wallclock. Also logs checkpoint saves."""

    def __init__(self, tag: str, n_epochs: int, every: int = 10):
        self.tag, self.n_epochs, self.every = tag, int(n_epochs), int(every)
        self._t_start = None
        self._t_ep1 = None

    def epoch_start(self):
        if self._t_start is None:
            self._t_start = time.time()
        self._t0 = time.time()

    def epoch_end(self, epoch: int, train_loss: float, val_loss: float, best_val: float, best_epoch: int):
        dt = time.time() - self._t0
        if epoch == 1 and self._t_ep1 is None:
            self._t_ep1 = dt
            print(f"[{self.tag}] ep1: {dt:.1f}s -> est. total: {dt*self.n_epochs/60:.0f} min", flush=True)
        if epoch == 1 or epoch % self.every == 0 or epoch == self.n_epochs:
            print(f"[{self.tag}] ep {epoch:04d}/{self.n_epochs} train={train_loss:.4e} "
                  f"val={val_loss:.4e} best={best_val:.4e}(ep{best_epoch})", flush=True)

    def saved(self, name: str, val: float, ep: int):
        print(f"[{self.tag}] Saved {name} (val={val:.6f}, ep={ep})", flush=True)


def assemble_kvs_field_arrays(pred_grid_thwc: np.ndarray, case_ids, t_grid, *,
                              coords_p2, coords_p1, eval_idx_p2, eval_idx_p1, bbox) -> dict:
    """pred_grid_thwc: (n_test, 201, H, W, 3) channels [u, v, p] -> kvs/field arrays.

    Velocity goes to coords_p2[eval_idx_p2] (n_eval_p2, 2); pressure to
    coords_p1[eval_idx_p1] (n_eval_p1,). Uses benchmark.rasterizer.grid_to_nodes.
    """
    pred = np.asarray(pred_grid_thwc, dtype=np.float64)
    n_test, T = pred.shape[0], pred.shape[1]
    p2_xy = np.asarray(coords_p2, dtype=np.float64)[np.asarray(eval_idx_p2)]
    p1_xy = np.asarray(coords_p1, dtype=np.float64)[np.asarray(eval_idx_p1)]
    u_eval = np.empty((n_test, T, p2_xy.shape[0], 2), dtype=np.float32)
    p_eval = np.empty((n_test, T, p1_xy.shape[0]), dtype=np.float32)
    for ci in range(n_test):
        for ti in range(T):
            uv = grid_to_nodes(pred[ci, ti, :, :, 0:2], p2_xy, bbox=bbox)   # (n_eval_p2, 2)
            pp = grid_to_nodes(pred[ci, ti, :, :, 2], p1_xy, bbox=bbox)      # (n_eval_p1,)
            u_eval[ci, ti] = uv.astype(np.float32)
            p_eval[ci, ti] = pp.astype(np.float32)
    return {
        "u_eval": u_eval, "p_eval": p_eval,
        "case_ids": np.array([str(c) for c in case_ids]),
        "t_grid": np.asarray(t_grid, dtype=np.float32),
    }


def save_ckpt(out_dir: Path, name: str, payload: dict) -> Path:
    """torch.save(payload) to out_dir/name (directory created if needed); returns the path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    p = Path(out_dir) / name
    torch.save(payload, p)
    return p
