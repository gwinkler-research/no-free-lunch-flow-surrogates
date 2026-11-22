# =============================================================================
# Purpose: structural-hallucination metric for the KVS pre-onset window
#   (Karman-like spatial structure appearing before the true shedding
#   onset).
# Pipeline role: library. pattern_match_pre() is imported by
#   kvs_paper_diagnose.py (via importlib, by file path) to compute alpha_rms
#   per model and run.
# Paper metric produced: alpha_rms, the structural hallucination measure of
#   the paper = RMS over t of the signed projection of the pre-onset residual
#   onto the steady Karman template (read together with H; not redundant,
#   r = 0.34).
# Note: pattern_match_pre() also returns cos_max and alpha_max. Only alpha_rms
#   is reported; alpha_max, a maximum over frames, is prone to outliers.
# =============================================================================
"""Structural hallucination metric for the KVS pre-onset window.

Failure mode: the model shows Karman-like spatial structure during the pre-onset phase,
where the truth has only a uniform laminar background. pattern_match_pre() compares the
pre-onset residual of the prediction with the steady Karman pattern of the truth, by
cosine similarity (amplitude-invariant) and by the projection coefficient alpha
(amplitude-relative).
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


PRE_T = 0.5   # pre-onset window t < PRE_T in s (the same window as H)


def pattern_match_pre(u_pred_grid, u_true_grid, pre_mask, steady_mask):
    """Spatial cosine-similarity (amplitude-invariant) AND projection coefficient
    (amplitude-aware) between pre-pred-residual and steady-Karman-template.

    Template = u_true[last_steady_frame] − mean(u_true[steady])  (per case).
    Pre-residual = u_pred[pre_t] − mean(u_true[pre])             (per case, per frame).

    Returns dict with arrays of shape (N,):
        cos_max:   max_t  cosine(pre_res(t), template)               [structure only]
        alpha_max: max_t  |<pre_res(t), template>| / ||template||²   [strength-aware]
        alpha_rms: rms over t of alpha(t)                            [strength-aware]

    alpha=k means: pre_res(t) ≈ k · template, i.e. k×Karman-strength already present in pre.
    """
    N, T, H, W, _ = u_pred_grid.shape
    u_lam_pre = u_true_grid[:, pre_mask].mean(axis=1, keepdims=True)
    u_lam_st  = u_true_grid[:, steady_mask].mean(axis=1, keepdims=True)
    template = u_true_grid[:, -1:] - u_lam_st                                # (N, 1, H, W, 2)
    template_flat = template.reshape(N, -1)
    template_norm = np.linalg.norm(template_flat, axis=1) + 1e-12
    template_norm_sq = (template_flat ** 2).sum(axis=1) + 1e-12

    pred_res = u_pred_grid[:, pre_mask] - u_lam_pre
    T_pre = pred_res.shape[1]
    pred_res_flat = pred_res.reshape(N, T_pre, -1)
    pred_norm = np.linalg.norm(pred_res_flat, axis=2) + 1e-12

    # Dot products per (case, t)
    dot = (pred_res_flat * template_flat[:, None, :]).sum(axis=2)            # (N, T_pre)
    cos = dot / (pred_norm * template_norm[:, None])
    # alpha_rms below (RMS_t of alpha) is the reported paper metric
    alpha = dot / template_norm_sq[:, None]                                  # signed projection coefficient

    return {
        "cos_max":    cos.max(axis=1),
        "alpha_max":  np.abs(alpha).max(axis=1),
        "alpha_rms":  np.sqrt((alpha ** 2).mean(axis=1)),
    }
