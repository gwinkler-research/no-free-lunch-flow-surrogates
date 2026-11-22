# ============================================================================
# Purpose: KVS metrics from predicted and true (u, v, p) fields on the eval
#   node set.
# Pipeline role: compute_kvs_metrics() is called by benchmark/eval_harness.py;
#   the result goes to results/benchmark/<dataset>_<split>_<model>_<seed>.json.
# Metrics (KVS table column in brackets):
#   - fft_freq_err_mean [Strouhal error] and fft_freq_err_median: relative
#     error of the dominant v frequency at the probe nodes, t >= 2.0 s.
#   - power_ratio [shedding power]: sum var(v_pred) / sum var(v_true) over
#     (case x probe), t >= 2.0 s.
#   - rmse_warped [phase-aligned RMSE]: DTW-aligned RMSE of v at the probe
#     nodes, full sequence.
#   - rollout_rmse [rollout RMSE]: RMSE of (u, v) over all 201 steps and all
#     P2 eval nodes. rollout_rmse_p: the same for p on the P1 eval nodes.
#   - var_ratio_unweighted: mean of the per-probe variance ratios. Not a
#     table column: near-zero true variance at laminar or boundary probes
#     inflates it by an order of magnitude or more.
# ============================================================================
"""KVS metrics on the eval node set.

Time axis: the refined_4x data end at T_end = 4.0 s and the inlet ramp ends at t = 1.0 s.
The spectral metrics (fft_freq_err, power_ratio, var_ratio_unweighted) use
t >= STEADY_T_MIN = 2.0 s, i.e. steps 100..200.
"""
from __future__ import annotations

import numpy as np

from tools.common.fft_analysis import fft_peak

STEADY_T_MIN = 2.0


# Strouhal error: |f_pred - f_true| / f_true of the dominant frequency per (case, probe),
# steady window only. Pairs with a non-finite peak or f_true <= 1e-9 are skipped.
# Returns (mean, median).
def _fft_freq_err(v_pred, v_true, t_grid, steady_t_min=STEADY_T_MIN):
    # v_*: (n_case, T, n_probe)
    mask = np.asarray(t_grid) >= steady_t_min
    dt = float(np.median(np.diff(np.asarray(t_grid))))
    errs = []
    for ci in range(v_pred.shape[0]):
        for pi in range(v_pred.shape[2]):
            fp = fft_peak(v_pred[ci, mask, pi], dt)
            ft = fft_peak(v_true[ci, mask, pi], dt)
            if np.isfinite(fp) and np.isfinite(ft) and ft > 1e-9:
                errs.append(abs(fp - ft) / ft)
    errs = np.asarray(errs, dtype=np.float64)
    if errs.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(errs)), float(np.median(errs))


# Shedding power: energy-weighted ratio sum var(pred) / sum var(true).
def _power_ratio(v_pred, v_true, t_grid, steady_t_min=STEADY_T_MIN):
    """Total v-power ratio sum var(vp) / sum var(vt) over (case x probe), steady window.

    Total predicted v-power divided by total true v-power. This equals a
    true-variance-weighted mean of the per-probe ratios: probes with vt ~ 0
    (boundary, laminar, no shedding) add almost nothing to numerator and
    denominator, instead of dominating an unweighted mean.
    """
    mask = np.asarray(t_grid) >= steady_t_min
    vp = np.var(np.asarray(v_pred, dtype=np.float64)[:, mask, :], axis=1)
    vt = np.var(np.asarray(v_true, dtype=np.float64)[:, mask, :], axis=1)
    return float(vp.sum() / max(vt.sum(), 1e-12))


# Not a column of the KVS table (see module header).
def _var_ratio_unweighted(v_pred, v_true, t_grid, steady_t_min=STEADY_T_MIN):
    """Unweighted mean(vp / vt) over (case x probe), steady window. Probes with
    vt ~ 1e-9 (laminar or boundary) inflate it by orders of magnitude."""
    mask = np.asarray(t_grid) >= steady_t_min
    vp = np.var(np.asarray(v_pred, dtype=np.float64)[:, mask, :], axis=1)
    vt = np.var(np.asarray(v_true, dtype=np.float64)[:, mask, :], axis=1)
    return float(np.mean(vp / np.maximum(vt, 1e-12)))


def _rmse(a, b):
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    return float(np.sqrt(np.mean((a - b) ** 2)))


# Phase-aligned RMSE: DTW alignment of each v probe series (full sequence), then the RMSE
# along the warping path; mean over (case, probe).
def _rmse_warped(v_pred, v_true):
    from dtw import dtw
    vals = []
    for ci in range(v_pred.shape[0]):
        for pi in range(v_pred.shape[2]):
            sp = np.asarray(v_pred[ci, :, pi], dtype=np.float64)
            st = np.asarray(v_true[ci, :, pi], dtype=np.float64)
            al = dtw(sp, st, keep_internals=True)
            vals.append(np.sqrt(np.mean((sp[al.index1] - st[al.index2]) ** 2)))
    return float(np.mean(vals)) if vals else float("nan")


# Entry point, called by benchmark/eval_harness.py.
def compute_kvs_metrics(u_pred, p_pred, u_true, p_true, t_grid, probe_local_idx):
    """u_*: (n_case, T, n_eval_p2, 2); p_*: (n_case, T, n_eval_p1); probe_local_idx: positions
    of the probe nodes in the n_eval_p2 array. Returns a dict with the metrics."""
    pli = np.asarray(probe_local_idx, dtype=np.int64)
    v_pred = np.asarray(u_pred)[:, :, pli, 1]
    v_true = np.asarray(u_true)[:, :, pli, 1]
    fft_mean, fft_median = _fft_freq_err(v_pred, v_true, t_grid)
    return {
        "fft_freq_err_mean": fft_mean,
        "fft_freq_err_median": fft_median,
        "power_ratio": _power_ratio(v_pred, v_true, t_grid),
        "var_ratio_unweighted": _var_ratio_unweighted(v_pred, v_true, t_grid),
        "rollout_rmse": _rmse(u_pred, u_true),
        "rollout_rmse_p": _rmse(p_pred, p_true),
        "rmse_warped": _rmse_warped(v_pred, v_true),
    }
