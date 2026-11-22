# ============================================================================
# Purpose: per-case CMP structure and timing metrics on the two wall planes,
#   given as a 4-channel in-plane velocity [ux_top, uy_top, ux_bot, uy_bot].
# Pipeline role: case_metrics() is called by gate2_metrics() in
#   scripts/run_cmp3d_gate2.py. nrmse_3d, nrmse_inplane and the cum-WSS
#   relative L2 (cumwss_rel_l2_inner_mean) are computed there, not here.
# Metrics per case, all on the inner-90 node mask passed in:
#   - spatial_corr_inner90: Pearson correlation of predicted and true cum-WSS.
#   - steady_drift_excess: spurious unsteadiness, drift_p - drift_t, where
#     drift = (temporal std of the shear speed, averaged over nodes) /
#     (mean shear speed), steady window only.
#   - wiwnu_excess: within-wafer non-uniformity of cum-WSS,
#     (sigma/mu)_pred - (sigma/mu)_true.
#   - radial_profile_l2, onset_latency_excess, rmse_steady_inner90 and
#     var_ratio_steady are written to the JSON as well but are not columns of
#     the CMP table.
# ============================================================================
"""CMP structure metrics on a 4-channel wall-plane velocity.

All functions take their data as arguments (no module globals), so the same metric applies
to any prediction/truth pair in the layout [ux_top, uy_top, ux_bot, uy_bot]. cum-WSS comes
from benchmark.wss_postproc.velocity_to_cum_wss.
"""
from __future__ import annotations
import numpy as np
from benchmark.wss_postproc import velocity_to_cum_wss

N_RADIAL = 20


# Shear speed |u_top - u_bot| per node and time step.
def speed(vel):  # (T, n, 4) -> (T, n)
    sv = vel[..., 0:2] - vel[..., 2:4]
    return np.linalg.norm(sv, axis=-1)


# Feeds onset_latency_excess (not a column of the CMP table).
def onset_t(vel, mask, t_grid):
    """Build-up time proxy: first time at which the cumulative sum of the mask-averaged
    shear speed reaches 50 % of its total."""
    s = speed(vel)[:, mask].mean(axis=1)
    e = np.cumsum(s) / (s.sum() + 1e-12)
    idx = np.searchsorted(e, 0.5)
    return float(t_grid[min(idx, len(t_grid) - 1)])


# Mean cum-WSS in n_radial equal-width rings from the wafer centre out to the outermost
# masked node. Feeds radial_profile_l2 (not a column of the CMP table).
def radial_profile(cum, mask, center, fem_xy, n_radial=N_RADIAL):
    r = np.linalg.norm(fem_xy[mask] - center[None, :], axis=1)
    rmax = r.max() + 1e-12
    bins = np.linspace(0, rmax, n_radial + 1)
    vals = cum[mask]
    prof = np.zeros(n_radial)
    for b in range(n_radial):
        sel = (r >= bins[b]) & (r < bins[b + 1])
        prof[b] = vals[sel].mean() if sel.any() else 0.0
    return prof


# Per-case CMP structure and timing metrics, called by scripts/run_cmp3d_gate2.py.
def case_metrics(pred_vel, truth_vel_ci, *, steady, t_grid, fem_xy, inner_mask,
                 wafer_center, truth_cum_ci):
    """Seven metrics for one case. pred_vel, truth_vel_ci: (T, n, 4). steady: (T,) bool
    time mask. inner_mask: (n,) bool. truth_cum_ci: velocity_to_cum_wss(truth_vel_ci,
    t_grid), shape (n,), precomputed by the caller."""
    m = inner_mask
    pv, tv = pred_vel, truth_vel_ci
    sp_p = speed(pv)[steady][:, m]; sp_t = speed(tv)[steady][:, m]
    rmse_steady = float(np.sqrt(np.mean((sp_p - sp_t) ** 2)))
    std_p = sp_p.std(axis=0).mean(); std_t = sp_t.std(axis=0).mean()
    var_ratio = float(std_p / (std_t + 1e-9))
    cum_p = velocity_to_cum_wss(pv, t_grid); cum_t = truth_cum_ci
    a, b = cum_p[m], cum_t[m]
    spatial_corr = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else 0.0
    onset_excess = abs(onset_t(pv, m, t_grid) - onset_t(tv, m, t_grid))
    rp_p = radial_profile(cum_p, m, wafer_center, fem_xy)
    rp_t = radial_profile(cum_t, m, wafer_center, fem_xy)
    radial_l2 = float(np.sqrt(np.mean((rp_p - rp_t) ** 2)))
    # Spurious unsteadiness: predicted drift minus true drift.
    mean_p = sp_p.mean(axis=0).mean() + 1e-9; drift_p = std_p / mean_p
    mean_t = sp_t.mean(axis=0).mean() + 1e-9; drift_t = std_t / mean_t
    drift_excess = float(drift_p - drift_t)
    # Within-wafer non-uniformity (sigma/mu of cum-WSS on inner-90): prediction minus truth.
    wn_p = a.std() / (a.mean() + 1e-9); wn_t = b.std() / (b.mean() + 1e-9)
    wiwnu_excess = float(wn_p - wn_t)
    return dict(rmse_steady_inner90=rmse_steady, var_ratio_steady=var_ratio,
                spatial_corr_inner90=spatial_corr, onset_latency_excess=onset_excess,
                radial_profile_l2=radial_l2, steady_drift_excess=drift_excess,
                wiwnu_excess=wiwnu_excess)
