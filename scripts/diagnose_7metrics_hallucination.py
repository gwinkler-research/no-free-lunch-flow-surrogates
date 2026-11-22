# =============================================================================
# Purpose: hallucination-aware diagnostic metrics for the KVS pre-onset window
#   (does the model show wake motion before the true shedding onset?).
# Pipeline role: library. compute_metrics() is imported by
#   kvs_paper_diagnose.py (via importlib, by file path).
# Paper metric produced: halluc_pixel_frac_pre, the hallucination extent H of
#   the paper = fraction of pre-onset samples where |pred| exceeds
#   tau = 0.1 * max(|truth_steady|) while |truth| stays below tau, collected in
#   the FIXED early window t < FIXED_PRE_T = 0.5 s. One window, the same for
#   every case and split, and identical to the window alpha_rms uses in
#   diagnose_structural_hallucination.py (PRE_T = 0.5).
# Note: strip_k_rmse correlates at about 0.998-0.999 with the rollout RMSE and
#   is therefore not part of the reported metric set. It is computed to back
#   one sentence of the methods text.
# =============================================================================
"""Hallucination-aware diagnostic metrics for the KVS wake.

Four base metrics, 1-3 per phase (pre/ramp/onset/steady), 4 over the whole rollout:
  1. rmse_phase              - pointwise RMSE
  2. var_ratio_phase         - energy ratio (1 = ideal; averages out spatial structure)
  3. cos_sim_phase           - structural similarity (scale-invariant)
  4. strip_k_rmse            - spatial localisation (y bands, k = 0..3)

Three hallucination metrics in the pre-onset window. They are needed because var_ratio_pre
can be 0.99 while a two-digit percentage of the samples hallucinate: energy can compensate
spatially.
  5. shedding_power_pre      - FFT energy at f >= 2.5 Hz at the wake probes in the pre window.
                               The truth is ~0 here. ratio = pred/truth, ideal 1.0.
  6. max_diff_pre_rel        - max |pred - truth| in pre, relative to the steady truth range.
                               Local worst case, immune to averaging.
  7. halluc_pixel_frac_pre   - fraction of pre samples with |pred| > tau*max(|truth_steady|)
                               where the truth itself has |truth| < tau*max(steady).
                               tau = 0.1 (10 % of the steady amplitude as threshold).

compute_metrics() also returns an energy-threshold onset latency (onset_latency_*).

Pre-onset window: ONE fixed window `t < FIXED_PRE_T = 0.5 s`, identical to the window of
`alpha_rms` (PRE_T = 0.5).
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


PHASES = [
    ("pre",    0.0,  0.5),
    ("ramp",   0.5,  1.5),
    ("onset",  1.5,  2.5),
    ("steady", 2.5,  np.inf),
]
FIXED_PRE_T = 0.5  # the pre-onset window of H in s: fixed, the same for every case and
#                    split, and identical to the alpha_rms window
#                    (diagnose_structural_hallucination.PRE_T).
HALLUC_TAU = 0.1  # 10% of steady-amplitude as hallucination threshold


def fft_energy_above_freq(signal_per_probe, dt, f_min_hz=2.5):
    """Power-spectrum energy above f_min_hz, per probe. (T, n_probes) -> (n_probes,)"""
    n_t = signal_per_probe.shape[0]
    freqs = np.fft.rfftfreq(n_t, dt)
    mask = freqs >= f_min_hz
    spec = np.abs(np.fft.rfft(
        signal_per_probe - signal_per_probe.mean(axis=0, keepdims=True), axis=0))
    return (spec[mask] ** 2).sum(axis=0)


def _halluc_window(u_pred, u_true, pre2d, probe_local, dt, truth_range, tau, min_pre_frames=8):
    """3 hallucination metrics (spectral shed_pwr, absolute halluc_frac, local max_diff)
    evaluated in the pre-onset mask pre2d (N_case, T). Called once, with the fixed
    window t < FIXED_PRE_T broadcast over the cases (definition of H in the paper)."""
    n_case = u_pred.shape[0]
    # spectral: shed_pwr_pre_ratio (pred/true v-power above 2.5 Hz; truth ~0 in pre)
    e_true = np.full(n_case, np.nan); e_pred = np.full(n_case, np.nan)
    for c in range(n_case):
        fr = pre2d[c]
        if int(fr.sum()) < min_pre_frames:
            continue
        e_true[c] = fft_energy_above_freq(u_true[c, fr][:, probe_local, 1], dt, f_min_hz=2.5).mean()
        e_pred[c] = fft_energy_above_freq(u_pred[c, fr][:, probe_local, 1], dt, f_min_hz=2.5).mean()
    val = np.isfinite(e_true) & np.isfinite(e_pred)
    et, ep = e_true[val], e_pred[val]
    res = {
        "shedding_pwr_pre_ratio": float(ep.mean() / max(et.mean(), 1e-12)) if val.any() else float("nan"),
        "shedding_pwr_pre_excess_max": float(np.maximum(ep - et, 0).max()) if val.any() else float("nan"),
        "shedding_pwr_n_valid": int(val.sum()),
    }
    # absolute (halluc_frac) + local (max_diff), per-case
    maxpc = np.full(n_case, np.nan); fracs = []
    for c in range(n_case):
        fr = pre2d[c]
        if not fr.any():
            continue
        maxpc[c] = float(np.abs(u_pred[c, fr] - u_true[c, fr]).max())
        tq = np.abs(u_true[c, fr]) < tau
        pw = np.abs(u_pred[c, fr]) > tau
        fracs.append(float((tq & pw).mean()))
    fin = np.isfinite(maxpc)
    res["max_diff_pre_rel"] = float(maxpc[fin].max() / max(truth_range, 1e-12)) if fin.any() else float("nan")
    res["max_diff_pre_p95_rel"] = float(np.percentile(maxpc[fin], 95) / max(truth_range, 1e-12)) if fin.any() else float("nan")
    res["halluc_pixel_frac_pre"] = float(np.mean(fracs)) if fracs else float("nan")
    return res


def compute_metrics(u_pred, u_true, t_grid, coords_eval, probe_local, n_strips=4):
    """All 7 metrics. u_pred, u_true: (N_case, T, N_p2, 2).

    The pre-onset hallucination window is FIXED at t < FIXED_PRE_T = 0.5 s: one
    window, the same for every case and split (definition of H in the paper). It is
    the same window alpha_rms uses, so H and alpha_rms are read on identical samples.
    """
    n_case, T, N, _ = u_pred.shape
    dt = float(t_grid[1] - t_grid[0])
    phase_masks = {name: ((t_grid >= a) & (t_grid < b)) for name, a, b in PHASES}

    # Strip bins on y
    y = coords_eval[:, 1]
    ymin, ymax = y.min(), y.max()
    bins = np.linspace(ymin, ymax + 1e-9, n_strips + 1)
    strip_idx = np.digitize(y, bins) - 1

    out = {}

    # M1+M2: rmse + var_ratio per phase
    for name, mask in phase_masks.items():
        if not mask.any():
            continue
        diff = u_pred[:, mask] - u_true[:, mask]
        out[f"rmse_{name}"] = float(np.sqrt(np.mean(diff ** 2)))
        v_pred = float(np.var(u_pred[:, mask]))
        v_true = float(np.var(u_true[:, mask]))
        out[f"var_ratio_{name}"] = v_pred / max(v_true, 1e-12)

    # M3: cos_sim per phase (field-flatten)
    pred_flat = u_pred.reshape(n_case, T, -1)
    true_flat = u_true.reshape(n_case, T, -1)
    num = (pred_flat * true_flat).sum(axis=-1)
    denom = (np.linalg.norm(pred_flat, axis=-1) *
             np.linalg.norm(true_flat, axis=-1) + 1e-12)
    cos_per_frame = num / denom
    for name, mask in phase_masks.items():
        if not mask.any(): continue
        out[f"cos_sim_{name}"] = float(cos_per_frame[:, mask].mean())

    # M4: strip RMSE (y-bands)
    for k in range(n_strips):
        sel = strip_idx == k
        if not sel.any():
            out[f"strip{k}_rmse"] = float("nan"); continue
        diff = u_pred[:, :, sel] - u_true[:, :, sel]
        out[f"strip{k}_rmse"] = float(np.sqrt(np.mean(diff ** 2)))

    # ---- hallucination metrics: ONE pre-onset window ----
    # FIXED  t < FIXED_PRE_T = 0.5 s : the same window for every case and split
    # (definition of H in the paper; the same window alpha_rms uses).
    steady_mask = phase_masks["steady"]
    truth_range = float(u_true[:, steady_mask].max() - u_true[:, steady_mask].min()) if steady_mask.any() else 1.0
    steady_amp = float(np.abs(u_true[:, steady_mask]).max()) if steady_mask.any() else 0.0
    tau = HALLUC_TAU * steady_amp  # 10% of steady max-amplitude
    out["halluc_threshold"] = tau

    pre2d = np.broadcast_to(t_grid < FIXED_PRE_T, (n_case, T))
    out.update(_halluc_window(u_pred, u_true, pre2d, probe_local, dt, truth_range, tau))

    # M8: onset_latency, the time offset between pred and truth when they first reach
    # an energy threshold. Measures how late the model hits the onset.
    # Negative: pred too early (hallucination); positive: pred too late. Ideal: 0.
    e_true_t = np.sqrt((u_true ** 2).sum(axis=(2, 3)))   # (N_case, T), global L2 energy per frame
    e_pred_t = np.sqrt((u_pred ** 2).sum(axis=(2, 3)))
    # threshold: 50 % of the mean steady energy of the truth, per case
    e_true_steady = e_true_t[:, steady_mask].mean(axis=1) if steady_mask.any() else e_true_t[:, -10:].mean(axis=1)
    threshold = 0.5 * e_true_steady   # (N_case,)
    latencies = np.full(n_case, np.nan)
    for ci in range(n_case):
        thr = threshold[ci]
        if thr <= 0:
            continue
        idx_true = np.argmax(e_true_t[ci] > thr)
        idx_pred = np.argmax(e_pred_t[ci] > thr)
        # argmax returns 0 if all False; only valid if the value at the index exceeds the threshold
        if e_true_t[ci, idx_true] <= thr or e_pred_t[ci, idx_pred] <= thr:
            continue
        latencies[ci] = float(t_grid[idx_pred] - t_grid[idx_true])
    valid = ~np.isnan(latencies)
    out["onset_latency_mean"] = float(latencies[valid].mean()) if valid.any() else float("nan")
    out["onset_latency_median"] = float(np.median(latencies[valid])) if valid.any() else float("nan")
    out["onset_latency_abs_mean"] = float(np.abs(latencies[valid]).mean()) if valid.any() else float("nan")
    out["onset_latency_n_valid"] = int(valid.sum())

    return out
