#!/usr/bin/env python3
# ============================================================================
# Purpose: the mirror-branch diagnostic of the Discussion (Sec. 7, wake mirror
#   symmetry). For each S4D-branch DON seed it compares the rollout RMSE of the
#   prediction with the RMSE of its mirror image across the cylinder axis
#   (y_c = 0.2; u even, v odd, nodes matched to the nearest evaluation node),
#   and reports per run how many cases the mirrored prediction fits better and
#   the RMSE of the case-wise better of the two ("best2").
# Pipeline role: reads the stored predictions of the S4D-branch DON runs and the
#   truth NPZ (both outside this repository; pass --runs-root / --truth-root) and
#   writes reports/kvs_mirror_aligned_rmse.json, which values/emit_values.py turns
#   into \vKvsMirrorCloserCases, \vKvsMirrorNCases and \vKvsMirrorGainPct.
# Eval-consistency: per-case MSE over time, the valid evaluation nodes (wall
#   distance > 0.006) and both velocity components; aggregate sqrt(mean case MSE).
#   The oscillating-case count uses the late window (snapshots 138..200), a
#   shedding band of 0.3..5 Hz, a peak-to-median power ratio of at least 5 and
#   U_max > 1.5.
# ============================================================================
"""Mirror-aligned rollout RMSE of the S4D-branch DON rows (Discussion, mirror branch)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
import numpy as np  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402
from benchmark.splits import load_split  # noqa: E402

YC = 0.2
WALL_MARGIN = 0.006
Y_TOP = 0.41
MAX_EDGE = 0.06

# Oscillating-case mask: late window, shedding band and peak-to-median power ratio.
LATE_WINDOW = slice(138, 201)
DT = 0.02
F_BAND_LO, F_BAND_HI = 0.3, 5.0
SNR_THRESHOLD = 5.0

RUNS = {"interp": "s4d_1phase4k_clean_interp_seed{}",
        "extrap": "s4d_1phase4k_dsspan_x_extrap_seed{}"}
SEEDS = (1, 2, 3)
OUT = _REPO / "reports" / "kvs_mirror_aligned_rmse.json"


def _geometry():
    spec = load_split("kvs", "interp")
    ei = np.asarray(spec.eval_idx_p2, np.int64)
    mesh = np.load(_REPO / "meshes" / "refined_4x" / "mesh.npz", allow_pickle=False)
    xy = mesh["coords_p2"][ei]
    tree = cKDTree(xy)
    mpos = xy.copy()
    mpos[:, 1] = 2.0 * YC - mpos[:, 1]
    dist, midx = tree.query(mpos)
    assert dist.max() <= MAX_EDGE, f"mirror match {dist.max():.3f} > max_edge"
    d_wall = np.minimum(xy[:, 1], Y_TOP - xy[:, 1])
    valid = d_wall > WALL_MARGIN
    return midx, valid


def _mirror(u, midx):
    """u: (C, T, N, 2) -> mirrored prediction (u even, v odd, node substitution)."""
    m = u[:, :, midx, :].copy()
    m[..., 1] *= -1.0
    return m


def _late_snr(u, pli):
    """Per case: peak power in the shedding band over the median power outside it,
    from the probe-averaged spectrum of v in the late window."""
    v = u[:, LATE_WINDOW, :, 1][:, :, pli]
    spec = np.fft.rfft(v - v.mean(axis=1, keepdims=True), axis=1)
    freqs = np.fft.rfftfreq(v.shape[1], d=DT)
    band = (freqs >= F_BAND_LO) & (freqs <= F_BAND_HI)
    power = (np.abs(spec) ** 2).mean(axis=2)
    peak_idx = np.argmax(np.where(band[None, :], power, -np.inf), axis=1)
    return np.array([power[c, peak_idx[c]] / max(np.median(power[c, ~band]), 1e-30)
                     for c in range(u.shape[0])])


def _osc_mask(truth_root, split):
    z = np.load(Path(truth_root) / f"kvs_{split}.npz", allow_pickle=False)
    u = z["u_eval"]
    cids = list(map(str, z["case_ids"]))
    pli = np.asarray(load_split("kvs", split).probe_local_idx, np.int64)
    umax = np.array([float(c.split("_")[1]) / 10.0 for c in cids])
    return (_late_snr(u, pli) >= SNR_THRESHOLD) & (umax > 1.5), cids


def _case_mse(pred, truth, node_sel):
    """Per-case MSE over (T, node_sel, u+v). pred/truth: (C, T, N, 2)."""
    d = (pred[:, :, node_sel, :].astype(np.float64) - truth[:, :, node_sel, :]) ** 2
    return d.mean(axis=(1, 2, 3))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-root", default=str(_REPO / "results" / "benchmark" / "runs"),
                    help="root of the run directories (predictions.npz)")
    ap.add_argument("--truth-root", default=str(_REPO / "results" / "benchmark" / "truth"),
                    help="root of the truth NPZ files")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    midx, valid = _geometry()
    print(f"[geo] n_valid={int(valid.sum())}/{valid.size}")

    runs = Path(a.runs_root)
    out = {"meta": {"stem": "s4d_1phase4k_clean|s4d_1phase4k_dsspan_x", "seeds": list(SEEDS),
                    "yc": YC, "wall_margin": WALL_MARGIN,
                    "max_edge": MAX_EDGE, "n_valid_nodes": int(valid.sum()),
                    "n_nodes": int(valid.size),
                    "definition": "per-case MSE over (T, valid eval nodes, u+v); aggregate "
                    "sqrt(mean case MSE); headline best2 = min(id, mirror); best_all omitted "
                    "(paper claim uses best2)"},
           "per_run": {}}

    for split in ("interp", "extrap"):
        tz = np.load(Path(a.truth_root) / f"kvs_{split}.npz", allow_pickle=False)
        truth = tz["u_eval"].astype(np.float32)
        osc, cids = _osc_mask(a.truth_root, split)
        # Double-mirror floor (truth case 0): error of the node substitution alone.
        t0 = truth[0:1]
        dm = _mirror(_mirror(t0, midx), midx)
        floor = float(np.sqrt(_case_mse(dm, t0, valid)[0]))
        out["meta"][f"double_mirror_floor_{split}_case0"] = floor
        print(f"[{split}] double-mirror floor case0 = {floor:.4f}")
        for seed in SEEDS:
            d = runs / RUNS[split].format(seed)
            with np.load(d / "predictions.npz", allow_pickle=False) as z:
                pred = z["u_eval"].astype(np.float32)
            mse_id_full = _case_mse(pred, truth, slice(None))
            mse_id = _case_mse(pred, truth, valid)
            mse_mir = _case_mse(_mirror(pred, midx), truth, valid)
            best2 = np.minimum(mse_id, mse_mir)
            mb = mse_mir < mse_id
            rec = {"rmse_raw_full": float(np.sqrt(mse_id_full.mean())),
                   "rmse_raw_valid": float(np.sqrt(mse_id.mean())),
                   "rmse_mirror_valid": float(np.sqrt(mse_mir.mean())),
                   "rmse_best2": float(np.sqrt(best2.mean())),
                   "n_mirror_better": int(mb.sum()),
                   "n_mirror_better_osc": int((mb & osc).sum()),
                   "n_osc": int(osc.sum()), "n_cases": int(mb.size),
                   "mirror_better_case_ids": [c for c, b in zip(cids, mb) if b]}
            out["per_run"][d.name] = rec
            print(f"[ok] {d.name}: raw={rec['rmse_raw_valid']:.4f} "
                  f"mirror={rec['rmse_mirror_valid']:.4f} best2={rec['rmse_best2']:.4f} "
                  f"mirror-better {rec['n_mirror_better']}/{rec['n_cases']} "
                  f"(osc {rec['n_mirror_better_osc']}/{rec['n_osc']})")
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"[done] -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
