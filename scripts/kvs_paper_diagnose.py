#!/usr/bin/env python3
# =============================================================================
# Purpose: run the KVS diagnostic metrics over every run directory of the
#   paper registry and both splits. It does not implement those metrics. It
#   loads compute_metrics() from scripts/diagnose_7metrics_hallucination.py and
#   pattern_match_pre() from scripts/diagnose_structural_hallucination.py by
#   file path (importlib.spec_from_file_location), so the reported numbers are
#   the ones those two functions define. Writes
#   reports/kvs_diag_paper<sfx>_<split>.json as
#   {run_dir: {strip_k_rmse, halluc_pixel_frac_pre, alpha_rms}}.
# Pipeline role: one of three KVS scoring paths. The core four columns come
#   from benchmark/eval_harness.py, the onset column from
#   scripts/onset_oscmed_apply.py; the columns produced here are read by
#   scripts/build_paper_matrix.py and become the H column
#   (halluc_pixel_frac_pre) and the alpha_rms column of the KVS table
#   (Table 5). strip_k_rmse is computed and stored for diagnosis only; it is
#   not part of the reported metric set.
# Eval-consistency: one fixed pre-onset window, shared by both metrics.
#   PRE_T = 0.5 s applies to H and to alpha_rms alike and equals FIXED_PRE_T in
#   diagnose_7metrics_hallucination.py; this is the window of the paper's
#   definition of H. This file reads no onset NPZ, so H does not depend on a
#   detector output. Truth comes from results/benchmark/truth/kvs_<split>.npz,
#   the raster is GRID_HW = (83, 442) over the fixed KVS_BBOX with weights from
#   benchmark.rasterizer, and STEADY_T_START = 2.5 s bounds the steady segment.
#   Prediction and truth pass through the identical raster path. The model
#   list is benchmark.paper_models.active_kvs_2a(), the same registry the
#   matrix is built from.
# =============================================================================
"""KVS diagnostic metrics (strip_k_rmse, H, alpha_rms) for the models of the paper registry.

Reuses the metric functions instead of reimplementing them:
  * compute_metrics    (scripts/diagnose_7metrics_hallucination.py)  -> strip{0..3}_rmse, halluc_pixel_frac_pre
  * pattern_match_pre  (scripts/diagnose_structural_hallucination.py) -> alpha_rms
alpha_rms needs no POD (only the rasterised velocity and the truth template), so no
extrap POD basis is needed.
Model list = benchmark.paper_models.active_kvs_2a() (KVS_2A_CLEAN: the models of the paper,
both splits).
Output: reports/kvs_diag_paper<sfx>_<split>.json = {run_dir: {strip_k_rmse, halluc_pixel_frac_pre, alpha_rms}}.

Pre-onset window: ONE fixed window `t < PRE_T = 0.5 s`, used for H and alpha_rms alike.

  python3 scripts/kvs_paper_diagnose.py --split interp
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
import numpy as np
from benchmark.splits import load_split
from benchmark.rasterizer import precompute_raster_weights, apply_raster_weights, KVS_BBOX
from benchmark.paper_models import active_kvs_2a, kvs_rundirs

GRID_HW = (83, 442)
PRE_T = 0.5          # pre-onset window t < PRE_T in s, fixed and shared by H and
#                      alpha_rms; equals diagnose_7metrics_hallucination.FIXED_PRE_T.
STEADY_T_START = 2.5  # start of the steady segment in s (truth template of alpha_rms)


def _load_fn(modfile, *names):
    """Import functions from a file in scripts/ by its path."""
    spec = importlib.util.spec_from_file_location(f"_m_{Path(modfile).stem}", _REPO / "scripts" / modfile)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return tuple(getattr(mod, n) for n in names)


def _raster_uv(u_p2, rw, n):
    """(n,T,Np2,2) -> (n,T,H,W,2) via the raster weights."""
    T, Np2 = u_p2.shape[1], u_p2.shape[2]
    g = apply_raster_weights(rw, u_p2.transpose(0, 1, 3, 2).reshape(n * T, 2, Np2).transpose(0, 2, 1))
    return g.reshape(n, T, GRID_HW[0], GRID_HW[1], 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["interp", "extrap"], default="interp")
    args = ap.parse_args()
    SPLIT = args.split

    (compute_metrics,) = _load_fn("diagnose_7metrics_hallucination.py", "compute_metrics")
    (pattern_match_pre,) = _load_fn("diagnose_structural_hallucination.py", "pattern_match_pre")

    # --- truth, geometry, eval nodes ---
    truth = np.load(_REPO / f"results/benchmark/truth/kvs_{SPLIT}.npz", allow_pickle=False)
    u_true = truth["u_eval"]                                  # (Nc,T,Np2,2)
    case_ids = [str(c) for c in truth["case_ids"]]
    t_grid = truth["t_grid"]
    spec = load_split("kvs", SPLIT)
    eval_idx_p2 = np.asarray(spec.eval_idx_p2, dtype=np.int64)
    probe_local = np.asarray(spec.probe_local_idx, dtype=np.int64)
    mesh = np.load(_REPO / "meshes/refined_4x/mesh.npz", allow_pickle=False)
    coords_eval = mesh["coords_p2"][eval_idx_p2]
    nC = len(case_ids)
    print(f"[{SPLIT}] truth u={u_true.shape}, cases={nC}", flush=True)

    # raster setup (for alpha_rms); rasterise the truth once
    rw = precompute_raster_weights(coords_eval, GRID_HW, KVS_BBOX)
    pre_mask = t_grid < PRE_T
    steady_mask = t_grid >= STEADY_T_START
    u_true_grid = _raster_uv(u_true, rw, nC)
    print(f"[raster] truth grid {u_true_grid.shape}", flush=True)

    out = {}
    for m in active_kvs_2a():
        for rd in kvs_rundirs(m, SPLIT):
            p = _REPO / "results/benchmark/runs" / rd / "predictions.npz"
            if not p.exists():
                print(f"  [skip] {rd}: no predictions.npz", flush=True); continue
            z = np.load(p, allow_pickle=False)
            if "u_eval" not in z.files or "case_ids" not in z.files:
                print(f"  [skip] {rd}: no u_eval/case_ids", flush=True); continue
            pred_ids = [str(c) for c in z["case_ids"]]
            pos = {c: i for i, c in enumerate(pred_ids)}
            if not all(c in pos for c in case_ids):
                print(f"  [skip] {rd}: case_ids mismatch", flush=True); continue
            sel = np.array([pos[c] for c in case_ids], dtype=np.int64)
            u_pred = z["u_eval"][sel]
            if u_pred.shape != u_true.shape:
                print(f"  [skip] {rd}: shape {u_pred.shape} != {u_true.shape}", flush=True); continue
            cm = compute_metrics(u_pred, u_true, t_grid, coords_eval, probe_local, n_strips=4)
            strip_mean = float(np.nanmean([cm.get(f"strip{k}_rmse", np.nan) for k in range(4)]))
            # H uses the fixed pre-onset window t < 0.5 s, the same window that
            # pattern_match_pre uses below for alpha_rms.
            H = cm["halluc_pixel_frac_pre"]
            u_pred_grid = _raster_uv(u_pred, rw, nC)
            pm = pattern_match_pre(u_pred_grid, u_true_grid, pre_mask, steady_mask)
            out[rd] = {"strip_k_rmse": strip_mean,
                       "halluc_pixel_frac_pre": (None if H is None else float(H)),
                       "alpha_rms": float(np.mean(pm["alpha_rms"]))}
            print(f"  {rd:48s} strip={strip_mean:.4f} H={out[rd]['halluc_pixel_frac_pre']} "
                  f"alpha_rms={out[rd]['alpha_rms']:.4f}", flush=True)

    from benchmark.paper_models import registry_suffix
    _sfx = registry_suffix()
    of = _REPO / "reports" / f"kvs_diag_paper{_sfx}_{SPLIT}.json"
    of.write_text(json.dumps(out, indent=2))
    print(f"\n[saved] {of}  ({len(out)} run dirs)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
