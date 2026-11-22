#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ============================================================================
# Purpose: the linear baseline (DMDc) of the CMP table. Projects the CMP
#   velocity onto the frozen 3D POD basis (rank k = 8), builds the control
#   input from the ramp s and its derivative ds/dt, solves for (A, B) by
#   least squares with a stability constraint, rolls the latent
#   forward from the encoded initial state, decodes and scores. No network
#   training, no GPU. The result is deterministic; --seed only names the
#   output file.
# Pipeline role: writes reports/cmp3d_gate2_cmp3d_dmdc_k8[_extrap]_fix_seed<n>.json,
#   which scripts/build_paper_matrix.py reads for the DMDc row. The metrics
#   come from gate2_metrics() in scripts/run_cmp3d_gate2.py (loaded by file
#   path), so the DMDc row and the trained CMP rows use the same function.
# Eval-consistency: latent and control statistics are fitted on the training
#   cases only and applied unchanged to the test rollout. gate2_metrics
#   normalises prediction and truth with the same NPZ statistics. Defaults that
#   set the numbers: POD rank 8, least-squares rcond 1e-10, stabilisation
#   "constrained" with spectral-radius cap 1.0. The POD basis is read from
#   results/benchmark/encoders/cmp3d_<split>_pod139_fix.npz and must belong
#   to the training NPZ. The split is the frozen benchmark/splits/cmp_<split>.json.
# ============================================================================
"""DMDc (linear baseline) on the CMP-3D POD latent (k = 8), scored with gate2_metrics.

DMDc has no gradient training and no seed dependence, so each split has one run (n = 1);
the JSON carries "deterministic": true.

Building blocks, all imported:
  * frozen 3D POD encoder       benchmark.adapters.cmp3d_latent_ar._Pod3dEncoder
  * lstsq + stabilise, rollout  benchmark.adapters.dmdc._dmdc_solve / _linear_rollout
  * metrics                     scripts/run_cmp3d_gate2.py:gate2_metrics
Control u = [s, ds/dt]: time-varying control in the DMDc sense (Proctor et al.), the same
(s, ds) features the autoregressive rows see; ds via np.gradient along time. Both control
columns and the latent z are normalised with training statistics.

  python3 -u scripts/cmp3d_dmdc_gate2.py --split interp    # -> reports/cmp3d_gate2_cmp3d_dmdc_k8_fix_seed1.json
  python3 -u scripts/cmp3d_dmdc_gate2.py --split extrap    # -> reports/cmp3d_gate2_cmp3d_dmdc_k8_extrap_fix_seed1.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)          # repository root first, so benchmark/ imports

import numpy as np

from benchmark.cmp3d_data import Cmp3dDatasetView
from benchmark.splits import load_split
from benchmark.adapters.cmp3d_latent_ar import _Pod3dEncoder
from benchmark.adapters.dmdc import _dmdc_solve, _linear_rollout

REPO = Path(_REPO_ROOT)
NPZ = REPO / "data" / "cmp" / "cmp3d_full100fix_fem_training.npz"   # CMP reference data (override with --npz)


def _load_gate2_metrics():
    """Import gate2_metrics from scripts/run_cmp3d_gate2.py by file path, without
    putting scripts/ on sys.path."""
    spec = importlib.util.spec_from_file_location("_rcg", REPO / "scripts" / "run_cmp3d_gate2.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.gate2_metrics


def _encode_states(encoder, view):
    ids = view.case_ids
    st = np.asarray(view.states(ids), np.float32)
    phys = st * view.stats_std + view.stats_mean
    z = np.stack([encoder.encode(phys[i]) for i in range(st.shape[0])], 0)  # (N,T,k)
    return z.astype(np.float64), phys


def _controls_s_ds(view) -> np.ndarray:
    """u = [s, ds/dt], shape (N, T, 2): ramp control and its time derivative
    (np.gradient), the same (s, ds) features the autoregressive rows see."""
    s = np.asarray(view.controls(view.case_ids), np.float64)        # (N,T)
    tg = np.asarray(view.t_grid, np.float64)
    dt = float(tg[1] - tg[0]) if len(tg) > 1 else 1.0
    ds = np.gradient(s, dt, axis=1)
    return np.stack([s, ds], axis=-1)                                # (N,T,2)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="interp", choices=["interp", "extrap"])
    ap.add_argument("--pod_k", type=int, default=8)
    ap.add_argument("--npz", default=None, help="path of the CMP FEM training tensor (overrides the default)")
    ap.add_argument("--rcond", type=float, default=1e-10)
    ap.add_argument("--seed", type=int, default=1)  # only names the output file; DMDc is deterministic
    ap.add_argument("--out", default=None, help="override report path (default: reports/cmp3d_gate2_<model>_seed<seed>.json)")
    args = ap.parse_args()
    t0 = time.time()
    gate2_metrics = _load_gate2_metrics()

    split_spec = load_split("cmp", args.split)
    npz_path = Path(args.npz) if args.npz else NPZ
    train = Cmp3dDatasetView("train", npz_path, split_spec)
    test = Cmp3dDatasetView("test", npz_path, split_spec)
    enc_ref = REPO / "results" / "benchmark" / "encoders" / f"cmp3d_{args.split}_pod139_fix.npz"
    encoder = _Pod3dEncoder(enc_ref, k=args.pod_k)
    k = encoder.latent_dim
    print(f"[dmdc3d] split={args.split} k={k} enc={enc_ref.name}", flush=True)

    # --- TRAIN: encode + controls u=[s, ds/dt], normalise on TRAIN only ---
    z_tr, _ = _encode_states(encoder, train)                       # (N,T,k)
    u_tr = _controls_s_ds(train)                                    # (N,T,2)
    zf = z_tr.reshape(-1, k)
    z_mean = zf.mean(0); z_std = zf.std(0) + 1e-8
    uf = u_tr.reshape(-1, u_tr.shape[2])
    u_mean = uf.mean(0); u_std = uf.std(0) + 1e-8
    z_tr_n = (z_tr - z_mean) / z_std
    u_tr_n = (u_tr - u_mean) / u_std

    A, B, info = _dmdc_solve(z_tr_n, u_tr_n, rcond=args.rcond,
                             stab_rho_cap=1.0, stab_method="constrained")
    rho_final = float(info.get("rho_stab", info.get("rho_raw", float("nan"))))
    print(f"[dmdc3d] solved A{A.shape} B{B.shape} rho(A)~{rho_final:.4f} "
          f"stab={info.get('stab_method')}", flush=True)

    # --- TEST: roll out from the encoded t=0 state, decode, score ---
    z_te, phys_te = _encode_states(encoder, test)
    u_te = _controls_s_ds(test)                                     # (N,T,2)
    n, T = z_te.shape[0], z_te.shape[1]
    z0_n = (z_te[:, 0, :] - z_mean) / z_std
    u_te_n = (u_te - u_mean) / u_std
    z_pred_n = _linear_rollout(A, B, z0_n, u_te_n)                  # (n,T,k)
    z_pred = z_pred_n * z_std + z_mean
    Node = phys_te.shape[2]
    pred = np.empty((n, T, Node, 3), np.float32)
    for ci in range(n):
        pred[ci] = encoder.decode(z_pred[ci])
    truth = test.states(test.case_ids) * test.stats_std + test.stats_mean

    model_name = "cmp3d_dmdc_k8_fix" if args.split == "interp" else "cmp3d_dmdc_k8_extrap_fix"

    met = gate2_metrics(pred, truth, test)
    met.update({"model_name": model_name, "adapter": "dmdc_cmp3d", "seed": args.seed,
                "split": args.split, "deterministic": True, "pod_k": k, "rcond": args.rcond,
                "control": "s_dsdt",
                "rho_A_final": rho_final,
                "stab_method": info.get("stab_method"),
                "best_epoch": 0, "total_epochs": 0, "n_params": int(A.size + B.size),
                "wallclock_s": round(time.time() - t0, 1)})
    out = Path(args.out) if args.out else REPO / "reports" / f"cmp3d_gate2_{model_name}_seed{args.seed}.json"
    out.write_text(json.dumps(met, indent=2))
    print(f"[dmdc3d] nrmse_3d={met['nrmse_3d']:.4f} wafer_rmse={met['wafer_rmse']:.4f} "
          f"cumwss_rel_l2_inner={met['cumwss_rel_l2_inner_mean']:.4f} -> {out} "
          f"({met['wallclock_s']}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
