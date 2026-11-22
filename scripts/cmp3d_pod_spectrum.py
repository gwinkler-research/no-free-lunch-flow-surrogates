#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cumulative POD explained-variance spectrum on cmp3d_full100 (TRAIN split).

Uses the snapshot and mean-centring convention of scripts/build_cmp3d_pod.py (same
Cmp3dDatasetView, same physical de-normalisation, same t_stride), but keeps the FULL
eigenvalue spectrum to report the cumulative EVR on a k-grid and the k needed per
threshold. Same output format as reports/kvs_pod_spectrum.json.

  python scripts/cmp3d_pod_spectrum.py --split interp --t_stride 2
Produces reports/cmp3d_pod_spectrum_fix.json (Fig. pod-spectra; Methods, "eight modes",
"99.1 %"). Needs the CMP FEM training tensor, which is not in this repository (--npz).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np

from benchmark.cmp3d_data import Cmp3dDatasetView
from benchmark.splits import load_split

REPO = Path(_REPO_ROOT)
NPZ = REPO / "data" / "cmp" / "cmp3d_full100fix_fem_training.npz"   # CMP reference data
K_GRID = [1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 139]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="interp", choices=["interp", "extrap"])
    ap.add_argument("--t_stride", type=int, default=2)
    ap.add_argument("--npz", default=None, help="path of the CMP FEM training tensor (overrides the default)")
    args = ap.parse_args()
    t0 = time.time()

    split_spec = load_split("cmp", args.split)
    view = Cmp3dDatasetView("train", Path(args.npz) if args.npz else NPZ, split_spec)
    ids = view.case_ids
    st = view.states(ids)                                   # (N,T,Node,3) norm
    phys = st * view.stats_std + view.stats_mean
    N, T, Node, C = phys.shape
    ts = np.arange(0, T, args.t_stride)
    X = phys[:, ts].reshape(-1, Node * C).astype(np.float64)  # (n_snap, D)
    n_snap, D = X.shape
    print(f"[spec] snapshots={n_snap} D={D} (N={N} T_sub={len(ts)})", flush=True)

    Xc = X - X.mean(0)
    G = Xc @ Xc.T                                            # (n_snap, n_snap)
    w = np.linalg.eigvalsh(G)                                # ascending, all modes
    w = np.clip(w[::-1], 0.0, None)                          # descending
    total = float(w.sum())
    cum = np.cumsum(w) / total

    def cum_at(k):
        return float(cum[min(k, len(cum)) - 1])

    cumEVR = {str(k): cum_at(k) for k in K_GRID if k <= len(cum)}

    def modes_for(frac):
        idx = int(np.searchsorted(cum, frac) + 1)
        return min(idx, len(cum))

    out = {
        "dataset": "cmp3d",
        "split": args.split,
        "snapshots": int(n_snap),
        "D": int(D),
        "n_train_cases": int(N),
        "t_stride": int(args.t_stride),
        "cumEVR": cumEVR,
        "modes_for_90pct": modes_for(0.90),
        "modes_for_95pct": modes_for(0.95),
        "modes_for_99pct": modes_for(0.99),
        "modes_for_999pct": modes_for(0.999),
    }
    out_path = REPO / "reports" / "cmp3d_pod_spectrum_fix.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[spec] EVR@1={cum_at(1):.4f} @8={cum_at(8):.4f} @12={cum_at(12):.4f} "
          f"| 90%@{out['modes_for_90pct']} 99%@{out['modes_for_99pct']} "
          f"({time.time()-t0:.0f}s)", flush=True)
    print(f"[spec] wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
