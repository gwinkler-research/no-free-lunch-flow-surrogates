# -*- coding: utf-8 -*-
"""Cumulative POD explained-variance spectrum of the wake training fields (interp TRAIN split).

Snapshot POD on the velocity components (u, v) of the 83 x 442 raster cache that feeds the
frozen encoder (Methods and Fig. pod-spectra: number of modes for 99 % energy and the share
of the leading mode, emitted as \\vKvsPodModesNinetyNine and \\vKvsPodModeOnePct).
Mean-centred, exact Gram eigendecomposition, same output format as
reports/cmp3d_pod_spectrum_fix.json. Needs the raster cache, which is not in this repository.

Usage:
  python scripts/kvs_pod_spectrum.py --split interp --t_stride 4 \
      --out reports/kvs_pod_spectrum.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from benchmark.runner import GridCacheView  # noqa: E402
from benchmark.splits import load_split  # noqa: E402

REPO = Path(_REPO_ROOT)
K_GRID = [1, 2, 4, 8, 12, 16, 24, 31, 32, 48, 64, 96, 128, 162, 192, 256]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--split", default="interp", choices=["interp", "extrap"])
    ap.add_argument("--t_stride", type=int, default=4)
    ap.add_argument("--cache", default="results/benchmark/cache/kvs_grid_83x442.npz")
    ap.add_argument("--channels", default="uv", choices=["uv", "uvp"])
    ap.add_argument("--out", default="reports/kvs_pod_spectrum.json")
    a = ap.parse_args(argv)
    t0 = time.time()
    spec = load_split("kvs", a.split)
    gv = GridCacheView(REPO / a.cache)
    ids = list(spec.train_ids)
    nch = 2 if a.channels == "uv" else 3
    ts = np.arange(0, 201, a.t_stride)
    H, W = gv.grid_hw
    D = H * W * nch
    X = np.empty((len(ids) * len(ts), D), np.float32)
    for i, c in enumerate(ids):
        fr = gv.frames_for([c], dtype=np.float32)[0][ts][..., :nch]      # (T_sub,H,W,nch)
        X[i * len(ts):(i + 1) * len(ts)] = fr.reshape(len(ts), D)
        if i % 100 == 0:
            print(f"[spec] loaded {i}/{len(ids)} cases ({time.time()-t0:.0f} s)", flush=True)
    n_snap = X.shape[0]
    print(f"[spec] snapshots={n_snap} D={D} (N={len(ids)} T_sub={len(ts)})", flush=True)
    X -= X.mean(0, keepdims=True)
    G = (X @ X.T).astype(np.float64)
    del X
    w = np.linalg.eigvalsh(G)
    w = np.clip(w[::-1], 0.0, None)
    cum = np.cumsum(w) / float(w.sum())

    def cum_at(k):
        return float(cum[min(k, len(cum)) - 1])

    def modes_for(frac):
        return int(min(np.searchsorted(cum, frac) + 1, len(cum)))

    out = {"dataset": "kvs", "split": a.split, "snapshots": int(n_snap), "D": int(D),
           "n_train_cases": len(ids), "t_stride": int(a.t_stride), "grid_hw": [int(H), int(W)],
           "channels": a.channels, "cache": a.cache,
           "cumEVR": {str(k): cum_at(k) for k in K_GRID if k <= len(cum)},
           "modes_for_90pct": modes_for(0.90), "modes_for_95pct": modes_for(0.95),
           "modes_for_99pct": modes_for(0.99), "modes_for_999pct": modes_for(0.999)}
    p = REPO / a.out
    p.write_text(json.dumps(out, indent=2))
    print(f"[spec] EVR@1={cum_at(1):.4f} @8={cum_at(8):.4f} @31={cum_at(31):.4f} @32={cum_at(32):.4f} "
          f"| 95%@{out['modes_for_95pct']} 99%@{out['modes_for_99pct']} 99.9%@{out['modes_for_999pct']} "
          f"({time.time()-t0:.0f} s)", flush=True)
    print(f"[spec] wrote {p}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
