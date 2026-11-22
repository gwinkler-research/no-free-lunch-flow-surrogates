#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build the frozen film POD encoder from the training split (Sec. 4.2: k = 8 of 139 stored modes).

Snapshot POD on the physical 3D velocity field (T, Node, 3) flattened to D = Node*3.
Stores basis (k, D), mean (D,), n_modes, n_node, n_channels, evr and split in
results/benchmark/encoders/cmp3d_<split>_pod<k>_fix.npz, or in the file given with --out.
The 3D latent-AR adapter projects and reconstructs with this basis (channel layout (Node, 3)).

  python scripts/build_cmp3d_pod.py --split interp --k 139
cmp3d_<split>_pod139_fix.npz is the encoder_ref of every film latent row. Needs the CMP FEM
training tensor, which is not in this repository (--npz).
"""
from __future__ import annotations

import argparse
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
NPZ = REPO / "data" / "cmp" / "cmp3d_full100fix_fem_training.npz"   # CMP reference data (override with --npz)
OUT_DIR = REPO / "results" / "benchmark" / "encoders"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="interp", choices=["interp", "extrap"])
    ap.add_argument("--k", type=int, default=139)
    ap.add_argument("--t_stride", type=int, default=2, help="time subsample for snapshots")
    ap.add_argument("--npz", default=None, help="path of the CMP FEM training tensor (overrides the default)")
    ap.add_argument("--out", default=None, help="output npz (default: encoders/cmp3d_<split>_pod<k>_fix.npz)")
    args = ap.parse_args()
    t0 = time.time()

    split_spec = load_split("cmp", args.split)
    npz_path = Path(args.npz) if args.npz else NPZ
    print(f"[pod] data source: {npz_path}", flush=True)
    view = Cmp3dDatasetView("train", npz_path, split_spec)
    ids = view.case_ids
    sm = view.stats_mean; ss = view.stats_std
    st = view.states(ids)                              # (N,T,Node,3) norm
    phys = st * ss + sm
    N, T, Node, C = phys.shape
    ts = np.arange(0, T, args.t_stride)
    X = phys[:, ts].reshape(-1, Node * C).astype(np.float64)   # (n_snap, D)
    print(f"[pod3d] snapshots={X.shape[0]} D={X.shape[1]} (N={N} T_sub={len(ts)})", flush=True)

    mean = X.mean(0)
    Xc = X - mean
    # snapshot method: eigendecomposition of the (n_snap x n_snap) Gram matrix
    G = Xc @ Xc.T
    w, V = np.linalg.eigh(G)                            # ascending
    order = np.argsort(w)[::-1]
    w = w[order]; V = V[:, order]
    k = min(args.k, (w > 1e-12).sum())
    w_k = w[:k]; V_k = V[:, :k]
    basis = (Xc.T @ V_k) / np.sqrt(np.maximum(w_k, 1e-12))   # (D, k)
    basis = basis.T                                          # (k, D)
    evr = float(w_k.sum() / w.sum())
    print(f"[pod3d] k={k}  cumulative EVR={evr:.4f}  ({time.time()-t0:.0f}s)", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = Path(args.out) if args.out else OUT_DIR / f"cmp3d_{args.split}_pod{k}_fix.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    with open(tmp, "wb") as fh:
        np.savez(fh, basis=basis.astype(np.float32), mean=mean.astype(np.float32),
                 n_modes=np.int64(k), n_node=np.int64(Node), n_channels=np.int64(C),
                 evr=np.float32(evr), split=args.split)
    tmp.replace(out)
    print(f"[pod3d] wrote {out} ({out.stat().st_size/1e6:.1f} MB)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
