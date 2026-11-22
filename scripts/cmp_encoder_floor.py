# -*- coding: utf-8 -*-
"""CMP encoder floor: decode(encode(truth)) through the frozen POD basis, scored by gate2.

The test-split truth fields are projected onto the first k POD modes and decoded again.
The round trip is scored with the same gate2 pipeline as every film model (inner-90
cum-WSS rel-L2, in-plane nRMSE, ...). This gives the reconstruction floor of Sec. 8 ("its
reconstruction floor lies above the direct operator's error") on the same reference data
and encoder as the printed film rows. values/emit_values.py reads the k = 8 entries.

The FEM training tensor and the POD encoder files are not in this repository; pass them
with --npz and --encoder.

Usage:
  python scripts/cmp_encoder_floor.py --split interp --split extrap \
      --npz <data dir>/cmp3d_full100fix_fem_training.npz \
      --encoder interp=<encoder dir>/cmp3d_interp_pod139_fix.npz \
      --encoder extrap=<encoder dir>/cmp3d_extrap_pod139_fix.npz \
      --k 8 --k 16 --k 32 --out results/benchmark/cmp3d_encoder_floor_fix.json
Output JSON: per split and k the scalar gate2 metrics (cumwss_rel_l2_inner_mean is the
headline).
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

from benchmark.adapters.cmp3d_latent_ar import _Pod3dEncoder
from benchmark.cmp3d_data import make_cmp3d_views
from scripts.run_cmp3d_gate2 import gate2_metrics
from tools.runtime.io import read_yaml_or_json

REPO = Path(_REPO_ROOT)
ENC_DEFAULT = {
    "interp": str(REPO / "results" / "benchmark" / "encoders" / "cmp3d_interp_pod139_fix.npz"),
    "extrap": str(REPO / "results" / "benchmark" / "encoders" / "cmp3d_extrap_pod139_fix.npz"),
}
CFG = {
    "interp": "configs/paper_cmp3d_interp_latent_ar_s4d_k8_fix.yaml",
    "extrap": "configs/paper_cmp3d_extrap_latent_ar_s4d_k8_fix.yaml",
}


def floor_for_split(split: str, npz: str, enc_path: str, ks: list[int]) -> dict:
    cfg = dict(read_yaml_or_json(REPO / CFG[split]))
    cfg["cmp3d_npz"] = npz
    cfg["cmp3d_encoder_ref"] = enc_path
    views, _ = make_cmp3d_views(cfg)
    test = views["test"]
    ids = test.case_ids
    t0 = time.time()
    truth = np.asarray(test.states(ids), np.float32) * test.stats_std + test.stats_mean   # (n,T,Node,3)
    print(f"[floor] {split}: truth {truth.shape} loaded in {time.time()-t0:.0f} s", flush=True)
    out = {"n_cases": int(truth.shape[0]), "case_ids": [str(c) for c in ids],
           "encoder_ref": enc_path, "npz": npz, "config": CFG[split]}
    for k in ks:
        enc = _Pod3dEncoder(Path(enc_path), k=k)
        rec = np.empty_like(truth)
        for i in range(truth.shape[0]):
            rec[i] = enc.decode(enc.encode(truth[i]))
        met = gate2_metrics(rec, truth, test)
        out[f"k{k}"] = {kk: (float(v) if np.isscalar(v) else v) for kk, v in met.items()
                        if np.isscalar(v)}
        print(f"[floor] {split} k={k}: cwl2 {met.get('cumwss_rel_l2_inner_mean', float('nan')):.4f}  "
              f"nrmse_ip {met.get('nrmse_inplane', float('nan')):.4f}  ({time.time()-t0:.0f} s)", flush=True)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--split", action="append", choices=["interp", "extrap"])
    ap.add_argument("--npz", default=str(REPO / "data" / "cmp" / "cmp3d_full100fix_fem_training.npz"))
    ap.add_argument("--encoder", action="append", default=None,
                    help="split=path override, e.g. interp=/path/pod.npz")
    ap.add_argument("--k", action="append", type=int)
    ap.add_argument("--seed", type=int, default=1, help="recorded in the output; the floor has no randomness")
    ap.add_argument("--out", default="results/benchmark/cmp3d_encoder_floor_fix.json")
    a = ap.parse_args(argv)
    splits = a.split or ["interp", "extrap"]
    ks = a.k or [8]
    enc = dict(ENC_DEFAULT)
    for e in a.encoder or []:
        s, p = e.split("=", 1); enc[s] = p
    res = {"meta": {"pipeline": "decode(encode(truth)) via _Pod3dEncoder(k) -> gate2_metrics "
                                "(identical to the film model evaluation)",
                    "date": time.strftime("%Y-%m-%d"), "seed": a.seed, "ks": ks}}
    for s in splits:
        res[s] = floor_for_split(s, a.npz, enc[s], ks)
    outp = REPO / a.out
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(res, indent=1))
    print(f"[floor] wrote {outp}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
