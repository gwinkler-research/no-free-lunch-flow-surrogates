# -*- coding: utf-8 -*-
"""Does the checkpoint choice of the KVS field DeepONet explain its damping?

The direct operator selects its best checkpoint on the validation loss of the FIRST 32 of 201
time steps (ramp window, benchmark/adapters/kvs_field_deeponet.py). This script re-infers the
v-component at the twelve wake probes for best.pt (chosen on the ramp) and last.pt (epoch 4000)
and reports the shedding power P = sum var(v_pred) / sum var(v_true) over case x probe in the
steady window t >= 2 s, plus the probe RMSE. If last.pt damps alike, the checkpoint choice is not
what damps the direct operator (Sec. 4.3 / Sec. 7 argument).

Needs the run directories, the truth NPZ and the raw KVS data, which are not in this repository.

Usage:
  python scripts/field_don_checkpoint_probe_check.py --split interp --seeds 0 1 2 \
      --runs-root <directory with the run directories> --data-root <raw KVS data> \
      --out results/benchmark/field_don_checkpoint_probe_check_interp.json
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
import torch

from benchmark.adapters import _common
from benchmark.adapters.kvs_field_deeponet import KvsFieldDeepONetAdapter, _norm_xy
from benchmark.runner import _make_views, load_config
from benchmark.splits import load_split

REPO = Path(_REPO_ROOT)
STEADY_T = 2.0


def probe_traces(ms, ckpt, test, probe_xy, dev):
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    model = ms["model"]
    model.load_state_dict(sd["state_dict"])
    model.to(dev).eval()
    bbox = np.asarray(sd["coord_bbox"], dtype=np.float64)
    th_mean, th_std = np.asarray(sd["theta_mean"], np.float32), np.asarray(sd["theta_std"], np.float32)
    u_mean = np.asarray(sd.get("u_mean", np.zeros(2, np.float32)), np.float32)
    u_std = np.asarray(sd.get("u_std", np.ones(2, np.float32)), np.float32)
    ids = test.case_ids
    n = len(ids)
    t_grid = np.asarray(test.t_grid, dtype=np.float64)
    T = t_grid.size
    t_n = torch.tensor((t_grid / max(float(t_grid[-1]), 1e-9)).astype(np.float32), device=dev)
    th = torch.tensor((np.asarray(test.theta, np.float32) - th_mean) / th_std, dtype=torch.float32, device=dev)
    s_np = np.stack([_common.ramp_s_dsdt(np.asarray(test.theta)[i, 1:], t_grid)[0] for i in range(n)], 0).astype(np.float32)
    s_t = torch.tensor(s_np, device=dev)
    xy = torch.tensor(_norm_xy(np.asarray(probe_xy, np.float64), bbox), device=dev)
    v = np.empty((n, T, xy.shape[0]), np.float32)
    with torch.no_grad():
        for i in range(n):
            o = model(s_t[i:i + 1].unsqueeze(-1), th[i:i + 1], xy, t_n)   # (1,T,M,3)
            v[i] = o[0, ..., 1].cpu().numpy() * u_std[1] + u_mean[1]
    return v, int(sd["epoch"]), t_grid


def power_ratio(vp, vt, t_grid):
    m = t_grid >= STEADY_T - 1e-9
    return float(np.var(vp[:, m, :], axis=1).sum() / np.var(vt[:, m, :], axis=1).sum())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--split", default="interp", choices=["interp", "extrap"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--runs-root", default=str(REPO / "results" / "benchmark" / "runs"))
    ap.add_argument("--truth", default=None, help="truth NPZ kvs_<split>.npz (default: <runs-root>/../truth/kvs_<split>.npz)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--data-root", default=str(REPO / "results" / "refined_4x_2026-05-07"),
                    help="root holding chunk_*/Umax_*_tau_*/fields.h5 (raw KVS data, not in this repository)")
    a = ap.parse_args(argv)
    cfg = load_config(REPO / f"configs/paper_kvs_{a.split}_field_deeponet.yaml")
    cfg = dict(cfg)
    cfg["chunk_roots"] = [str(Path(a.data_root) / Path(c).name) for c in cfg["chunk_roots"]]
    cfg["require_h5"] = True
    spec = load_split("kvs", a.split)
    truth_path = Path(a.truth) if a.truth else Path(a.runs_root).parent / "truth" / f"kvs_{a.split}.npz"
    with np.load(truth_path, allow_pickle=False) as z:
        truth = {k: z[k] for k in z.files}
    dev = torch.device(a.device)
    out = {"meta": {"split": a.split, "steady_t": STEADY_T, "truth": str(truth_path),
                    "runs_root": a.runs_root, "date": time.strftime("%Y-%m-%d")}, "seeds": {}}
    for seed in a.seeds:
        views, _ = _make_views(cfg, seed)
        test = views["test"]
        adapter = KvsFieldDeepONetAdapter(dict(cfg))
        ms = adapter.build(dict(cfg), seed)
        ms["dev"] = dev
        pos = {str(c): i for i, c in enumerate(map(str, truth["case_ids"]))}
        perm = np.array([pos[str(c)] for c in test.case_ids])
        vt = truth["u_eval"][perm][:, :, spec.probe_local_idx, 1]
        run_dir = Path(a.runs_root) / f"kvs_field_deeponet_norm_{a.split}_seed{seed}"
        res = {}
        for name in ("best.pt", "last.pt"):
            t0 = time.time()
            vp, ep, t_grid = probe_traces(ms, run_dir / name, test, spec.probe_xy, dev)
            m = t_grid >= STEADY_T - 1e-9
            res[name] = {"epoch": ep, "P": power_ratio(vp, vt, t_grid),
                         "probe_rmse_steady": float(np.sqrt(np.mean((vp[:, m] - vt[:, m]) ** 2))),
                         "probe_rmse_full": float(np.sqrt(np.mean((vp - vt) ** 2)))}
            print(f"[ckpt] seed {seed} {name} epoch {ep}: P {res[name]['P']:.4f}, probe RMSE steady "
                  f"{res[name]['probe_rmse_steady']:.4f} ({time.time()-t0:.0f} s)", flush=True)
        out["seeds"][str(seed)] = res
    if a.out:
        p = REPO / a.out
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, indent=1))
        print(f"[ckpt] wrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
