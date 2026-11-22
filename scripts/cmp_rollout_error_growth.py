#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ============================================================================
# Purpose: resolve the CMP bulk error along the trajectory, to compare the
#   step-by-step growth of the rollout error of the latent autoregressive rows
#   with the one-shot rows. For every CMP row on the interpolation
#   split it re-infers the full 3D velocity field through the SAME code path as
#   the reported numbers (scripts/run_cmp3d_gate2.py --load-best: build,
#   one-epoch fit for the adapter statistics, best.pt over the net,
#   predict_eval; the DMDc row through the solve of
#   scripts/cmp3d_dmdc_gate2.py) and evaluates the scorer's in-plane nRMSE per
#   time step instead of pooled over the trajectory. Per model and seed it
#   keeps the mean and the median over the test cases, then the seed mean, and
#   two summary numbers: the error at the first predicted step and at the last
#   step, with their ratio.
# Pipeline role: Results CMP and Discussion (rollout error growth of the
#   latent-AR rows against the one-shot rows); Fig. cmp-rollout-growth. Reads
#   the run directories (best.pt only), the CMP FEM training tensor and the POD
#   encoder, none of which is in this repository (--runs-root, --npz,
#   --encoder-root), and the per-row configs
#   configs/paper_cmp3d_interp_*_fix.yaml. Writes
#   results/benchmark/cmp_rollout_error_growth_interp.json and, optionally, a
#   quick-look PNG.
# Eval-consistency: no metric is reimplemented. The per-step number is the
#   scorer's own expression err = (pred - mean)/std - (truth - mean)/std on the
#   in-plane channels (gate2_metrics, nrmse_inplane), only with the root mean
#   square taken over nodes and channels of one (case, step) instead of over
#   the whole (case, step, node, channel) block. Every (case, step) cell has
#   the same number of entries, so the root of the mean of the squared per-step
#   curve is the pooled nrmse_inplane. That identity is the self-check, written
#   next to the scorer's gate2_metrics() value on the same prediction and next
#   to the stored reinfer JSON. The one-epoch statistics fit runs in a scratch
#   directory, so best.pt is only read and nothing in the run directory is
#   written. Autoregressive rows and DMDc start from the POD projection of the
#   TRUE initial state, so their step 0 is the encoder floor at t = 0 and their
#   first predicted step is index 1; the one-shot rows predict every step
#   including index 0. Both index conventions are recorded.
# ============================================================================
"""Per-time-step in-plane nRMSE of every CMP row (interp split).

  CUDA_VISIBLE_DEVICES=0 python3 -u scripts/cmp_rollout_error_growth.py \
      [--models gru_k8,oneshot_gru] [--seeds 1,2,3]
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import platform
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))          # repo root first, so scripts/ cannot shadow packages

import numpy as np
import torch

from benchmark.cmp3d_data import Cmp3dDatasetView, make_cmp3d_views     # noqa: E402
from benchmark.splits import load_split                                 # noqa: E402
from benchmark.adapters.cmp3d_latent_ar import _Pod3dEncoder            # noqa: E402
from benchmark.adapters.dmdc import _dmdc_solve, _linear_rollout        # noqa: E402
from benchmark.paper_models import CMP_2B                               # noqa: E402
from tools.runtime.io import read_yaml_or_json                          # noqa: E402
from tools.common.seeding import set_seed                               # noqa: E402

RUNS_ROOT = _REPO / "results" / "benchmark" / "runs"
NPZ = _REPO / "data" / "cmp" / "cmp3d_full100fix_fem_training.npz"
ENC_ROOT = _REPO / "results" / "benchmark" / "encoders"
OUT = _REPO / "results" / "benchmark" / "cmp_rollout_error_growth_interp.json"
SPLIT = "interp"
CELL_FIRST_PRED = {"latent×AR": 1, "latent×one-shot": 0, "direct×one-shot": 0}


def _load_by_path(name: str, rel: str):
    """Import a scripts/ module by file path without putting scripts/ on sys.path
    (same guard as scripts/cmp3d_dmdc_gate2.py::_load_gate2_metrics)."""
    spec = importlib.util.spec_from_file_location(name, _REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def _git(repo: Path, *args) -> str:
    try:
        return subprocess.run(["git", "-C", str(repo), *args], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:                       # pragma: no cover
        return "n/a"


def per_step_errors(pred, truth, sm, ss):
    """Scorer-identical normalised error, resolved per (case, step).

    err = (pred - sm)/ss - (truth - sm)/ss exactly as gate2_metrics; the rms is
    taken over nodes x in-plane channels (and over all three channels for the
    secondary 3-channel curve). Returns e_ip, e_3d, truth_rms_ip: each (n, T)."""
    n, T = pred.shape[:2]
    e_ip = np.empty((n, T)); e_3d = np.empty((n, T)); tr_ip = np.empty((n, T))
    for c in range(n):
        pn = (pred[c] - sm) / ss
        tn = (truth[c] - sm) / ss
        err = pn - tn                                            # (T, Node, 3) float32
        e2 = err.astype(np.float64) ** 2
        e_ip[c] = np.sqrt(e2[..., :2].mean(axis=(1, 2)))
        e_3d[c] = np.sqrt(e2.mean(axis=(1, 2)))
        tr_ip[c] = np.sqrt((tn.astype(np.float64)[..., :2] ** 2).mean(axis=(1, 2)))
        del pn, tn, err, e2
    return e_ip, e_3d, tr_ip


def infer_trained(cfg_path: Path, seed: int, gate2, args):
    """The --load-best flow of scripts/run_cmp3d_gate2.py::main, minus the JSON."""
    cfg = dict(read_yaml_or_json(cfg_path))
    cfg.setdefault("model_name", cfg_path.stem)
    cfg["cmp3d_npz"] = str(args.npz)
    if "cmp3d_encoder_ref" in cfg:
        cfg["cmp3d_encoder_ref"] = str(Path(args.encoder_root) / Path(cfg["cmp3d_encoder_ref"]).name)
    set_seed(seed)
    views, _ = make_cmp3d_views(cfg)
    train, val, test = views["train"], views["val"], views["test"]
    adapter = gate2._get_adapter(cfg["adapter"])
    run_dir = Path(args.runs_root) / f"{cfg['model_name']}_seed{seed}"
    cfg["ckpt_dir"] = str(run_dir)
    ms = adapter.build(cfg, seed)
    cfg_s = copy.deepcopy(cfg)
    cfg_s["model"]["epochs"] = 1; cfg_s["model"]["n_epochs"] = 1
    cfg_s["ckpt_dir"] = str(Path(args.tmp_root) / f"{cfg['model_name']}_seed{seed}_loadbest_tmp")
    ms, _ = adapter.fit(ms, train, val, cfg_s, seed)
    ck = run_dir / "best.pt"
    sd = torch.load(ck, map_location="cpu", weights_only=False)
    state = sd["state_dict"] if isinstance(sd, dict) and "state_dict" in sd else sd
    net = ms.get("model") if ms.get("model") is not None else ms.get("dyn")
    net.load_state_dict(state)
    pred = adapter.predict_eval(ms, test, cfg)                   # (n, T, Node, 3) phys
    truth = test.states(test.case_ids) * test.stats_std + test.stats_mean
    prov = {"run_dir": str(run_dir), "checkpoint": str(ck), "checkpoint_sha256": _sha256(ck),
            "checkpoint_epoch": (sd.get("epoch") if isinstance(sd, dict) else None),
            "config": str(cfg_path.relative_to(_REPO)), "adapter": cfg["adapter"],
            "n_params": int(ms["n_params"]), "stats_fit_tmp_dir": cfg_s["ckpt_dir"]}
    return pred, truth, test, prov


def infer_dmdc(args, pod_k=8, rcond=1e-10):
    """The DMDc flow of scripts/cmp3d_dmdc_gate2.py::main, using its own
    helpers; nothing re-derived."""
    dm = _load_by_path("_dmdc3d", "scripts/cmp3d_dmdc_gate2.py")
    split_spec = load_split("cmp", SPLIT)
    train = Cmp3dDatasetView("train", args.npz, split_spec)
    test = Cmp3dDatasetView("test", args.npz, split_spec)
    enc_ref = Path(args.encoder_root) / f"cmp3d_{SPLIT}_pod139_fix.npz"
    encoder = _Pod3dEncoder(enc_ref, k=pod_k)
    k = encoder.latent_dim
    z_tr, _ = dm._encode_states(encoder, train)
    u_tr = dm._controls_s_ds(train)
    zf = z_tr.reshape(-1, k); z_mean = zf.mean(0); z_std = zf.std(0) + 1e-8
    uf = u_tr.reshape(-1, u_tr.shape[2]); u_mean = uf.mean(0); u_std = uf.std(0) + 1e-8
    A, B, info = _dmdc_solve((z_tr - z_mean) / z_std, (u_tr - u_mean) / u_std, rcond=rcond,
                             stab_rho_cap=1.0, stab_method="constrained")
    z_te, phys_te = dm._encode_states(encoder, test)
    u_te = dm._controls_s_ds(test)
    n, T = z_te.shape[0], z_te.shape[1]
    z_pred = _linear_rollout(A, B, (z_te[:, 0, :] - z_mean) / z_std, (u_te - u_mean) / u_std) * z_std + z_mean
    pred = np.empty((n, T, phys_te.shape[2], 3), np.float32)
    for ci in range(n):
        pred[ci] = encoder.decode(z_pred[ci])
    truth = test.states(test.case_ids) * test.stats_std + test.stats_mean
    prov = {"run_dir": None, "checkpoint": None, "checkpoint_sha256": None,
            "config": "closed-form (scripts/cmp3d_dmdc_gate2.py, fix data, fix POD basis)",
            "adapter": "dmdc_cmp3d", "pod_k": k, "rcond": rcond, "stab_method": info.get("stab_method"),
            "n_params": int(A.size + B.size)}
    return pred, truth, test, prov


def _summ(curve: np.ndarray, first_idx: int, t_grid: np.ndarray) -> dict:
    ramp = t_grid < 1.0; steady = t_grid >= 1.0
    e_first = float(curve[first_idx]); e_last = float(curve[-1])
    return {"first_pred_index": int(first_idx),
            "err_first_pred_step": e_first, "err_last_step": e_last,
            "ratio_last_over_first": e_last / e_first if e_first > 0 else None,
            "err_index0": float(curve[0]), "err_index1": float(curve[1]),
            "ratio_last_over_index1": e_last / float(curve[1]) if curve[1] > 0 else None,
            "err_max": float(curve.max()), "argmax_index": int(curve.argmax()),
            "mean_ramp_t_lt_1": float(curve[ramp].mean()), "mean_steady_t_ge_1": float(curve[steady].mean())}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs-root", default=str(RUNS_ROOT))
    ap.add_argument("--npz", default=str(NPZ))
    ap.add_argument("--encoder-root", default=str(ENC_ROOT))
    ap.add_argument("--tmp-root", default=os.environ.get("TMPDIR", "/tmp"),
                    help="scratch dir for the one-epoch statistics fit (never the run dir)")
    ap.add_argument("--models", default=None, help="comma list of CMP_2B keys (default: all)")
    ap.add_argument("--seeds", default=None, help="comma list of seeds (default: the seeds in benchmark/paper_models.py)")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--skip-gate2-check", action="store_true",
                    help="skip the gate2_metrics() self-check on the re-inferred field")
    args = ap.parse_args(argv)
    args.npz = Path(args.npz)
    Path(args.tmp_root).mkdir(parents=True, exist_ok=True)
    gate2 = _load_by_path("_rcg", "scripts/run_cmp3d_gate2.py")
    want = set(args.models.split(",")) if args.models else None
    seeds_override = [int(s) for s in args.seeds.split(",")] if args.seeds else None
    t_start = time.time()

    out = {"split": SPLIT, "metric": {
        "name": "nrmse_inplane_per_step",
        "definition": "for case c and step t: sqrt(mean over nodes and channels (u_x,u_y) of "
                      "((pred - stats_mean)/stats_std - (truth - stats_mean)/stats_std)^2); "
                      "stats_mean/stats_std = test.stats_mean/test.stats_std of the FEM tensor "
                      "(same normalisation as scripts/run_cmp3d_gate2.py::gate2_metrics nrmse_inplane)",
        "pooled_identity": "sqrt(mean over (c,t) of curve[c,t]^2) == gate2_metrics nrmse_inplane",
        "aggregation": "curve_mean = mean over test cases, curve_median = median over test cases, "
                       "seed_mean = mean over seeds of the per-seed curves",
        "secondary": "nrmse_3d_per_step (same, all three channels) and truth_rms_inplane_per_step "
                     "(sqrt mean of the normalised truth, for reference only)",
        "source_function": "scripts/run_cmp3d_gate2.py::gate2_metrics (expression reused, "
                           "per-step rms in scripts/cmp_rollout_error_growth.py::per_step_errors)"},
        "models": {}}

    t_grid = None
    for row in CMP_2B:
        key = row["key"]
        if want and key not in want:
            continue
        seeds = seeds_override or list(row["seeds"])
        if row.get("dmdc"):
            seeds = [1]
        first_idx = CELL_FIRST_PRED[row["cell"]]
        mrec = {"label": row["label"], "cell": row["cell"], "role": row["role"],
                "first_pred_index": first_idx, "seeds": {}, "seed_mean": {}, "summary": {}}
        for seed in seeds:
            t0 = time.time()
            print(f"[growth] {key} seed{seed} ...", flush=True)
            if row.get("dmdc"):
                pred, truth, test, prov = infer_dmdc(args)
                model_name = "cmp3d_dmdc_k8_fix"
                rein = _REPO / "reports" / f"cmp3d_gate2_{model_name}_seed{seed}.json"
            else:
                cfg_path = _REPO / "configs" / f"paper_cmp3d_{SPLIT}_{row['reinfer_key']}_fix.yaml"
                pred, truth, test, prov = infer_trained(cfg_path, seed, gate2, args)
                model_name = f"{row['reinfer_key']}_fix"
                rein = _REPO / "reports" / "_reinfer_fix" / f"cmp3d_6metric_{SPLIT}_{model_name}_seed{seed}.json"
            tg = np.asarray(test.t_grid, np.float64)
            if t_grid is None:
                t_grid = tg
                out.update({"n_test_cases": int(pred.shape[0]), "n_steps": int(pred.shape[1]),
                            "n_nodes": int(pred.shape[2]), "dt": float(tg[1] - tg[0]),
                            "t_grid": tg.tolist(), "test_case_ids": [str(c) for c in test.case_ids],
                            "stats_mean": test.stats_mean.tolist(), "stats_std": test.stats_std.tolist()})
            e_ip, e_3d, tr_ip = per_step_errors(pred, truth, test.stats_mean, test.stats_std)
            out.setdefault("truth_rms_inplane_per_step", np.sqrt((tr_ip ** 2).mean(0)).tolist())
            pooled = float(np.sqrt((e_ip ** 2).mean()))
            srec = {"model_name": model_name, "curve_mean": e_ip.mean(0).tolist(),
                    "curve_median": np.median(e_ip, 0).tolist(),
                    "curve_mean_3ch": e_3d.mean(0).tolist(),
                    "per_case": {str(c): e_ip[i].tolist() for i, c in enumerate(test.case_ids)},
                    "pooled_nrmse_inplane_from_curve": pooled, **prov}
            if not args.skip_gate2_check:
                g = gate2.gate2_metrics(pred, truth, test)
                srec.update({"scorer_gate2_nrmse_inplane": g["nrmse_inplane"],
                             "scorer_gate2_nrmse_3d": g["nrmse_3d"],
                             "scorer_gate2_cumwss_rel_l2_inner_mean": g["cumwss_rel_l2_inner_mean"],
                             "pooled_minus_scorer": pooled - g["nrmse_inplane"]})
            if rein.exists():
                r = json.loads(rein.read_text())
                ref = srec.get("scorer_gate2_nrmse_inplane", pooled)
                srec.update({"reinfer_json": str(rein.relative_to(_REPO)),
                             "reinfer_nrmse_inplane": r["nrmse_inplane"], "reinfer_nrmse_3d": r["nrmse_3d"],
                             "reinfer_cumwss_rel_l2_inner_mean": r["cumwss_rel_l2_inner_mean"],
                             "delta_vs_reinfer_nrmse_inplane": ref - r["nrmse_inplane"],
                             "rel_delta_vs_reinfer_nrmse_inplane": (ref - r["nrmse_inplane"]) / r["nrmse_inplane"]})
            srec["summary"] = _summ(e_ip.mean(0), first_idx, tg)
            srec["wallclock_s"] = round(time.time() - t0, 1)
            mrec["seeds"][str(seed)] = srec
            print(f"[growth] {key} seed{seed}: pooled={pooled:.6f} scorer={srec.get('scorer_gate2_nrmse_inplane')} "
                  f"reinfer={srec.get('reinfer_nrmse_inplane')} first[{first_idx}]={srec['summary']['err_first_pred_step']:.6f} "
                  f"last={srec['summary']['err_last_step']:.6f} ratio={srec['summary']['ratio_last_over_first']} "
                  f"({srec['wallclock_s']}s)", flush=True)
            del pred, truth, e_ip, e_3d, tr_ip
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        cm = np.array([s["curve_mean"] for s in mrec["seeds"].values()])
        cmed = np.array([s["curve_median"] for s in mrec["seeds"].values()])
        mrec["seed_mean"] = {"n_seeds": int(cm.shape[0]), "curve_mean": cm.mean(0).tolist(),
                             "curve_mean_std_over_seeds": cm.std(0, ddof=0).tolist(),
                             "curve_median": cmed.mean(0).tolist(),
                             "curve_mean_3ch": np.array([s["curve_mean_3ch"] for s in mrec["seeds"].values()]).mean(0).tolist()}
        per_seed = [s["summary"] for s in mrec["seeds"].values()]
        mrec["summary"] = {"of_seed_mean_curve": _summ(cm.mean(0), first_idx, t_grid),
                           "seed_mean_of_per_seed": {k: float(np.mean([p[k] for p in per_seed]))
                                                     for k in ("err_first_pred_step", "err_last_step",
                                                               "ratio_last_over_first", "err_index0", "err_index1",
                                                               "ratio_last_over_index1", "err_max",
                                                               "mean_ramp_t_lt_1", "mean_steady_t_ge_1")},
                           "pooled_nrmse_inplane_seed_mean": float(np.mean([s["pooled_nrmse_inplane_from_curve"]
                                                                            for s in mrec["seeds"].values()])),
                           "reinfer_nrmse_inplane_seed_mean": (float(np.mean([s["reinfer_nrmse_inplane"]
                                                                              for s in mrec["seeds"].values()]))
                                                               if all("reinfer_nrmse_inplane" in s
                                                                      for s in mrec["seeds"].values()) else None)}
        out["models"][key] = mrec

    out["provenance"] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "host": socket.gethostname(), "python": platform.python_version(), "torch": torch.__version__,
        "cuda_device": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "data_npz": str(args.npz), "data_npz_bytes": args.npz.stat().st_size,
        "data_npz_mtime": datetime.fromtimestamp(args.npz.stat().st_mtime, timezone.utc).isoformat(timespec="seconds"),
        "encoder_npz": str(Path(args.encoder_root) / f"cmp3d_{SPLIT}_pod139_fix.npz"),
        "encoder_sha256": _sha256(Path(args.encoder_root) / f"cmp3d_{SPLIT}_pod139_fix.npz"),
        "runs_root": str(args.runs_root), "split_json": "benchmark/splits/cmp_interp.json",
        "code": {"repo_head": _git(_REPO, "rev-parse", "HEAD"),
                 "repo_dirty": bool(_git(_REPO, "status", "--porcelain", "--untracked-files=no")),
                 "scorer": "scripts/run_cmp3d_gate2.py",
                 "dmdc_driver": "scripts/cmp3d_dmdc_gate2.py"},
        "wallclock_s": round(time.time() - t_start, 1)}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n")
    print(f"[growth] wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
