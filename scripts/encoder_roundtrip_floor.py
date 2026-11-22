#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ============================================================================
# Purpose: measure the resolution of the frozen wake encoder, per split, with
#   the SAME scorer that grades every latent row. It loads a frozen ConvAE
#   file, pushes the test fields of one split through encode -> decode, writes
#   the result as an ordinary prediction artifact, and calls the eval harness
#   on it. The metrics of that artifact (rollout RMSE, phase-aligned RMSE,
#   shedding power P) are the reference level for every latent row on that split.
# Pipeline role: Methods, frozen encoder (instrument resolution), and Results,
#   wake interpolation (recurrent branches at the encoder floor). Produces
#   results/benchmark/kvs_<split>_convae_83x442_floor_0.json, which
#   values/emit_values.py turns into the \vKvsEncFloor* macros. Needs the
#   encoder file and the raw KVS data (--data-root), which are not in this
#   repository.
# Eval-consistency: no training, no fit-time state. The decode uses the
#   x_mean/x_std stored in the encoder file, exactly as latent rows do. The
#   floor of the interp split must reproduce the metrics block of the encoder
#   training artifact (kvs_interp_convae_83x442_clean_0.json) to within 1e-9;
#   that equality is the self-check of this script (--expect-json).
# ============================================================================
"""Encoder round-trip floor per split, through the standard eval harness.

  python3 -u scripts/encoder_roundtrip_floor.py --split interp \
      --encoder <abs path to kvs_interp_convae_83x442_bc32_47132e7b07.pt> \
      --expect-json results/benchmark/kvs_interp_convae_83x442_clean_0.json
  python3 -u scripts/encoder_roundtrip_floor.py --split extrap \
      --encoder <abs path to kvs_extrap_convae_83x442_bc32.pt>
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from benchmark.adapters import get_adapter                      # noqa: E402
from benchmark.adapters.latent_dynamics import _load_frozen_encoder  # noqa: E402
from benchmark.eval_harness import evaluate                     # noqa: E402
from benchmark.prediction_io import load_prediction, save_prediction, validate  # noqa: E402
from benchmark.runner import _make_views, config_hash, git_sha, load_config  # noqa: E402


def sha256(p: Path, chunk=1 << 24) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--split", required=True, choices=["interp", "extrap"])
    ap.add_argument("--encoder", required=True, help="frozen ConvAE file (.pt) to measure")
    ap.add_argument("--config", default=None,
                    help="grid_convae config for data/raster paths (default: the encoder config of the split)")
    ap.add_argument("--model-name", default="convae_83x442_floor")
    ap.add_argument("--data-root", default=None,
                    help="directory that holds the relative chunk_roots / mesh_npz / raster_cache of the config "
                         "(raw KVS data, not in this repository); default: this repository")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--expect-json", default=None,
                    help="existing harness JSON whose metrics block this run must reproduce (self-check)")
    a = ap.parse_args(argv)

    import torch
    dev = torch.device(a.device)
    default_cfg = {"interp": "configs/clean/paper_kvs_interp_convae_83x442_clean.yaml",
                   "extrap": "configs/paper_kvs_extrap_convae_83x442.yaml"}
    cfg_path = a.config or (_REPO / default_cfg[a.split])
    cfg = load_config(cfg_path)
    assert cfg["split"] == a.split, (cfg["split"], a.split)
    cfg["require_h5"] = False          # the raster cache carries the fields; no H5 needed for a round trip
    if a.data_root:
        root = Path(a.data_root).resolve()
        _abs = lambda p: str(p) if Path(p).is_absolute() else str(root / p)   # noqa: E731
        cfg["chunk_roots"] = [_abs(p) for p in cfg["chunk_roots"]]
        cfg["mesh_npz"] = _abs(cfg["mesh_npz"])
        cfg["raster_cache"] = _abs(cfg["raster_cache"])
    cfg["model_name"] = a.model_name
    cfg["model"]["device"] = str(dev)

    enc = Path(a.encoder).resolve()
    ck = torch.load(enc, map_location="cpu", weights_only=False)
    if ck.get("split") != a.split:
        raise SystemExit(f"encoder file declares split={ck.get('split')!r}, requested {a.split!r}")
    ae, _, _, _, _, x_mean, x_std = _load_frozen_encoder(str(enc), dev)

    views, split_spec = _make_views(cfg, 0)
    ms = {"device": dev, "model": ae,
          "x_mean": x_mean.cpu().numpy().reshape(3), "x_std": x_std.cpu().numpy().reshape(3),
          "encoder_ckpt_path": str(enc), "best_epoch": -1}
    adapter = get_adapter(cfg["adapter"])(cfg)

    t1 = time.time()
    arrays, meta = adapter.predict_eval(ms, views["test"], split_spec, cfg)
    n_test = len(views["test"].case_ids)
    meta = {"train_wallclock_s": None, "infer_latency_ms": (time.time() - t1) * 1000.0 / max(1, n_test),
            "git_sha": git_sha(), "config_hash": config_hash(cfg), "n_params": int(sum(p.numel() for p in ae.parameters())),
            "encoder_sha256": sha256(enc), "encoder_file": str(enc),
            "floor_of": "encode->decode round trip of the test fields through the frozen encoder",
            "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(), **(meta or {})}

    run_dir = _REPO / "results" / "benchmark" / "runs" / f"{a.model_name}_{a.split}_seed0"
    run_dir.mkdir(parents=True, exist_ok=True)
    save_prediction(run_dir, dataset=cfg["dataset"], split=a.split, model=a.model_name, seed=0,
                    target_type="field", arrays=arrays, meta=meta)
    validate(load_prediction(run_dir), split_spec)
    truth = (Path(a.data_root).resolve() / "results" / "benchmark" / "truth" / f"kvs_{a.split}.npz") if a.data_root else None
    evaluate(run_dir, truth_npz=truth)
    out_json = _REPO / "results" / "benchmark" / f"kvs_{a.split}_{a.model_name}_0.json"
    got = json.loads(out_json.read_text())["metrics"]
    print(f"[floor] {a.split}: " + ", ".join(f"{k}={got[k]:.4f}" for k in ("rollout_rmse", "rmse_warped", "power_ratio")),
          f"-> {out_json}", flush=True)

    if a.expect_json:
        exp = json.loads(Path(a.expect_json).read_text())["metrics"]
        bad = {k: (exp[k], got[k]) for k in ("rollout_rmse", "rmse_warped", "power_ratio") if abs(exp[k] - got[k]) > 1e-9}
        if bad:
            print(f"[floor] SELF-CHECK FAIL against {a.expect_json}: {bad}", flush=True)
            return 1
        print(f"[floor] self-check PASS: metrics reproduce {a.expect_json} to 1e-9", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
