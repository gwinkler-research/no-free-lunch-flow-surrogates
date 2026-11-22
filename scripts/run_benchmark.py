#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Purpose: command-line entry point for KVS runs. Calls
#   benchmark.runner.run_benchmark (train, predict, save, evaluate) for one
#   config and seed.
# Pipeline role: KVS counterpart of scripts/run_cmp3d_gate2.py. The runner
#   accepts only dataset "kvs"; CMP configs go through run_cmp3d_gate2.py.
"""Benchmark runner CLI for KVS configs.

  python3 scripts/run_benchmark.py --config configs/paper_kvs_extrap_convae_83x442.yaml --seed 0
  python3 scripts/run_benchmark.py --config configs/paper_kvs_extrap_armlp.yaml --seed 0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# repository root first on sys.path, so benchmark/ imports when the script runs directly
from benchmark.runner import load_config, run_benchmark  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-evaluate", action="store_true", help="skip the eval_harness call at the end")
    ap.add_argument("--out-dir", default=None)
    return ap


def main() -> int:
    a = build_parser().parse_args()
    cfg = load_config(a.config)
    run_benchmark(a.config, seed=a.seed, evaluate=not a.no_evaluate, out_dir=a.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
