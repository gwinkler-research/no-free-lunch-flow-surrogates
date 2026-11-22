#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ============================================================================
# Purpose: count what the Strouhal-error gate of the wake scorer actually
#   excludes. The paper's Strouhal error averages over case-probe pairs, and the
#   scorer (benchmark/metrics_kvs.py::_fft_freq_err) keeps a pair only when
#   the peak estimator returns a finite frequency above zero for the TRUE
#   trace. This script applies exactly that estimator and that condition to
#   the true probe traces of both splits and writes the counts, so the
#   sentence in Sec. Metrics that describes the gate is backed by a number
#   the values layer can quote.
# Pipeline role: backs the gate sentence of Sec. Metrics (Strouhal error)
#   through the macros \vKvsFftPairs* and \vKvsFftDropped* of
#   values/emit_values.py. Reads the truth files
#   results/benchmark/truth/kvs_<split>.npz (not included; directory given by
#   --truth-root) and the probe indices of benchmark/splits/kvs_eval_nodes.npz.
#   Writes results/benchmark/kvs_strouhal_gate_counts.json.
# Eval-consistency: same window (t >= 2 s), same dt, same fft_peak defaults
#   (Hann window, fourfold zero padding, parabolic refinement) and the same
#   acceptance test (finite and > 1e-9) as the scorer. It changes no metric;
#   it only reports how many pairs the existing gate keeps.
# ============================================================================
"""Count the case-probe pairs the Strouhal gate keeps and drops, per split.

  python3 -u scripts/strouhal_gate_count.py [--truth-root <dir with kvs_<split>.npz>]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from benchmark.metrics_kvs import STEADY_T_MIN            # noqa: E402
from tools.common.fft_analysis import fft_peak            # noqa: E402

OUT = _REPO / "results" / "benchmark" / "kvs_strouhal_gate_counts.json"
TRUTH_ROOT = _REPO / "results" / "benchmark" / "truth"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--truth-root", default=str(TRUTH_ROOT))
    args = ap.parse_args(argv)
    pli = np.load(_REPO / "benchmark" / "splits" / "kvs_eval_nodes.npz")["probe_local_idx"].astype(int)
    out = {"gate": "pair kept iff fft_peak(true trace) is finite and > 1e-9 (peak away from zero frequency)",
           "window_s": STEADY_T_MIN, "n_probes": int(len(pli)), "splits": {}}
    for split in ("interp", "extrap"):
        p = Path(args.truth_root) / f"kvs_{split}.npz"
        z = np.load(p, allow_pickle=False)
        v = np.asarray(z["u_eval"], np.float32)[:, :, pli, 1]
        t = np.asarray(z["t_grid"], float)
        m = t >= STEADY_T_MIN
        dt = float(np.median(np.diff(t)))
        ft = np.array([[fft_peak(v[c, m, q], dt) for q in range(v.shape[2])] for c in range(v.shape[0])])
        keep = np.isfinite(ft) & (ft > 1e-9)
        # Cases with U_max < 1 are counted separately: they back the Metrics
        # sentence that pairs at quiet probes of sub-critical cases stay in.
        umax = np.array([int(str(c).split("_")[1]) / 10.0 for c in z["case_ids"]])
        sub = umax < 1.0
        out["splits"][split] = {
            "truth_npz": str(p), "truth_sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            "n_cases": int(v.shape[0]), "n_pairs": int(ft.size),
            "n_kept": int(keep.sum()), "n_dropped": int(ft.size - keep.sum()),
            "n_cases_subcritical": int(sub.sum()), "n_pairs_subcritical": int(keep[sub].size),
            "n_kept_subcritical": int(keep[sub].sum()),
            "dropped_case_ids": sorted({str(c) for c in np.asarray(z["case_ids"])[np.where(~keep)[0]]}),
        }
        print(f"[fft-gate] {split}: {ft.size} pairs, kept {keep.sum()}, dropped {ft.size - keep.sum()}")
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(f"[fft-gate] wrote {OUT.relative_to(_REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
