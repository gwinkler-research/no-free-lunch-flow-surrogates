#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ============================================================================
# Purpose: put a number on the wake's intrinsic time scale for the a priori
#   rule of the Discussion. It reads the dominant shedding frequency of every
#   true test trajectory with the SAME peak estimator the Strouhal error
#   uses, takes the median over the probes that
#   actually oscillate, and reports the shedding period range over the cases
#   above the critical Reynolds number, together with the critical U_max that
#   follows from the paper's Re definition (Re = (2/3) U_max D / nu, Re_c = 47).
# Pipeline role: backs Sec. Datasets (shedding period against the ramp) and
#   the practical rule of the Discussion through the macros
#   \vKvsShedPeriodMin/Max and \vKvsUmaxCrit of values/emit_values.py. Reads
#   the truth files results/benchmark/truth/kvs_<split>.npz (not included;
#   directory given by --truth-root) and the probe indices of
#   benchmark/splits/kvs_eval_nodes.npz. Writes
#   results/benchmark/kvs_truth_shedding_period.json.
# Eval-consistency: same window (t >= 2 s), same dt and the same fft_peak as
#   metrics_kvs._fft_freq_err. A probe counts as oscillating when its true
#   trace has a standard deviation above 0.01 in the window; a case enters
#   the range when at least three probes oscillate and its Reynolds number
#   exceeds Re_c. The coverage is the union of the two test sets (U_max 0.5
#   to 5.0 in steps of 0.1). No metric is touched.
# ============================================================================
"""Shedding period range of the true wake trajectories, per case, above Re_c.

  python3 -u scripts/kvs_shedding_period_range.py [--truth-root <dir>]
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

OUT = _REPO / "results" / "benchmark" / "kvs_truth_shedding_period.json"
TRUTH_ROOT = _REPO / "results" / "benchmark" / "truth"
NU, D, RE_C = 1e-3, 0.1, 47.0            # paper: nu, cylinder diameter, critical Re (Sec. Datasets)
SIGMA_MIN, MIN_PROBES = 0.01, 3


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--truth-root", default=str(TRUTH_ROOT))
    args = ap.parse_args(argv)
    pli = np.load(_REPO / "benchmark" / "splits" / "kvs_eval_nodes.npz")["probe_local_idx"].astype(int)
    umax_crit = RE_C * NU / (D * 2.0 / 3.0)
    cases, sources = [], {}
    for split in ("interp", "extrap"):
        p = Path(args.truth_root) / f"kvs_{split}.npz"
        z = np.load(p, allow_pickle=False)
        sources[split] = {"truth_npz": str(p), "truth_sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
        v = np.asarray(z["u_eval"], np.float32)[:, :, pli, 1]
        t = np.asarray(z["t_grid"], float)
        m = t >= STEADY_T_MIN
        dt = float(np.median(np.diff(t)))
        for c, cid in enumerate(z["case_ids"]):
            cid = str(cid)
            umax = int(cid.split("_")[1]) / 10.0
            ok = np.where(v[c, m].std(0) > SIGMA_MIN)[0]
            if len(ok) < MIN_PROBES:
                continue
            f = float(np.median([fft_peak(v[c, m, q], dt) for q in ok]))
            cases.append({"split": split, "case_id": cid, "umax": umax,
                          "re": (2.0 / 3.0) * umax * D / NU, "f_hz": f, "period_s": 1.0 / f})
    above = [r for r in cases if r["re"] > RE_C]
    per = np.array([r["period_s"] for r in above])
    lo, hi = above[int(np.argmin(per))], above[int(np.argmax(per))]
    out = {"definitions": {"window_s": STEADY_T_MIN, "sigma_min": SIGMA_MIN, "min_probes": MIN_PROBES,
                           "nu": NU, "D": D, "re_c": RE_C, "estimator": "fft_peak (Hann, 4x padding, parabolic)"},
           "umax_crit": umax_crit, "n_cases_oscillating": len(cases), "n_cases_above_re_c": len(above),
           "period_min_s": lo["period_s"], "period_min_case": lo["case_id"],
           "period_max_s": hi["period_s"], "period_max_case": hi["case_id"],
           "sources": sources, "cases": cases}
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(f"[shed] U_max,c = {umax_crit:.3f}; {len(above)} cases above Re_c; period "
          f"{lo['period_s']:.3f} s ({lo['case_id']}) to {hi['period_s']:.3f} s ({hi['case_id']})")
    print(f"[shed] wrote {OUT.relative_to(_REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
