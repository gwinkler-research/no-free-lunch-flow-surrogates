#!/usr/bin/env python3
# Purpose: assembles the two result matrices of the paper (KVS 7 metrics, CMP 9
#   metric columns) from the model registry in benchmark/paper_models.py, one
#   row per model x {interp, extrap}, values = mean +/- std over the seeds.
# Inputs: CMP per-seed JSONs from reports/_reinfer_fix/ plus the DMDc JSONs
#   reports/cmp3d_gate2_cmp3d_dmdc_k8[_extrap]_fix_seed1.json. KVS per-seed
#   JSONs results/benchmark/kvs_<split>_<core_key>_<seed>.json, written by
#   benchmark/eval_harness.evaluate(), the oscMed10 onset NPZ
#   (scripts/onset_oscmed_apply.py) and the diagnostic JSON
#   (scripts/kvs_paper_diagnose.py) in reports/.
# Output: reports/PAPER_MATRIX_CLEAN/kvs_matrix.{csv,json} and
#   reports/PAPER_MATRIX_FIX/cmp_matrix.{csv,json}.
# Data versions: the CMP chain "fix" is the CMP data set and scoring used in
#   the paper (file names ending in _fix). The KVS registry "clean" is the
#   model list of the paper: interp rows on the interp encoder (names ending
#   in _clean), extrap rows as listed in benchmark/paper_models.py. The KVS
#   output directory follows from the registry suffix and the CMP output
#   directory from the chain, so one run builds both matrices.
# Convention: _kvs_core() stops with SystemExit if the number of seed JSONs
#   found for a core_key differs from the n declared in the registry. A
#   surplus or missing file would otherwise change the mean without warning.
#   The onset column comes from the table KVS_ONSET_SOURCE in
#   benchmark/paper_models.py (oscMed10 detector).
"""Build the two result matrices of the paper from the registry (benchmark/paper_models.py).

KVS (7 metrics) and CMP (9 metric columns), rows = registry models x {interp, extrap},
values = mean +- std over the seeds that exist.

CMP: reports/_reinfer_fix/cmp3d_6metric_<split>_<key>_fix_seed<n>.json
     DMDc: reports/cmp3d_gate2_cmp3d_dmdc_k8[_extrap]_fix_seed1.json
     -> reports/PAPER_MATRIX_FIX/   (9 columns, including the in-plane split)

KVS: registry 'clean' (the default of KVS_PAPER_REGISTRY) -> reports/PAPER_MATRIX_CLEAN/
  KVS core-4 : results/benchmark/kvs_<split>_<core_key>_<seed>.json  (run_benchmark)
  KVS onset  : reports/<KVS_ONSET_SOURCE[sfx].stem>[_extrap].npz  (mean_abs_dt per run dir)
  KVS strip+H+alpha: reports/kvs_diag_paper<sfx>_<split>.json  (scripts/kvs_paper_diagnose.py)

A missing source gives an empty cell (null), not a crash; the report lists the gaps.
The KVS core JSONs are the exception: a seed count that differs from the registry stops
the build.

  python3 scripts/build_paper_matrix.py --only cmp
  python3 scripts/build_paper_matrix.py --only kvs
"""
from __future__ import annotations
import argparse
import csv
import json
import re
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
import numpy as np
from benchmark.paper_models import (CMP_2B, KVS_CORE, KVS_ONSET_SOURCE, all_core_keys,  # noqa: F401
                                    active_kvs_2a, cmp_metrics, kvs_core_key,
                                    kvs_metrics, kvs_rundirs, registry_suffix)

REPORTS = _REPO / "reports"
RUNS_JSON = _REPO / "results" / "benchmark"

_SFX = registry_suffix()          # '_clean', the KVS registry of the paper

# CMP chain: source directory, file name suffix, output directory. 'fix' is the CMP data
# set and scoring used in the paper.
CMP_CHAINS = {
    "fix":    dict(reinfer_dir="_reinfer_fix", suffix="_fix", out="PAPER_MATRIX_FIX"),
}


def _agg(vals):
    """mean/std/n over the available (not None, finite) seed values; std with ddof = 0."""
    v = [float(x) for x in vals if x is not None and np.isfinite(x)]
    if not v:
        return dict(mean=None, std=None, n=0)
    return dict(mean=float(np.mean(v)), std=float(np.std(v)), n=len(v))


# ----------------------------------------------------------------------------- CMP
# Source path from the paper_models.py registry entry: the re-inference JSON stem for
# models scored by re-inference, the DMDc gate-2 JSON for the linear baseline. The
# chain adds the source directory and the name suffix.
def cmp_seed_json(split, m, seed, chain):
    c = CMP_CHAINS[chain]
    if m.get("dmdc"):
        suff = "_extrap" if split == "extrap" else ""
        return REPORTS / f"cmp3d_gate2_cmp3d_dmdc_k8{suff}{c['suffix']}_seed{seed}.json"
    return (REPORTS / c["reinfer_dir"]
            / f"cmp3d_6metric_{split}_{m['reinfer_key']}{c['suffix']}_seed{seed}.json")


def build_cmp(chain):
    metrics = cmp_metrics(chain)
    rows, gaps = [], []
    for m in CMP_2B:
        for split in ("interp", "extrap"):
            per_metric = {k: [] for k in metrics}
            for s in m["seeds"]:
                f = cmp_seed_json(split, m, s, chain)
                if not f.exists():
                    gaps.append(f"CMP {m['key']} {split} seed{s}: {f.name}")
                    continue
                d = json.loads(f.read_text())
                for k in metrics:
                    per_metric[k].append(d.get(k))
            row = dict(model=m["key"], label=m["label"], cell=m["cell"], role=m["role"],
                       split=split, expected_n=m["n"])
            for k in metrics:
                row[k] = _agg(per_metric[k])
            rows.append(row)
    return rows, gaps


# ----------------------------------------------------------------------------- KVS
def _kvs_core(split, core_key, expected_n):
    """Glob all core JSONs (seed sets differ per model) -> dict metric -> [seed values].

    Stops if n != expected_n: a surplus file (an extra seed, an ensemble JSON) would
    otherwise enter the mean silently, and a missing one would give a wrong mean silently."""
    # The glob alone also matches a LONGER core_key that starts with this one (e.g.
    # kvs_extrap_ar_mlp_0.json and kvs_extrap_ar_mlp_clean_0.json in the same directory),
    # which would trip the hard-fail below with a misleading "6 found, expected 3".
    # Drop exactly the files that a strictly longer registry core_key claims as its own
    # <key>_<seed>.json. Everything else the glob finds is kept, so the irregular stem
    # kvs_extrap_kvs_field_deeponet_norm_extrap_0.json still counts for its row.
    longer = [k for k in all_core_keys() if len(k) > len(core_key) and k.startswith(core_key)]
    claimed = {re.compile(rf"^kvs_{re.escape(split)}_{re.escape(k)}_\d+\.json$") for k in longer}
    files = sorted(f for f in RUNS_JSON.glob(f"kvs_{split}_{core_key}_*.json")
                   if not any(p.match(f.name) for p in claimed))
    # Hard-fail, not a warning: an extra seed file or a missing one would otherwise
    # be averaged in or out silently and change the mean.
    if len(files) != expected_n:
        raise SystemExit(
            f"[matrix] KVS core {core_key} {split}: {len(files)} JSONs found, "
            f"expected {expected_n} (registry): {[f.name for f in files]}")
    out = {"rollout_rmse": [], "power_ratio": [], "rmse_warped": [], "fft_freq_err": []}
    for f in files:
        try:
            mt = json.loads(f.read_text()).get("metrics", {})
        except Exception:
            continue
        out["rollout_rmse"].append(mt.get("rollout_rmse"))
        out["power_ratio"].append(mt.get("power_ratio"))
        out["rmse_warped"].append(mt.get("rmse_warped"))
        out["fft_freq_err"].append(mt.get("fft_freq_err", mt.get("fft_freq_err_mean")))
    return out


def _onset_source():
    """(file stem, column name) of the onset detector of the active registry."""
    src = KVS_ONSET_SOURCE.get(_SFX)
    if src is None:
        raise SystemExit(f"[matrix] no onset source for registry suffix {_SFX!r}")
    return src["stem"], src["column"]


def _onset_map(split):
    """Run dir -> mean_abs_dt from the onset NPZ of the active registry."""
    stem, _ = _onset_source()
    f = REPORTS / (f"{stem}.npz" if split == "interp" else f"{stem}_extrap.npz")
    if not f.exists():
        return None
    z = np.load(f, allow_pickle=False)
    return {str(n): float(d) for n, d in zip(z["model_names"], z["mean_abs_dt"])}


def _diag_map(path):
    """JSON {run_dir: {metric: val}} or None."""
    f = REPORTS / path
    if not f.exists():
        return None
    return json.loads(f.read_text())


def build_kvs():
    metrics = kvs_metrics(_SFX)
    onset_stem, onset_col = _onset_source()
    rows, gaps = [], []
    onset = {sp: _onset_map(sp) for sp in ("interp", "extrap")}
    diag = {sp: _diag_map(f"kvs_diag_paper{_SFX}_{sp}.json") for sp in ("interp", "extrap")}
    for m in active_kvs_2a():
        for split in ("interp", "extrap"):
            core_key = kvs_core_key(m, split)
            core = _kvs_core(split, core_key, m["n"])
            rundirs = kvs_rundirs(m, split)
            row = dict(model=m["key"], label=m["label"], cell=m["cell"], role=m["role"],
                       split=split, expected_n=m["n"])
            # core-4
            for k in ("rollout_rmse", "power_ratio", "rmse_warped", "fft_freq_err"):
                row[k] = _agg(core[k])
                if row[k]["n"] == 0:
                    gaps.append(f"KVS {m['key']} {split}: core {k} missing (kvs_{split}_{core_key}_*.json)")
            # onset Δt* (per run-dir -> mean over seeds)
            om = onset[split]
            ov = [om.get(rd) for rd in rundirs] if om else []
            row[onset_col] = _agg(ov)
            if not om:
                gaps.append(f"KVS onset {split}: {onset_stem} npz missing")
            # strip + H + alpha_rms from kvs_paper_diagnose (kvs_diag_paper<sfx>_<split>.json)
            dg = diag[split]
            row["strip_k_rmse"] = _agg([dg.get(rd, {}).get("strip_k_rmse") for rd in rundirs] if dg else [])
            row["halluc_pixel_frac_pre"] = _agg([dg.get(rd, {}).get("halluc_pixel_frac_pre") for rd in rundirs] if dg else [])
            row["alpha_rms"] = _agg([dg.get(rd, {}).get("alpha_rms") for rd in rundirs] if dg else [])
            if not dg:
                gaps.append(f"KVS diag {split}: reports/kvs_diag_paper{_SFX}_{split}.json missing (run scripts/kvs_paper_diagnose.py)")
            rows.append(row)
    return rows, gaps


# ----------------------------------------------------------------------------- IO
def write_matrix(name, rows, metrics, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{name}.json").write_text(json.dumps(rows, indent=2))
    with (out_dir / f"{name}.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        head = (["model", "label", "cell", "split"] + list(metrics)
                + [f"{k}_std" for k in metrics] + [f"{k}_n" for k in metrics])
        w.writerow(head)
        for r in rows:
            mean = [("" if r[k]["mean"] is None else f"{r[k]['mean']:.4f}") for k in metrics]
            std = [("" if r[k]["std"] is None else f"{r[k]['std']:.4f}") for k in metrics]
            ns = [str(r[k]["n"]) for k in metrics]   # n per metric, not one min_n per row
            w.writerow([r["model"], r["label"], r["cell"], r["split"]] + mean + std + ns)
    empty = [(r["model"], r["split"], k) for r in rows for k in metrics if r[k]["n"] == 0]
    deficits = [(r["model"], r["split"], k, r[k]["n"], r.get("expected_n"))
                for r in rows for k in metrics
                if 0 < r[k]["n"] < (r.get("expected_n") or r[k]["n"])]
    if empty:
        print(f"  WARNING {name}: {len(empty)} EMPTY cells (n=0, no data):")
        for mod, sp, k in empty[:15]:
            print(f"     {mod} {sp} {k}: EMPTY")
    if deficits:
        print(f"  WARNING {name}: n deficit (found < declared) in {len(deficits)} cells:")
        for mod, sp, k, n, en in deficits[:15]:
            print(f"     {mod} {sp} {k}: n={n} (expected {en})")
    print(f"[saved] {out_dir / (name + '.json')}  +  {name}.csv  ({len(rows)} rows)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["kvs", "cmp"], default=None)
    ap.add_argument("--cmp-chain", choices=sorted(CMP_CHAINS), default="fix")
    ap.add_argument("--out-dir", default=None,
                    help="output directory for both matrices (overrides the defaults)")
    args = ap.parse_args()

    # The KVS output directory follows from the registry suffix and the CMP output
    # directory from the chain. --out-dir sends both matrices to one directory.
    override = Path(args.out_dir) if args.out_dir else None
    cmp_dir = override or REPORTS / CMP_CHAINS[args.cmp_chain]["out"]
    kvs_dir = override or REPORTS / ("PAPER_MATRIX" + _SFX.upper())
    onset_stem, onset_col = _onset_source()
    print(f"[matrix] CMP chain: {args.cmp_chain}  ->  {cmp_dir}")
    print(f"[matrix] KVS registry: {_SFX or '(default)'}  ->  {kvs_dir}  "
          f"(onset: {onset_stem} / {onset_col})")

    allgaps = []
    if args.only in (None, "cmp"):
        rows, gaps = build_cmp(args.cmp_chain)
        write_matrix("cmp_matrix", rows, cmp_metrics(args.cmp_chain), cmp_dir)
        allgaps += gaps
    if args.only in (None, "kvs"):
        rows, gaps = build_kvs()
        write_matrix("kvs_matrix", rows, kvs_metrics(_SFX), kvs_dir)
        allgaps += gaps
    if allgaps:
        print(f"\n[GAPS] {len(allgaps)}:")
        for g in allgaps:
            print(f"  - {g}")
    else:
        print("\n[OK] no gaps, every cell is filled.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
