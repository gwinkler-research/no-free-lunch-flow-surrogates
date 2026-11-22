# ============================================================================
# Purpose: score one saved prediction artifact against the truth reference.
#   evaluate() loads run_dir/predictions.npz, validates it against the frozen
#   SplitSpec, reorders the truth onto the artifact's case order, computes the
#   metrics and writes results/benchmark/<dataset>_<split>_<model>_<seed>.json
#   as {"metrics": ..., "meta": ...}. It trains nothing and runs no model.
# Pipeline role: benchmark/runner.py calls it at the end of every KVS run.
#   scripts/build_paper_matrix.py reads rollout_rmse, rmse_warped, power_ratio
#   and fft_freq_err for the KVS table from these per-seed JSONs. KVS only;
#   the CMP table is scored by scripts/run_cmp3d_gate2.py.
# Eval-consistency:
#   (1) The truth comes from one fixed file per (dataset, split),
#       results/benchmark/truth/<dataset>_<split>.npz, never from the run.
#   (2) _reorder_to() matches the truth rows to the artifact's case_ids via
#       str(case_id).
#   (3) Eval nodes and probe indices come from the frozen SplitSpec
#       (benchmark/splits/), not from the artifact, so a model cannot choose
#       where it is measured.
# ============================================================================
"""Evaluation harness: score a stored prediction artifact against the truth reference.

evaluate(run_dir) does the following:
  - load the prediction artifact (benchmark.prediction_io) and the matching SplitSpec
  - validate the artifact schema against the split
  - load the truth reference (results/benchmark/truth/<dataset>_<split>.npz, or truth_npz)
  - reorder the truth onto the artifact's case order
  - compute the metrics (benchmark.metrics_kvs)
  - write results/benchmark/<dataset>_<split>_<model>_<seed>.json ({"metrics": ..., "meta": ...})
  - return (metrics: dict, meta: dict)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from benchmark.metrics_kvs import compute_kvs_metrics
from benchmark.prediction_io import load_prediction, validate
from benchmark.splits import load_split

REPO = Path(__file__).resolve().parents[1]
_DEFAULT_TRUTH_DIR = REPO / "results" / "benchmark" / "truth"
_DEFAULT_OUT_DIR = REPO / "results" / "benchmark"


def _reorder_to(case_ids_target, case_ids_have):
    pos = {c: i for i, c in enumerate(map(str, case_ids_have))}
    return np.array([pos[str(c)] for c in case_ids_target], dtype=np.int64)


def evaluate(run_dir, *, truth_npz=None, splits_dir=None, out_dir=None, split_spec=None):
    """Score the artifact in run_dir and write the result JSON.

    truth_npz: truth file to use instead of results/benchmark/truth/<dataset>_<split>.npz.
    splits_dir: directory of the frozen split files (default benchmark/splits/).
    out_dir: output directory of the result JSON (default results/benchmark/).
    split_spec: optional prebuilt SplitSpec, for test sets without a frozen
    {dataset}_{kind}.json (load_split accepts only those).
    """
    run_dir = Path(run_dir)
    art = load_prediction(run_dir)
    spec = split_spec if split_spec is not None else \
        load_split(art.dataset, art.split, **({"splits_dir": splits_dir} if splits_dir is not None else {}))
    validate(art, spec)
    truth_path = Path(truth_npz) if truth_npz is not None else (_DEFAULT_TRUTH_DIR / f"{art.dataset}_{art.split}.npz")
    with np.load(truth_path, allow_pickle=False) as z:
        truth = {k: z[k] for k in z.files}
    perm = _reorder_to(art.case_ids, truth["case_ids"])

    if art.dataset == "kvs":
        metrics = compute_kvs_metrics(
            art.arrays["u_eval"], art.arrays["p_eval"], truth["u_eval"][perm], truth["p_eval"][perm],
            art.arrays["t_grid"], spec.probe_local_idx,
        )
    else:
        raise ValueError(f"unknown dataset {art.dataset!r}")

    out_dir = Path(out_dir) if out_dir is not None else _DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / f"{art.dataset}_{art.split}_{art.meta['model']}_{art.meta['seed']}.json"
    out_json.write_text(json.dumps({"metrics": metrics, "meta": art.meta}, indent=2))
    return metrics, art.meta
