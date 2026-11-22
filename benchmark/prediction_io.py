# ============================================================================
# Purpose: on-disk schema of a prediction artifact (predictions.npz +
#   prediction_meta.json), its I/O (save_prediction, load_prediction) and the
#   schema check validate() against a SplitSpec.
# Pipeline role: benchmark/runner.py writes the test predictions of every KVS
#   run through save_prediction(); eval_harness.evaluate() reads them back
#   through load_prediction() and validate().
# Eval-consistency: REQUIRED_ARRAYS and validate() enforce the
#   (n_test, 201, n_eval, C) shape per (dataset, target_type) and reject
#   non-finite values, so a transposed or mis-shaped array fails here and not
#   later in the metrics. case_ids are checked as a set; eval_harness reorders
#   the truth onto the stored order.
# ============================================================================
"""Prediction-artifact schema and I/O for the evaluation harness.

A model run writes into its run directory:
  predictions.npz        the arrays (see REQUIRED_ARRAYS) plus 'target_type'
  prediction_meta.json   {dataset, split, model, seed, target_type, n_params, train_wallclock_s,
                          infer_latency_ms, config_hash, git_sha, created_utc, ...}

target_type:
  "kvs" / "field"            -> u_eval (n_test,201,n_eval_p2,2), p_eval (n_test,201,n_eval_p1),
                                case_ids (n_test,), t_grid (201,)  [+ optional extra arrays]
"""
from __future__ import annotations

import datetime as _dt
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Allowed (dataset, target_type) pairs and their required arrays. save_prediction()
# checks that the arrays are present; validate() checks their shapes against a SplitSpec.
_VALID = {("kvs", "field")}
REQUIRED_ARRAYS = {
    ("kvs", "field"): ("u_eval", "p_eval", "case_ids", "t_grid"),
}


@dataclass(frozen=True)
class PredictionArtifact:
    dataset: str
    split: str
    target_type: str
    arrays: dict
    meta: dict

    @property
    def case_ids(self) -> list[str]:
        return [str(s) for s in self.arrays["case_ids"]]


def save_prediction(out_dir, *, dataset, split, model, seed, target_type, arrays, meta):
    if (dataset, target_type) not in _VALID:
        raise ValueError(f"unknown (dataset, target_type) combination: ({dataset!r}, {target_type!r})")
    req = REQUIRED_ARRAYS[(dataset, target_type)]
    missing = [k for k in req if k not in arrays]
    if missing:
        raise ValueError(f"missing required arrays {missing} for ({dataset},{target_type})")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_payload = {k: np.asarray(v) for k, v in arrays.items()}
    npz_payload["target_type"] = np.array(target_type)
    tmp = out_dir / "predictions.npz.tmp"
    with open(tmp, "wb") as fh:
        np.savez(fh, **npz_payload)
    tmp.replace(out_dir / "predictions.npz")
    meta_full = {
        "dataset": dataset, "split": split, "model": model,
        "seed": (int(seed) if (isinstance(seed, int) or (isinstance(seed, str) and seed.lstrip("-").isdigit())) else str(seed)),
        "target_type": target_type,
        "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    }
    meta_full.update(meta or {})
    (out_dir / "prediction_meta.json").write_text(json.dumps(meta_full, indent=2))
    return out_dir / "predictions.npz"


def load_prediction(run_dir) -> PredictionArtifact:
    run_dir = Path(run_dir)
    with np.load(run_dir / "predictions.npz", allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files if k != "target_type"}
        target_type = str(z["target_type"])
    meta = json.loads((run_dir / "prediction_meta.json").read_text())
    return PredictionArtifact(dataset=meta["dataset"], split=meta["split"], target_type=target_type, arrays=arrays, meta=meta)


def validate(artifact: PredictionArtifact, split_spec) -> None:
    ds, tt = artifact.dataset, artifact.target_type
    if (ds, tt) not in _VALID:
        raise ValueError(f"unknown target_type {(ds, tt)}")
    if ds != split_spec.dataset:
        raise ValueError(f"dataset mismatch: artifact {ds!r} vs split {split_spec.dataset!r}")
    a = artifact.arrays
    # Set comparison only; eval_harness reorders the truth onto the artifact order.
    if set(artifact.case_ids) != set(split_spec.test_ids):
        raise ValueError(f"case_ids != test split (as a set): {len(set(artifact.case_ids))} vs {len(set(split_spec.test_ids))}")
    n_test = len(artifact.case_ids)
    for arr in a.values():
        if np.issubdtype(np.asarray(arr).dtype, np.floating) and not np.all(np.isfinite(arr)):
            raise ValueError("non-finite values (NaN/Inf) in the artifact")
    if ds == "kvs" and tt == "field":
        n_p2, n_p1 = split_spec.eval_idx_p2.size, split_spec.eval_idx_p1.size
        if a["u_eval"].shape != (n_test, 201, n_p2, 2):
            raise ValueError(f"u_eval shape {a['u_eval'].shape} != {(n_test, 201, n_p2, 2)} (eval_idx_p2={n_p2})")
        if a["p_eval"].shape != (n_test, 201, n_p1):
            raise ValueError(f"p_eval shape {a['p_eval'].shape} != {(n_test, 201, n_p1)} (eval_idx_p1={n_p1})")
        if np.asarray(a["t_grid"]).shape != (201,):
            raise ValueError(f"t_grid shape {np.asarray(a['t_grid']).shape} != (201,)")
