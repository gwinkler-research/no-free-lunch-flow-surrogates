# Purpose: load the frozen train/val/test case IDs
#   (benchmark/splits/<dataset>_<kind>.json) and the eval-node index sets used
#   by the metrics (benchmark/splits/<dataset>_eval_nodes.npz).
# Pipeline role: called by benchmark/runner.py (KVS), benchmark/cmp3d_data.py
#   and scripts/cmp3d_dmdc_gate2.py (CMP) and benchmark/eval_harness.py. For
#   KVS the runner replaces the ratio-based split of KarmanH5Dataset
#   (tools.common.splitting.split_runs) with these frozen IDs.
# Leakage guard: load_split() raises ValueError if a split part is empty or if
#   two parts share a case ID. This keeps test cases out of train and val in
#   every pipeline that loads its split here.
# No data and no normalisation statistics are touched in this module.
"""Load and validate the frozen benchmark splits and eval-node sets.

Committed files under benchmark/splits/:
  {dataset}_{kind}.json        {"train": [case_id...], "val": [...], "test": [...], ...}
  {dataset}_eval_nodes.npz     KVS: eval_idx_p2, eval_idx_p1, probe_node_idx, probe_local_idx,
                                    probe_xy, roi_weight_p2
                               CMP: eval_idx_fem, wafer_radius, probe_node_idx, probe_local_idx, probe_xy
dataset in {"kvs","cmp"}, kind in {"interp","extrap"}.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_DATASETS = ("kvs", "cmp")
_KINDS = ("interp", "extrap")
_DEFAULT_SPLITS_DIR = Path(__file__).resolve().parent / "splits"


@dataclass(frozen=True)
class SplitSpec:
    dataset: str
    kind: str
    train_ids: list[str]
    val_ids: list[str]
    test_ids: list[str]
    eval_idx_p2: np.ndarray | None       # KVS only
    eval_idx_p1: np.ndarray | None       # KVS only
    eval_idx_fem: np.ndarray | None      # CMP only
    probe_node_idx: np.ndarray           # global node indices of the probe points
    probe_local_idx: np.ndarray          # positions of the probe nodes in the matching eval array
    probe_xy: np.ndarray                 # (P, 2) target coordinates of the probe points
    extra: dict                          # e.g. {"wafer_radius": 0.149, "roi_weight_p2": ...}


def load_split(dataset: str, kind: str, *, splits_dir: Path | str = _DEFAULT_SPLITS_DIR) -> SplitSpec:
    if dataset not in _DATASETS:
        raise ValueError(f"dataset must be in {_DATASETS}, got {dataset!r}")
    if kind not in _KINDS:
        raise ValueError(f"kind must be in {_KINDS}, got {kind!r}")
    splits_dir = Path(splits_dir)
    js = json.loads((splits_dir / f"{dataset}_{kind}.json").read_text())
    train_ids, val_ids, test_ids = list(js["train"]), list(js["val"]), list(js["test"])
    s_tr, s_va, s_te = set(train_ids), set(val_ids), set(test_ids)
    if not (s_tr and s_va and s_te):
        raise ValueError(f"empty split part in {dataset}_{kind}.json")
    # leakage guard: fail if two split parts share a case ID
    if not (s_tr.isdisjoint(s_va) and s_tr.isdisjoint(s_te) and s_va.isdisjoint(s_te)):
        raise ValueError(f"split parts in {dataset}_{kind}.json are not disjoint")

    with np.load(splits_dir / f"{dataset}_eval_nodes.npz") as z:
        d = {k: z[k] for k in z.files}
    probe_node_idx = np.asarray(d.get("probe_node_idx", np.zeros(0, np.int32)))
    probe_local_idx = np.asarray(d.get("probe_local_idx", np.zeros(0, np.int32)))
    probe_xy = np.asarray(d.get("probe_xy", np.zeros((0, 2), np.float32)))
    if dataset == "kvs":
        eval_idx_p2 = np.asarray(d["eval_idx_p2"]).astype(np.int64)
        eval_idx_p1 = np.asarray(d["eval_idx_p1"]).astype(np.int64)
        eval_idx_fem = None
        extra = {"roi_weight_p2": np.asarray(d.get("roi_weight_p2"))}
        if np.unique(eval_idx_p2).size != eval_idx_p2.size:
            raise ValueError("eval_idx_p2 contains duplicate indices")
    else:
        eval_idx_p2 = eval_idx_p1 = None
        eval_idx_fem = np.asarray(d["eval_idx_fem"]).astype(np.int64)
        extra = {"wafer_radius": float(d.get("wafer_radius", 0.149))}
        if np.unique(eval_idx_fem).size != eval_idx_fem.size:
            raise ValueError("eval_idx_fem contains duplicate indices")
    return SplitSpec(
        dataset=dataset, kind=kind, train_ids=train_ids, val_ids=val_ids, test_ids=test_ids,
        eval_idx_p2=eval_idx_p2, eval_idx_p1=eval_idx_p1, eval_idx_fem=eval_idx_fem,
        probe_node_idx=probe_node_idx.astype(np.int64), probe_local_idx=probe_local_idx.astype(np.int64),
        probe_xy=probe_xy.astype(np.float64), extra=extra,
    )
