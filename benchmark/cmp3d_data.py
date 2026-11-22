# ============================================================================
# Purpose: data view for the CMP-3D runs. Reads the normalised states
#   (states_norm) and the per-case arrays of a CMP training NPZ and selects the
#   cases of one part (train, val or test) of a frozen split.
# Pipeline role: make_cmp3d_views() is called by scripts/run_cmp3d_gate2.py;
#   scripts/cmp3d_dmdc_gate2.py builds Cmp3dDatasetView directly. KVS runs go
#   through benchmark/runner.py instead.
# Eval-consistency: the case lists come from load_split("cmp", ...) in
#   benchmark/splits.py, which checks that train, val and test are disjoint.
#   stats_mean and stats_std are stored in the NPZ and only read here.
# ============================================================================
"""CMP-3D dataset view over a training NPZ (e.g. data/cmp/cmp3d_full100fix_fem_training.npz).

The node set is the extruded 3D gap mesh (node_xyz) with 3 velocity channels (u_x, u_y, u_z).
wss_bot_pos and wss_top_pos index the two full z-planes inside the node set that the
cum-WSS post-processing uses.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from benchmark.splits import load_split

REPO = Path(__file__).resolve().parent.parent


class Cmp3dDatasetView:
    def __init__(self, split_name: str, npz, split_spec):
        self.split_name = split_name
        self.split_spec = split_spec
        self._z = np.load(Path(npz), mmap_mode="r", allow_pickle=True)
        names = [str(s) for s in self._z["case_names"]]
        pos = {c: i for i, c in enumerate(names)}
        wanted = {"train": split_spec.train_ids, "val": split_spec.val_ids,
                  "test": split_spec.test_ids}[split_name]
        missing = [c for c in wanted if c not in pos]
        if missing:
            raise KeyError(f"CMP-3D {split_name} IDs missing from case_names: {missing[:5]}")
        self.case_ids = list(wanted)
        self._idx = np.array([pos[c] for c in self.case_ids], dtype=np.int64)
        self._pos = {c: k for k, c in enumerate(self.case_ids)}
        self._theta = np.asarray(self._z["theta"], np.float32)[self._idx]
        self.info = json.loads(str(self._z["info"]))
        dt = float(self.info.get("dt", 0.01))
        self._t_grid = (np.arange(self._z["states_norm"].shape[1], dtype=np.float32) * dt)
        self.node_xyz = np.asarray(self._z["node_xyz"], np.float32)
        self.z_level_id = np.asarray(self._z["z_level_id"])
        self.in_plane_id = np.asarray(self._z["in_plane_id"])
        self.wss_bot_pos = np.asarray(self._z["wss_bot_pos"], np.int64)
        self.wss_top_pos = np.asarray(self._z["wss_top_pos"], np.int64)
        self.stats_mean = np.asarray(self._z["stats_mean"], np.float32)
        self.stats_std = np.asarray(self._z["stats_std"], np.float32)
        self.gap_z = float(self.info["z_levels"][-1])

    @property
    def theta(self) -> np.ndarray:
        return self._theta

    @property
    def t_grid(self) -> np.ndarray:
        return self._t_grid

    def _rows(self, case_ids):
        return self._idx[[self._pos[c] for c in case_ids]]

    def states(self, case_ids) -> np.ndarray:
        return np.asarray(self._z["states_norm"][self._rows(case_ids)], dtype=np.float32)

    def controls(self, case_ids) -> np.ndarray:
        return np.asarray(self._z["controls"][self._rows(case_ids)], dtype=np.float32)

    def wafer_centers(self, case_ids) -> np.ndarray:
        return np.asarray(self._z["wafer_centers"][self._rows(case_ids)], dtype=np.float32)

    def wafer_mask(self, case_ids) -> np.ndarray:
        return np.asarray(self._z["wafer_mask"][self._rows(case_ids)], dtype=bool)


def make_cmp3d_views(cfg: dict):
    npz = cfg["cmp3d_npz"]
    npz = (REPO / npz) if not Path(npz).is_absolute() else Path(npz)
    sd = cfg.get("splits_dir")
    # CMP-3D uses the frozen CMP splits (benchmark/splits/cmp_<kind>.json, same case
    # names); load_split() checks that the three parts are disjoint.
    split_spec = (load_split("cmp", cfg["split"], splits_dir=sd)
                  if sd else load_split("cmp", cfg["split"]))
    views = {s: Cmp3dDatasetView(s, npz, split_spec) for s in ("train", "val", "test")}
    return views, split_spec
