# ============================================================================
# Purpose: the run protocol of every KVS row. run_benchmark(config_path,
#   seed=...) seeds the process, builds the train/val/test DatasetViews from
#   the frozen split, instantiates the adapter named by cfg["adapter"] and runs
#   build -> fit -> predict_eval -> save_prediction -> evaluate. It records
#   train_wallclock_s, infer_latency_ms, git_sha, config_hash and n_params.
#   Model hyperparameters live only in the config YAML, which is stored in
#   run_meta.json.
# Pipeline role: writes the run directory (predictions.npz, run_meta.json,
#   loss_history.json, best/last checkpoint) and passes it to
#   benchmark.eval_harness.evaluate(), whose per-seed JSON is read by
#   scripts/build_paper_matrix.py for the KVS table. CLI wrapper:
#   scripts/run_benchmark.py. Only dataset "kvs" is accepted; CMP-3D runs go
#   through scripts/run_cmp3d_gate2.py and benchmark.cmp3d_data.
# Eval-consistency:
#   (1) Split: KarmanH5Dataset derives its own ratio-based split;
#       _make_views() replaces h5ds._sel and h5ds.case_ids with the frozen IDs
#       from benchmark/splits/kvs_<kind>.json, so every model and every seed
#       sees the same partition and the case order of the truth artifact.
#   (2) Seeding: set_seed(seed) runs once, before the views and the adapter
#       are built.
#   (3) No normalisation here: KarmanH5Dataset returns raw float32 fields and
#       train-only statistics are fitted inside the adapter.
#   train_wallclock_s (build + fit) and infer_latency_ms (predict_eval time
#   per test case) feed the cost table. They are comparable across rows only
#   when each run had the machine to itself.
# ============================================================================
"""Benchmark runner: config -> build -> fit -> predict -> save -> evaluate.

run_benchmark(config_path, *, seed, evaluate=True, out_dir=None) is the entry point for KVS
benchmark runs; scripts/run_benchmark.py is the CLI wrapper. The runner holds no model
hyperparameters; they live in the config YAML.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np

from benchmark.adapters import get_adapter
from benchmark.eval_harness import evaluate as _evaluate
from benchmark.prediction_io import load_prediction, save_prediction, validate
from benchmark.rasterizer import KVS_BBOX
from benchmark.splits import load_split
from tools.common.seeding import set_seed
from tools.data.karman_h5_dataset import KarmanH5Dataset
from tools.runtime.io import read_yaml_or_json, write_json

REPO = Path(__file__).resolve().parents[1]
_DEFAULT_OUT_ROOT = REPO / "results" / "benchmark" / "runs"

_COMMON_KEYS = ("dataset", "split", "adapter", "model_name")
_KVS_KEYS = ("chunk_roots", "mesh_npz", "grid_hw")
_VALID_ADAPTERS = ("grid_convae", "latent_dynamics", "latent_oneshot",
                   "kvs_field_deeponet", "dmdc")
_VALID_DYNAMICS = ("ar_mlp", "s4d_branch_don", "mamba_branch_don", "gru_branch_don")


def load_config(path) -> dict:
    cfg = dict(read_yaml_or_json(path))
    for k in _COMMON_KEYS:
        if k not in cfg:
            raise ValueError(f"config missing required key {k!r}")
    if cfg["dataset"] != "kvs":
        raise ValueError(f"dataset must be 'kvs', got {cfg['dataset']!r}")
    for k in _KVS_KEYS:
        if k not in cfg:
            raise ValueError(f"kvs config missing required key {k!r}")
    if cfg["split"] not in ("interp", "extrap"):
        raise ValueError(f"split must be 'interp'|'extrap', got {cfg['split']!r}")
    if cfg["adapter"] not in _VALID_ADAPTERS:
        raise ValueError(f"unknown adapter {cfg['adapter']!r}; known {_VALID_ADAPTERS}")
    if cfg["adapter"] == "grid_convae" and "raster_cache" not in cfg:
        raise ValueError(f"adapter {cfg['adapter']!r} needs 'raster_cache' in config")
    if cfg["adapter"] == "latent_dynamics":
        if cfg.get("dynamics") not in _VALID_DYNAMICS:
            raise ValueError(f"latent_dynamics needs 'dynamics' in {_VALID_DYNAMICS}, got {cfg.get('dynamics')!r}")
        if not cfg.get("encoder_ref"):
            raise ValueError("latent_dynamics needs 'encoder_ref' (path to frozen grid_convae checkpoint)")
    if cfg["adapter"] == "latent_oneshot":
        if not cfg.get("encoder_ref"):
            raise ValueError("latent_oneshot needs 'encoder_ref' (path to frozen grid_convae checkpoint)")
        bt = str(cfg.get("branch_type", cfg.get("model", {}).get("branch_type", "gru"))).strip().lower()
        if bt not in ("gru", "s4d"):
            raise ValueError(f"latent_oneshot needs branch_type in ('gru','s4d'), got {bt!r}")
    if "model" not in cfg or not isinstance(cfg["model"], dict):
        raise ValueError("config needs a 'model' dict (hyperparameters)")
    if cfg["dataset"] == "kvs":
        gh = list(cfg["grid_hw"])
        if len(gh) != 2:
            raise ValueError(f"grid_hw must be [H, W], got {cfg['grid_hw']!r}")
        cfg["grid_hw"] = [int(gh[0]), int(gh[1])]
    return cfg


def default_run_dir(out_root, model_name: str, split: str, seed: int) -> Path:
    """Default per-run artifact dir. Split-aware so an extrap run does NOT
    overwrite the interp run's best.pt/predictions.npz (and vice versa).
    The results-JSON path is independent of this (eval_harness derives it
    from dataset/split/model/seed)."""
    return Path(out_root) / f"{model_name}_{split}_seed{seed}"


def config_hash(cfg: dict) -> str:
    return hashlib.sha1(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:16]


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


class GridCacheView:
    """mmap view of a kvs_grid_<HxW>.npz cache, indexable by case_id.

    `frames_for(case_ids, dtype=np.float32)` -> (len, T, H, W, 3) in the requested order.
    The cache is built in KarmanH5Dataset registry order, so any subset/reorder works.
    Pass `dtype=np.float16` to skip the fp32 upcast (cache itself is fp16).
    """

    def __init__(self, path):
        self.path = Path(path)
        # Preferred: the extracted raw layout <stem>.data.npy + <stem>.meta.npz.
        # np.load(npz, mmap_mode="r") does not memory-map an npz member: the member is
        # read into RAM in full on access (about 40 GB). Only a plain .npy is mapped.
        # Machines with less RAM than that need the raw layout.
        data_npy = self.path.parent / (self.path.name[:-4] + ".data.npy")
        meta_npz = self.path.parent / (self.path.name[:-4] + ".meta.npz")
        if data_npy.is_file() and meta_npz.is_file():
            self._data = np.load(data_npy, mmap_mode="r")    # true memmap, (n,T,H,W,3) fp16
            meta = np.load(meta_npz)
            self.case_ids = [str(s) for s in meta["case_ids"]]
            self.grid_hw = (int(meta["grid_hw"][0]), int(meta["grid_hw"][1]))
            self.bbox = tuple(float(v) for v in meta["bbox"])
        else:
            self._z = np.load(self.path, mmap_mode="r")
            self._data = self._z["data"]                     # note: loads fully into RAM
            self.case_ids = [str(s) for s in self._z["case_ids"]]
            self.grid_hw = (int(self._z["grid_hw"][0]), int(self._z["grid_hw"][1]))
            self.bbox = tuple(float(v) for v in self._z["bbox"])
        self._pos = {c: i for i, c in enumerate(self.case_ids)}

    def frames_for(self, case_ids, dtype=np.float32) -> np.ndarray:
        idx = [self._pos[str(c)] for c in case_ids]
        return np.asarray(self._data[idx], dtype=dtype)


class DatasetView:
    """Thin handle the adapters consume: KarmanH5Dataset (θ / case_ids / mesh) + the
    matching GridCacheView for grid adapters."""

    def __init__(self, split_name: str, h5ds: KarmanH5Dataset, grid, split_spec, mesh):
        self.split_name = split_name           # "train" | "val" | "test"
        self.h5ds = h5ds
        self.grid = grid
        self.split_spec = split_spec
        self.mesh = mesh                        # dict: coords_p1, coords_p2, fluid_mask_p1, fluid_mask_p2

        # case_ids: public attribute on KarmanH5Dataset, list[str] in split order
        self.case_ids = h5ds.case_ids          # list[str]

        # theta: build from the registry entries selected by the split
        # h5ds._registry[i] = (case_id, h5_path, theta_array(6,))
        # h5ds._sel = sorted indices into _registry for this split
        self._theta = np.stack(
            [h5ds._registry[i][2] for i in h5ds._sel], axis=0
        ).astype(np.float32)                   # (n, 6)

        # t_grid: private but stable attribute, shape (201,) float32
        self._t_grid = h5ds._t_grid.copy()     # (201,) float32

    @property
    def theta(self) -> np.ndarray:
        return self._theta

    @property
    def t_grid(self) -> np.ndarray:
        return self._t_grid

    def frames_for_mesh(self, case_ids, dtype=np.float32):
        """Return (u (N, T, n_p2, 2), p (N, T, n_p1)) in the order of case_ids.

        Reads each case through KarmanH5Dataset.__getitem__, which opens and
        closes one HDF5 file per case. Adapters call it once per split and keep
        the result in memory.
        """
        if self.h5ds is None:
            raise RuntimeError("DatasetView has no h5ds for frames_for_mesh")
        id_to_pos = {c: i for i, c in enumerate(self.case_ids)}
        missing = [c for c in case_ids if c not in id_to_pos]
        if missing:
            raise KeyError(f"case_ids not in split: {missing[:5]}")
        us, ps = [], []
        for c in case_ids:
            item = self.h5ds[id_to_pos[c]]
            us.append(np.asarray(item["u"], dtype=dtype))
            ps.append(np.asarray(item["p"], dtype=dtype))
        return np.stack(us, axis=0), np.stack(ps, axis=0)


def _make_views(cfg: dict, seed: int):
    chunk_roots = [str((REPO / p) if not Path(p).is_absolute() else Path(p)) for p in cfg["chunk_roots"]]
    mesh_npz = (REPO / cfg["mesh_npz"]) if not Path(cfg["mesh_npz"]).is_absolute() else Path(cfg["mesh_npz"])
    with np.load(mesh_npz) as z:
        keys = ["coords_p1", "coords_p2", "fluid_mask_p1", "fluid_mask_p2"]
        mesh = {k: z[k][:].copy() for k in keys}
    split_spec = load_split(cfg["dataset"], cfg["split"])
    grid_path = None
    if "raster_cache" in cfg:
        gp = cfg["raster_cache"]
        grid_path = (REPO / gp) if not Path(gp).is_absolute() else Path(gp)
    # KarmanH5Dataset derives its own ratio-based split (split_runs), which does not
    # match the frozen benchmark JSON (benchmark/splits/<dataset>_<kind>.json).
    # Replace h5ds._sel / case_ids with the frozen IDs, so that all rows and the truth
    # artifacts use the same partition.
    frozen_ids = {"train": split_spec.train_ids, "val": split_spec.val_ids, "test": split_spec.test_ids}
    views = {}
    for split_name in ("train", "val", "test"):
        h5ds = KarmanH5Dataset(chunk_roots, mesh_npz, split=split_name, seed=seed,
                               require_h5=bool(cfg.get("require_h5", True)))
        id_to_reg_idx = {cid: i for i, cid in enumerate(h5ds.all_case_ids)}
        wanted = list(frozen_ids[split_name])
        missing = [c for c in wanted if c not in id_to_reg_idx]
        if missing:
            raise RuntimeError(f"frozen {split_name}-IDs missing from H5 registry: {missing[:5]} (+{len(missing)-5} more)" if len(missing) > 5 else f"frozen {split_name}-IDs missing from H5 registry: {missing}")
        new_sel = np.sort(np.array([id_to_reg_idx[c] for c in wanted], dtype=np.int64))
        h5ds._sel = new_sel
        h5ds.case_ids = [h5ds.all_case_ids[i] for i in new_sel]
        gv = GridCacheView(grid_path) if grid_path is not None else None
        views[split_name] = DatasetView(split_name, h5ds, gv, split_spec, mesh)
    return views, split_spec


def run_benchmark(config_path, *, seed: int, evaluate: bool = True, out_dir=None) -> Path:
    cfg = load_config(config_path)
    set_seed(seed)
    views, split_spec = _make_views(cfg, seed)
    adapter = get_adapter(cfg["adapter"])(cfg)

    t0 = time.time()
    model_state = adapter.build(cfg, seed)
    model_state, hist = adapter.fit(model_state, views["train"], views["val"], cfg, seed)
    train_wallclock_s = time.time() - t0

    out_dir = Path(out_dir) if out_dir is not None else default_run_dir(
        _DEFAULT_OUT_ROOT, cfg["model_name"], cfg["split"], seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "loss_history.json", hist)
    write_json(out_dir / "run_meta.json", {
        "config": cfg, "seed": int(seed), "git_sha": git_sha(), "config_hash": config_hash(cfg),
        "train_wallclock_s": train_wallclock_s,
        "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        **{k: v for k, v in (model_state.get("run_meta") or {}).items()},
    })
    for ck in ("best_ckpt_path", "last_ckpt_path"):
        if model_state.get(ck):
            (out_dir / Path(model_state[ck]).name).write_bytes(Path(model_state[ck]).read_bytes())

    t1 = time.time()
    arrays, meta = adapter.predict_eval(model_state, views["test"], split_spec, cfg)
    infer_latency_ms = (time.time() - t1) * 1000.0 / max(1, len(views["test"].case_ids))
    meta = {"train_wallclock_s": train_wallclock_s, "infer_latency_ms": infer_latency_ms,
            "git_sha": git_sha(), "config_hash": config_hash(cfg),
            "n_params": int(model_state.get("n_params", 0)), **(meta or {})}

    run_dir = out_dir
    save_prediction(run_dir, dataset=cfg["dataset"], split=cfg["split"],
                    model=cfg["model_name"], seed=seed,
                    target_type=meta.get("target_type", "field"), arrays=arrays, meta=meta)
    validate(load_prediction(run_dir), split_spec)
    if evaluate:
        _evaluate(run_dir)
    print(f"[runner] done -> {run_dir}", flush=True)
    return run_dir
