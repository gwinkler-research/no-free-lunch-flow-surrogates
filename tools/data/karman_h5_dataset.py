# ============================================================================
# Purpose: read KVS cases from their per-case HDF5 files on the refined_4x
#   mesh. KarmanH5Dataset builds a registry over the chunk roots, sorted by
#   case_id, and returns per item the raw fields u on the P2 nodes
#   (W, 237351, 2) and p on the P1 nodes (W, 59691), the excitation
#   parameters theta = [U_max_inlet, tau1..tau5], the case_id and the time
#   stamps t_grid. W = 201 (full sequence) unless `window` is set. No
#   normalisation happens here; adapters fit their own statistics.
# Pipeline role: benchmark/runner.py._make_views builds it once per split
#   part, and the adapters read theta, t_grid and the fields through
#   DatasetView. The raster cache, and hence the frozen encoder, is built from
#   these fields.
# Eval-consistency: the split computed here (split_runs(len(registry), 0.7,
#   0.15, seed)) is only a default. benchmark/runner.py replaces _sel and
#   case_ids with the frozen IDs from benchmark/splits/kvs_<kind>.json right
#   after construction. verify_mesh_hash compares the mesh coordinates of each
#   chunk with mesh.npz and refuses incompatible mesh assets. Random time
#   windows are drawn only when `window` is set (from window_seed); the
#   reported rows use full 201-step sequences.
# Without HDF5 files (require_h5=False) the fields come from the raster cache
#   and t_grid is rebuilt from the case config (_snapshot_t_grid).
# ============================================================================
"""Mesh-native dataset over the refined_4x HDF5 chunks.

Reads u (P2) and p (P1) as raw float32 directly from the per-case HDF5 files (no
intermediate format, no normalisation, no sampling). The default split comes from
tools.common.splitting.split_runs on the registry sorted by case_id. It changes when the
number of cases changes, which is why benchmark runs use the frozen split lists instead.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

try:
    import h5py
except ImportError:          # no HDF5 files (require_h5=False, raster cache only):
    h5py = None              # error only on an actual HDF5 access
import numpy as np
import torch
from torch.utils.data import Dataset

from tools.common.splitting import split_runs


def _tkey_to_seconds(key: str) -> float:
    a, b = key.split("_")
    return int(a) + int(b) / 1_000_000.0


def _quantize_tkey(t: float) -> float:
    """The round trip of a time through the HDF5 key encoding "{int}_{frac:06d}",
    so that the analytic grid is bit-identical to the one read from the files."""
    t = float(t)
    ti = int(np.floor(t))
    frac = int(round((t - ti) * 1e6))
    # carry (e.g. 0.9999995 -> frac rounds to 1_000_000)
    ti += frac // 1_000_000
    frac = frac % 1_000_000
    return ti + frac / 1e6


def _snapshot_t_grid(params: dict) -> np.ndarray:
    """Saved snapshot times t = k * dt_sim for k = 0, save_every, ..., T_end / dt_sim,
    from a case config. For the KVS data (T_end = 4, dt_sim = 5e-4, save_every = 40)
    this is the 201-point grid 0, 0.02, ..., 4.0 s."""
    dt_sim = float(params["dt_sim"])
    save_every = int(params["save_every"])
    n_steps = int(round(float(params["T_end"]) / dt_sim))
    ks = np.arange(0, n_steps + 1, save_every)
    return np.array([_quantize_tkey(k * dt_sim) for k in ks], dtype=np.float64)


class KarmanH5Dataset(Dataset):
    """One item = one case (full 201-step sequence or one time window).

    Returns dict:
        u            Tensor (W, n_p2=237351, 2)  float32
        p            Tensor (W, n_p1=59691)       float32
        theta        Tensor (6,)                  float32   # [U_max_inlet, tau1..tau5]
        case_id      str                          # "Umax_05_tau_00"
        t_grid       Tensor (W,)                  float32   # time stamps [s]
        window_start int                          # 0 for the full sequence
    W = window if set, else 201.
    """

    def __init__(
        self,
        chunk_roots,
        mesh_npz,
        split: str,
        *,
        seed: int = 260216,
        ratios: tuple[float, float] = (0.7, 0.15),
        window: int | None = None,
        window_seed: int | None = None,
        verify_mesh_hash: bool = True,
        require_h5: bool = True,
    ):
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be 'train'|'val'|'test', got {split!r}")
        self.chunk_roots = [Path(p) for p in chunk_roots]
        self.split = split
        self.window = int(window) if window is not None else None
        self._window_rng = np.random.default_rng(window_seed) if window_seed is not None else None

        # --- load the mesh asset (plain arrays, fork-safe) ---
        with np.load(Path(mesh_npz)) as z:
            self.coords_p1 = z["coords_p1"]
            self.coords_p2 = z["coords_p2"]
            self.fluid_mask_p1 = z["fluid_mask_p1"]
            self.fluid_mask_p2 = z["fluid_mask_p2"]
            self.edge_index_p2 = z["edge_index_p2"]
            self._mesh_hash = str(z["mesh_hash"])

        # --- registry: (case_id, h5_path, theta), sorted by case_id ---
        reg = []
        for root in self.chunk_roots:
            for case_dir in sorted(root.glob("Umax_*_tau_*")):
                if not case_dir.is_dir():
                    continue
                h5p = case_dir / "fields.h5"
                cfgp = case_dir / "config.json"
                if not cfgp.is_file():
                    continue
                if require_h5 and not h5p.is_file():
                    continue
                cfg = json.loads(cfgp.read_text())
                theta = np.asarray([cfg["U_max_inlet"], *cfg["ramp"]["tau"]], dtype=np.float32)
                if theta.shape != (6,):
                    raise ValueError(f"theta shape {theta.shape} != (6,) in {cfgp}")
                reg.append((case_dir.name, h5p, theta))
        if not reg:
            raise ValueError(f"no cases under {self.chunk_roots}")
        reg.sort(key=lambda r: r[0])
        self._registry = reg
        self.all_case_ids = [r[0] for r in reg]

        # --- Split ---
        train_idx, val_idx, test_idx = split_runs(len(reg), ratios[0], ratios[1], seed)
        sel = {"train": train_idx, "val": val_idx, "test": test_idx}[split]
        self._sel = np.sort(np.asarray(sel))   # registry order, so case_ids stay sorted
        self.case_ids = [reg[i][0] for i in self._sel]

        # --- read the time-step keys once from the first HDF5 file (identical for all cases) ---
        first_h5 = reg[self._sel[0]][1]
        if first_h5.is_file():
            if h5py is None:
                raise ImportError("h5py is required for HDF5 access (pip install h5py)")
            with h5py.File(first_h5, "r") as f:
                ukeys = sorted(f["u"].keys(), key=_tkey_to_seconds)
            self._tkeys = ukeys
            self._t_grid = np.asarray([_tkey_to_seconds(k) for k in ukeys], dtype=np.float32)
            if len(self._tkeys) != 201:
                raise ValueError(f"expected 201 time steps, found {len(self._tkeys)} in {first_h5}")
        else:
            # No HDF5 file (require_h5=False): the fields come from the raster cache and
            # the time grid is rebuilt from the simulation config, identical to the
            # HDF5 time-key grid.
            first_cfg = json.loads((reg[self._sel[0]][1].parent / "config.json").read_text())
            t_grid = _snapshot_t_grid(first_cfg)
            if t_grid.shape[0] != 201:
                raise ValueError(f"expected 201 time steps from config, got {t_grid.shape[0]}")
            self._tkeys = [f"{t:.6f}".replace(".", "_") for t in t_grid]
            self._t_grid = np.asarray(t_grid, dtype=np.float32)

        # --- mesh-hash check on the first HDF5 file of each chunk root ---
        if verify_mesh_hash:
            for root in self.chunk_roots:
                cand = sorted(root.glob("Umax_*_tau_*/fields.h5"))
                if not cand:
                    continue
                if h5py is None:
                    raise ImportError("h5py is required for HDF5 access (pip install h5py)")
                with h5py.File(cand[0], "r") as f:
                    raw_p2 = np.asarray(f["mesh/coordinates_p2"][...], dtype=np.float32)
                h = hashlib.md5(np.ascontiguousarray(raw_p2).tobytes()).hexdigest()
                if h != self._mesh_hash:
                    raise ValueError(
                        f"mesh-hash mismatch: {cand[0]} has {h}, mesh.npz has {self._mesh_hash}"
                    )

    def __len__(self) -> int:
        return len(self._sel)

    def _window_bounds(self) -> tuple[int, int]:
        if self.window is None:
            return 0, 201
        rng = self._window_rng if self._window_rng is not None else np.random.default_rng()
        start = int(rng.integers(0, 201 - self.window + 1))
        return start, start + self.window

    def __getitem__(self, idx: int) -> dict:
        case_id, h5p, theta = self._registry[self._sel[idx]]
        if not h5p.is_file():
            raise RuntimeError(
                f"fields.h5 missing for {case_id}: running without HDF5 files (require_h5=False); "
                "field data must come from the raster cache, not from __getitem__/frames_for_mesh")
        if h5py is None:
            raise ImportError("h5py is required for HDF5 access (pip install h5py)")
        s, e = self._window_bounds()
        keys = self._tkeys[s:e]
        with h5py.File(h5p, "r", swmr=True) as f:
            u = np.stack([np.asarray(f["u"][k][...], dtype=np.float32) for k in keys], axis=0)
            p = np.stack([np.asarray(f["p"][k][...], dtype=np.float32) for k in keys], axis=0)
        return {
            "u": torch.from_numpy(u),
            "p": torch.from_numpy(p),
            "theta": torch.from_numpy(theta.copy()),
            "case_id": case_id,
            "t_grid": torch.from_numpy(self._t_grid[s:e].copy()),
            "window_start": s,
        }
