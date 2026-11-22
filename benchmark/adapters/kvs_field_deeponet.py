# ============================================================================
# Purpose: train and score the field DeepONet, the KVS row with no encoder and
#   no autoregression. The model maps (theta, s(t)) x (x, y, t) straight to
#   (u, v, p), so this adapter mostly handles data and sampling: it derives the
#   TRAIN coordinate bounding box and theta statistics, loads the mesh-native
#   train+val fields once into resident float16 host arrays (from a prebuilt
#   cache directory if cfg["kvs_field_cache"] is set, else case by case through
#   the dataset view), and trains on random (case batch, time window, node)
#   minibatches. predict_eval() evaluates on the full fixed eval node set,
#   chunked, and de-standardises back to physical units.
# Pipeline role: the direct x one-shot cell of the model matrix. Its
#   predictions.npz goes through eval_harness into the field DeepONet row of
#   the KVS table. The network itself lives in tools/models/kvs_field_deeponet.py.
# Eval-consistency: every statistic is TRAIN-only and stored in the checkpoint:
#   the coordinate bounding box from all P2 mesh nodes, theta mean/std from the
#   train cases, and, when normalize_channels is on, the u and p channel
#   statistics from the train arrays. Time is normalised as t / t[-1]. The
#   evaluation nodes come from the fixed SplitSpec (eval_idx_p2 / eval_idx_p1),
#   the same points as every other KVS row. set_seed(seed) runs in build() and
#   fit(). One np.random.RandomState(seed) draws, in order, the fixed validation
#   node subsets, then every epoch's case permutation and every minibatch's time
#   window and node choice. Without a cache, an optional training node pool
#   (train_node_pool_p2/p1) is drawn once with np.random.default_rng(seed).
#   Best-checkpoint selection is on the validation loss, computed every epoch on
#   the fixed node subsets and on the first t_window snapshots only. There is no
#   early stopping. Checkpoints go to
#   results/benchmark/runs/<model_name>_seed<seed>_ckpts/ (the split is not part
#   of the directory name).
# ============================================================================
"""KVS field-space DeepONet adapter (direct map).

Wires tools.models.kvs_field_deeponet.KvsFieldDeepONet into the benchmark runner.
Direct map: no encoder, no autoregression, no raster grid (it works on the mesh
nodes). The model is geometry-free; this adapter does all bbox / theta normalisation.

Trunk input is normalised (x_n, y_n, t_n):
  * coord_bbox = per-axis [min, max] of view.mesh["coords_p2"] (all P2 nodes)
    taken from the TRAIN view, stored in run_meta and the checkpoint as
    `coord_bbox`.
  * t_n = t / t[-1].
Velocity (u, v) is evaluated at coords_p2[eval_idx_p2], pressure (p) at
coords_p1[eval_idx_p1]: two trunk forward passes with shared branch coefficients,
output[..., 0:2] -> u_eval, output[..., 2] -> p_eval, in physical units.

Memory and runtime
------------------
The full KVS mesh has 237 351 P2 nodes and 59 691 P1 nodes; one case is
(201, 237351, 2) + (201, 59691) fp32, about 0.4 GiB. The fields are therefore
kept in host RAM as float16 and loaded once:

  1. At fit start, load the train+val fields (all cases, all 201 snapshots) into
     host np.float16 arrays, either from the prebuilt cache (U.npy, P.npy and
     meta.npz with the node pool and case order) or with one
     `view.frames_for_mesh` call per case. Nothing is re-read in the epoch loop.
     The paper configs use a cache at a fixed pool of 24000 P2 / 8000 P1 nodes.
  2. Training loop: per minibatch pick `case_batch` cases, sample
     `pts_per_sample` resident node indices (fresh each minibatch) and a time
     window of t_window snapshots, slice the resident f16 arrays, cast the slice
     to f32, move only that minibatch to the device and run one batched forward
     (B = case_batch). Loss MSE(u, v) + MSE(p). AdamW + CosineAnnealingLR,
     best.pt + last.pt, EpochLogger.
  3. predict_eval: on the full fixed eval_idx_p2/_p1, the same eval coverage as
     every other KVS row.

Peak host memory with all nodes resident (float16):
  (n_train + n_val) * T * (n_p2 * 2 + n_p1) * 2 B
  interp: 782 * 201 * (237351*2 + 59691) * 2 ~= 156 GiB.
Peak device memory per training step (fp32, about 4x for autograd):
  case_batch * t_window * pts_per_sample * trunk_hidden * 4 B * 4.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from benchmark.adapters import BenchmarkAdapter, register_adapter
from benchmark.adapters import _common
from benchmark.adapters._common import EpochLogger, save_ckpt
from tools.common.seeding import set_seed
from tools.models.kvs_field_deeponet import KvsFieldDeepONet

REPO = Path(__file__).resolve().parents[2]
_T_BENCHMARK = 201   # required output T for the kvs/field schema


def _device(cfg_model: dict) -> torch.device:
    d = str(cfg_model.get("device", "auto"))
    if d == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(d)


def _coord_bbox(coords_p2: np.ndarray) -> np.ndarray:
    """Per-axis [[xmin, xmax], [ymin, ymax]] over all P2 nodes (TRAIN view)."""
    c = np.asarray(coords_p2, dtype=np.float64)
    return np.stack([c.min(axis=0), c.max(axis=0)], axis=1).astype(np.float64)  # (2,2)


def _norm_xy(xy: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    """Map physical (x,y) to [0,1]^2 via the TRAIN coord_bbox (axis-wise)."""
    xy = np.asarray(xy, dtype=np.float64)
    lo = bbox[:, 0]                       # (2,)
    span = np.maximum(bbox[:, 1] - bbox[:, 0], 1e-12)   # (2,)
    return ((xy - lo) / span).astype(np.float32)


@register_adapter("kvs_field_deeponet")
class KvsFieldDeepONetAdapter(BenchmarkAdapter):
    """Direct Branch.Trunk map: (s_seq, theta) x (x_n, y_n, t_n) -> (u, v, p).

    The train+val fields (all cases, all T, at the cache or pool nodes) are
    loaded once into host float16 arrays; the epoch loop only indexes those
    arrays and moves small per-batch slices to the device, running one
    case-batched forward (B = case_batch). The full field is never put on the
    device and no case file is reopened inside the epoch loop.
    """

    def build(self, cfg: dict, seed: int) -> dict:
        set_seed(seed)
        m = cfg["model"]
        dev = _device(m)
        model = KvsFieldDeepONet(
            theta_dim=6,
            branch_hidden=int(m.get("branch_hidden", 128)),
            branch_layers=int(m.get("branch_layers", 2)),
            latent_dim=int(m.get("latent_dim", 128)),
            trunk_hidden=int(m.get("trunk_hidden", 256)),
            trunk_layers=int(m.get("trunk_layers", 4)),
            n_channels=3,
        ).to(dev)
        return {"model": model, "dev": dev, "n_params": int(model.n_params),
                "run_meta": {"adapter": "kvs_field_deeponet"}}

    def fit(self, ms: dict, train, val, cfg: dict, seed: int):
        set_seed(seed)
        m = cfg["model"]
        dev = ms["dev"]
        model: KvsFieldDeepONet = ms["model"]
        n_ep = int(m.get("epochs", 2000))
        # `case_batch` (alias `batch_size`) is the number of cases per optimiser
        # step, sliced out of the resident host array and run in one batched
        # forward.
        case_batch = int(m.get("case_batch", m.get("batch_size", 32)))
        tw = int(m.get("t_window", 32))
        pps = int(m.get("pts_per_sample", 2048))
        lr = float(m.get("lr", 1e-3))
        wd = float(m.get("weight_decay", 1e-5))
        val_pts = int(m.get("val_pts", 4096))
        rng = np.random.RandomState(seed)

        # ---- coord_bbox from the TRAIN view's P2 nodes ----
        bbox = _coord_bbox(train.mesh["coords_p2"])
        ms["coord_bbox"] = bbox

        # ---- theta-norm from TRAIN only ----
        th_tr = np.asarray(train.theta, dtype=np.float32)
        th_mean = th_tr.mean(0)
        th_std = th_tr.std(0) + 1e-6
        ms["theta_mean"] = th_mean
        ms["theta_std"] = th_std

        # ---- t_norm = t / t[-1] ----
        t_grid = np.asarray(train.t_grid, dtype=np.float64)
        t_n_full = (t_grid / max(float(t_grid[-1]), 1e-9)).astype(np.float32)
        t_n = torch.tensor(t_n_full, device=dev)
        T = int(t_grid.size)

        coords_p2 = np.asarray(train.mesh["coords_p2"])
        coords_p1 = np.asarray(train.mesh["coords_p1"])

        # ---- prebuilt field cache: float16 U.npy / P.npy plus meta.npz (node pool,
        # pool coordinates, case order) at a fixed node pool. If `kvs_field_cache`
        # points at that directory, the resident arrays are read from it by row
        # instead of from the per-case files. The bbox still covers all train P2
        # nodes; predict_eval is unaffected (coordinate-based, full eval_idx).
        cache_dir = cfg.get("kvs_field_cache")
        if cache_dir:
            cdir = Path(cache_dir)
            if not cdir.is_absolute():
                cdir = REPO / cdir
            meta = np.load(cdir / "meta.npz", allow_pickle=False)
            sel2 = np.asarray(meta["sel2"]); sel1 = np.asarray(meta["sel1"])
            coords_p2 = np.asarray(meta["coords_p2_pool"])
            coords_p1 = np.asarray(meta["coords_p1_pool"])
            n_p2_full = int(coords_p2.shape[0]); n_p1_full = int(coords_p1.shape[0])
            cache_ids = [str(s) for s in meta["case_ids"]]
            id2row = {c: i for i, c in enumerate(cache_ids)}
            U_all = np.load(cdir / "U.npy", mmap_mode="r")
            P_all = np.load(cdir / "P.npy", mmap_mode="r")

            def _load_resident(view):
                rows = [id2row[c] for c in view.case_ids]
                miss = [c for c in view.case_ids if c not in id2row]
                if miss:
                    raise KeyError(f"cache missing case_ids: {miss[:5]}")
                return (np.ascontiguousarray(U_all[rows]),
                        np.ascontiguousarray(P_all[rows]))
        else:
            # ---- optional fixed training node pool: subsampled once with
            # default_rng(seed). Default None -> all nodes. ----
            pool_p2 = m.get("train_node_pool_p2")
            pool_p1 = m.get("train_node_pool_p1")
            prng = np.random.default_rng(seed)
            n_p2_all = coords_p2.shape[0]
            n_p1_all = coords_p1.shape[0]
            sel2 = (np.sort(prng.choice(n_p2_all, int(pool_p2), replace=False))
                    if pool_p2 and int(pool_p2) < n_p2_all else np.arange(n_p2_all))
            sel1 = (np.sort(prng.choice(n_p1_all, int(pool_p1), replace=False))
                    if pool_p1 and int(pool_p1) < n_p1_all else np.arange(n_p1_all))
            coords_p2 = coords_p2[sel2]
            coords_p1 = coords_p1[sel1]
            n_p2_full = int(coords_p2.shape[0])
            n_p1_full = int(coords_p1.shape[0])

            def _load_resident(view):
                ids = list(view.case_ids)
                nc = len(ids)
                U = np.empty((nc, T, n_p2_full, 2), dtype=np.float16)
                P = np.empty((nc, T, n_p1_full), dtype=np.float16)
                for i, cid in enumerate(ids):
                    u, p = view.frames_for_mesh([cid], dtype=np.float32)
                    U[i] = u[0][:, sel2].astype(np.float16)
                    P[i] = p[0][:, sel1].astype(np.float16)
                    del u, p
                return U, P

        # ---- normalised query coords at the (pooled) train nodes ----
        xy_p2 = torch.tensor(_norm_xy(coords_p2, bbox), device=dev)  # (n_p2,2)
        xy_p1 = torch.tensor(_norm_xy(coords_p1, bbox), device=dev)  # (n_p1,2)

        U_tr, P_tr = _load_resident(train)
        U_va, P_va = _load_resident(val)
        # close any source handle the view may hold open (no re-read after this)
        for v in (train, val):
            h5 = getattr(v, "h5ds", None)
            if h5 is not None and hasattr(h5, "close"):
                try:
                    h5.close()
                except Exception:
                    pass

        # ---- per-channel normalisation (TRAIN stats) ----
        # Raw u, v, p differ ~5-30x in scale (p >> u_y); an unweighted MSE
        # under-fits u_y, which carries the vortex-shedding signal. z-scoring
        # each channel (mean/std over all resident train nodes, times and cases)
        # balances the loss. predict_eval de-standardises back to physical units
        # for the eval harness.
        if bool(m.get("normalize_channels", False)):
            u_mean = U_tr.mean((0, 1, 2), dtype=np.float64).astype(np.float32)   # (2,)
            u_std = (U_tr.astype(np.float32).std((0, 1, 2)) + 1e-6).astype(np.float32)
            p_mean = np.float32(P_tr.mean(dtype=np.float64))
            p_std = np.float32(P_tr.astype(np.float32).std() + 1e-6)
            for arr in (U_tr, U_va):
                for i in range(arr.shape[0]):                # in-place, per case
                    arr[i] = ((arr[i].astype(np.float32) - u_mean) / u_std).astype(np.float16)
            for arr in (P_tr, P_va):
                for i in range(arr.shape[0]):
                    arr[i] = ((arr[i].astype(np.float32) - p_mean) / p_std).astype(np.float16)
            print(f"[kvs_field_don] channel-norm ON  u_std={u_std}  p_std={p_std:.4f}", flush=True)
        else:
            u_mean = np.zeros(2, np.float32); u_std = np.ones(2, np.float32)
            p_mean = np.float32(0.0); p_std = np.float32(1.0)
        ms["u_mean"] = u_mean; ms["u_std"] = u_std
        ms["p_mean"] = p_mean; ms["p_std"] = p_std

        N = len(train.case_ids)
        Nv = len(val.case_ids)

        def _s_seq(view, idx):
            """Piecewise-linear ramp s(t) for one case index -> (T,) fp32."""
            tau = np.asarray(view.theta)[idx, 1:]
            return _common.ramp_s_dsdt(tau, t_grid)[0].astype(np.float32)

        # split pts_per_sample between P2 (velocity) and P1 (pressure) nodes
        n_p2_s = max(1, min(pps // 2, n_p2_full))
        n_p1_s = max(1, min(pps - pps // 2, n_p1_full))

        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max(n_ep, 1), eta_min=lr * 1e-3)
        log = EpochLogger("kvs_field_don", n_ep)
        hist = {"train": [], "val": []}
        best, best_ep = float("inf"), 0
        ck_dir = REPO / "results" / "benchmark" / "runs" / f"{cfg['model_name']}_seed{seed}_ckpts"
        ck_dir.mkdir(parents=True, exist_ok=True)
        best_ckpt = ck_dir / "best.pt"
        last_ckpt = ck_dir / "last.pt"

        def _ckpt_payload(ep):
            return {"state_dict": model.state_dict(), "epoch": ep,
                    "coord_bbox": bbox, "theta_mean": th_mean,
                    "theta_std": th_std, "u_mean": u_mean, "u_std": u_std,
                    "p_mean": p_mean, "p_std": p_std}

        # fixed val node subsets (seed-deterministic), the same every epoch so the
        # val loss is comparable; at most val_pts of the resident nodes.
        v_c2 = rng.choice(n_p2_full, size=min(val_pts, n_p2_full),
                          replace=n_p2_full < val_pts)
        v_c1 = rng.choice(n_p1_full, size=min(val_pts, n_p1_full),
                          replace=n_p1_full < val_pts)
        vt = min(tw, T)
        v_xy_p2 = xy_p2[torch.as_tensor(v_c2, device=dev)]
        v_xy_p1 = xy_p1[torch.as_tensor(v_c1, device=dev)]

        # pre-stack the ramp s(t) per split (theta only, no field data)
        s_tr = np.stack([_s_seq(train, i) for i in range(N)], 0)   # (N,T)
        s_va = np.stack([_s_seq(val, j) for j in range(Nv)], 0)    # (Nv,T)
        th_tr_n = ((th_tr - th_mean) / th_std).astype(np.float32)  # (N,6)
        th_va_n = ((np.asarray(val.theta, np.float32) - th_mean) / th_std
                   ).astype(np.float32)                            # (Nv,6)

        for ep in range(1, n_ep + 1):
            log.epoch_start()
            model.train()
            perm = rng.permutation(N)
            el, nb = 0.0, 0
            for bi in range(0, N, case_batch):
                ci = perm[bi:bi + case_batch]
                B = len(ci)
                t0w = rng.randint(0, max(1, T - tw)) if T > tw else 0
                te = min(t0w + tw, T)
                tw_b = te - t0w
                t_b = t_n[t0w:te]                                  # (tw,)
                # fresh point indices over all resident nodes each minibatch,
                # shared across the case batch and sorted for the slice.
                c2 = rng.choice(n_p2_full, size=n_p2_s,
                                replace=n_p2_full < n_p2_s)
                c1 = rng.choice(n_p1_full, size=n_p1_s,
                                replace=n_p1_full < n_p1_s)
                c2.sort()
                c1.sort()
                # slice the resident host f16 arrays -> small f32 batch ->
                # device. (B,tw,M2,2) / (B,tw,M1).
                tu = torch.as_tensor(
                    np.ascontiguousarray(
                        U_tr[ci][:, t0w:te][:, :, c2].astype(np.float32)),
                    device=dev)                                    # (B,tw,M2,2)
                tp = torch.as_tensor(
                    np.ascontiguousarray(
                        P_tr[ci][:, t0w:te][:, :, c1].astype(np.float32)),
                    device=dev)                                    # (B,tw,M1)
                s_b = torch.as_tensor(
                    s_tr[ci][:, t0w:te], device=dev
                ).reshape(B, tw_b, 1)                               # (B,tw,1)
                th_b = torch.as_tensor(th_tr_n[ci], device=dev)    # (B,6)
                c2_t = torch.as_tensor(c2, device=dev)
                c1_t = torch.as_tensor(c1, device=dev)
                # case-batched forward (all B cases in one pass)
                out2 = model(s_b, th_b, xy_p2[c2_t], t_b)          # (B,tw,M2,3)
                out1 = model(s_b, th_b, xy_p1[c1_t], t_b)          # (B,tw,M1,3)
                loss = (F.mse_loss(out2[..., 0:2], tu)
                        + F.mse_loss(out1[..., 2], tp))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                el += float(loss.item())
                nb += 1
            sch.step()
            tr = el / max(nb, 1)
            # ---- validation: index the resident val array, fixed nodes ----
            model.eval()
            vl_sum, vn = 0.0, 0
            with torch.no_grad():
                for j0 in range(0, Nv, case_batch):
                    js = list(range(j0, min(j0 + case_batch, Nv)))
                    Bv = len(js)
                    tu = torch.as_tensor(
                        np.ascontiguousarray(
                            U_va[js][:, :vt][:, :, v_c2].astype(np.float32)),
                        device=dev)
                    tp = torch.as_tensor(
                        np.ascontiguousarray(
                            P_va[js][:, :vt][:, :, v_c1].astype(np.float32)),
                        device=dev)
                    s_b = torch.as_tensor(
                        s_va[js][:, :vt], device=dev).reshape(Bv, vt, 1)
                    th_b = torch.as_tensor(th_va_n[js], device=dev)
                    vo2 = model(s_b, th_b, v_xy_p2, t_n[:vt])
                    vo1 = model(s_b, th_b, v_xy_p1, t_n[:vt])
                    vl_sum += (F.mse_loss(vo2[..., 0:2], tu)
                               + F.mse_loss(vo1[..., 2], tp)).item()
                    vn += 1
            vl = vl_sum / max(vn, 1)
            hist["train"].append(tr)
            hist["val"].append(vl)
            if vl < best:
                best, best_ep = vl, ep
                save_ckpt(ck_dir, best_ckpt.name, _ckpt_payload(ep))
                if ep == 1 or ep % log.every == 0:
                    log.saved(best_ckpt.name, vl, ep)
            log.epoch_end(ep, tr, vl, best, best_ep)

        save_ckpt(ck_dir, last_ckpt.name, _ckpt_payload(n_ep))
        if best_ckpt.exists():
            model.load_state_dict(torch.load(
                best_ckpt, map_location=dev, weights_only=False)["state_dict"])
        ms.update({
            "model": model, "best_ckpt_path": str(best_ckpt),
            "last_ckpt_path": str(last_ckpt), "best_epoch": best_ep,
            "run_meta": {**ms.get("run_meta", {}), "best_epoch": best_ep,
                         "best_val": best, "total_epochs": n_ep,
                         "n_p2_resident": int(n_p2_full),
                         "n_p1_resident": int(n_p1_full),
                         "case_batch": int(case_batch),
                         "coord_bbox": bbox.tolist()},
        })
        return ms, hist

    def predict_eval(self, ms: dict, test, split_spec, cfg: dict):
        dev = ms["dev"]
        model: KvsFieldDeepONet = ms["model"].to(dev)
        model.eval()
        bbox = np.asarray(ms["coord_bbox"], dtype=np.float64)
        th_mean, th_std = ms["theta_mean"], ms["theta_std"]
        # channel de-standardisation (identity if normalize_channels was off)
        u_mean = np.asarray(ms.get("u_mean", np.zeros(2, np.float32)), np.float32)
        u_std = np.asarray(ms.get("u_std", np.ones(2, np.float32)), np.float32)
        p_mean = np.float32(ms.get("p_mean", 0.0)); p_std = np.float32(ms.get("p_std", 1.0))
        ids = test.case_ids
        n_test = len(ids)

        # full fixed eval_idx (same eval coverage as every KVS row)
        ev_p2 = np.asarray(split_spec.eval_idx_p2, dtype=np.int64)
        ev_p1 = np.asarray(split_spec.eval_idx_p1, dtype=np.int64)
        coords_p2 = np.asarray(test.mesh["coords_p2"])[ev_p2]
        coords_p1 = np.asarray(test.mesh["coords_p1"])[ev_p1]
        xy_p2 = torch.tensor(_norm_xy(coords_p2, bbox), device=dev)
        xy_p1 = torch.tensor(_norm_xy(coords_p1, bbox), device=dev)
        n_ep2, n_ep1 = ev_p2.size, ev_p1.size

        t_grid = np.asarray(test.t_grid, dtype=np.float64)
        T = t_grid.size
        t_n = torch.tensor(
            (t_grid / max(float(t_grid[-1]), 1e-9)).astype(np.float32), device=dev)
        th = torch.tensor((np.asarray(test.theta, np.float32) - th_mean) / th_std,
                          dtype=torch.float32, device=dev)
        s_np = np.stack([_common.ramp_s_dsdt(np.asarray(test.theta)[i, 1:], t_grid)[0]
                         for i in range(n_test)], axis=0).astype(np.float32)
        s_t = torch.tensor(s_np, device=dev)

        u_eval = np.empty((n_test, T, n_ep2, 2), dtype=np.float32)
        p_eval = np.empty((n_test, T, n_ep1), dtype=np.float32)
        # memory guard: chunk the nodes so a forward pass never allocates a
        # device tensor larger than ~ max_pt * T * latent.
        max_pt = max(1, int(8192 * 201 // max(T, 1)))
        with torch.no_grad():
            for i in range(n_test):
                s_seq = s_t[i:i + 1].unsqueeze(-1)            # (1, T, 1)
                th_b = th[i:i + 1]                            # (1, 6)
                for a in range(0, n_ep2, max_pt):
                    sl = slice(a, min(a + max_pt, n_ep2))
                    o = model(s_seq, th_b, xy_p2[sl], t_n)    # (1,T,M,3)
                    u_eval[i, :, sl, :] = o[0, ..., 0:2].cpu().numpy() * u_std + u_mean
                for a in range(0, n_ep1, max_pt):
                    sl = slice(a, min(a + max_pt, n_ep1))
                    o = model(s_seq, th_b, xy_p1[sl], t_n)    # (1,T,M,3)
                    p_eval[i, :, sl] = o[0, ..., 2].cpu().numpy() * p_std + p_mean

        arrays = {
            "u_eval": u_eval,
            "p_eval": p_eval,
            "case_ids": np.array([str(c) for c in ids]),
            "t_grid": t_grid.astype(np.float32),
        }
        meta = {"target_type": "field",
                "coord_bbox": bbox.tolist(),
                "best_epoch": int(ms.get("best_epoch", -1))}
        return arrays, meta
