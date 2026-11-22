#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ============================================================================
# Purpose: CMP counterpart of scripts/run_benchmark.py and home of the CMP
#   metric definition. main() reads a config, seeds, builds the 3D views, runs
#   the adapter's build/fit/predict_eval and writes one JSON. The training
#   loop lives in the adapter. gate2_metrics() computes all CMP metrics from a
#   predicted and a true field: nrmse_3d with its in-plane and u_z parts,
#   rmse_per_zlevel, wafer_rmse, the cum-WSS relative L2 on the inner wafer,
#   and the structure metrics of benchmark/cmp_wss_metrics.py.
# Pipeline role: scripts/build_paper_matrix.py builds the CMP table from the
#   per-seed JSONs of this script, stored as
#   reports/_reinfer_fix/cmp3d_6metric_<split>_<key>_fix_seed<n>.json. The
#   DMDc row comes from scripts/cmp3d_dmdc_gate2.py, which uses gate2_metrics()
#   from this file.
# Eval-consistency: prediction and truth are normalised with the same
#   statistics (stats_mean, stats_std of the NPZ) before any error is taken.
#   The steady window is t >= 1.5 s. Two inner-wafer masks are used, both with
#   R = WAFER_RADIUS_DEFAULT = 0.149 m: the cum-WSS relative L2 uses all
#   in-plane nodes with r/R < 0.9; the structure metrics use _inner90_mask
#   (wafer_mask and r/R <= 0.9). --seed seeds Python, NumPy and PyTorch before
#   the views and the adapter are built. The model name is cfg["model_name"]
#   (default: the config file stem) and names the output file.
# ============================================================================
"""CMP-3D driver: train a 3D surrogate through its adapter and score it.

Runs adapter.build/fit/predict_eval on the extruded 3D node set (the training loop lives in
the adapter), then computes the CMP metrics:
  * nrmse_3d        normalised-field RMSE over all nodes and time steps (ramp and steady),
                    split into nrmse_inplane (u_x, u_y) and nrmse_z (u_z)
  * rmse_per_zlevel normalised RMSE per z-level (is the z-structure learned?)
  * wafer_rmse      normalised RMSE on the wafer-region nodes
  * cumwss_rel_l2   cum-WSS relative L2 on the inner wafer, from the two wall planes
  * the structure metrics of benchmark.cmp_wss_metrics.case_metrics
Truth is the test-set velocity of the same NPZ, de-normalised with stats_mean/stats_std.

  CFG=configs/paper_cmp3d_interp_sdeeponet_nowss_fix.yaml
  python3 -u scripts/run_cmp3d_gate2.py --config $CFG --seed 1
  python3 -u scripts/run_cmp3d_gate2.py --config $CFG --seed 1 --load-best
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import torch

from benchmark.cmp3d_data import make_cmp3d_views
from benchmark.wss_postproc import velocity_to_cum_wss
from benchmark.cmp_wss_metrics import case_metrics          # per-case structure metrics
from benchmark.metrics_cmp import _inner90_mask, WAFER_RADIUS_DEFAULT
from tools.runtime.io import read_yaml_or_json
from tools.common.seeding import set_seed

REPO = Path(_REPO_ROOT)
ADAPTERS = {}


def _get_adapter(name):
    if name == "cmp3d_sdeeponet":
        from benchmark.adapters.cmp3d_sdeeponet import Cmp3dSDeepONetAdapter
        return Cmp3dSDeepONetAdapter()
    if name == "cmp3d_latent_ar":
        from benchmark.adapters.cmp3d_latent_ar import Cmp3dLatentARAdapter
        return Cmp3dLatentARAdapter()
    if name == "cmp3d_latent_oneshot":
        from benchmark.adapters.cmp3d_latent_oneshot import Cmp3dLatentOneShotAdapter
        return Cmp3dLatentOneShotAdapter()
    raise ValueError(f"unknown cmp3d adapter {name!r}")


def _planes_to_4ch(vel, bot_pos, top_pos):
    """vel: (T, Node, 3) physical -> (T, n_ip, 4) [ux_top, uy_top, ux_bot, uy_bot]."""
    return np.concatenate([vel[:, top_pos, :2], vel[:, bot_pos, :2]], axis=-1)


def gate2_metrics(pred_phys, truth_phys, test):
    """pred/truth: (n_case, T, Node, 3) physical velocity. Returns the metric dict
    (case means plus *_per_case lists)."""
    sm = test.stats_mean; ss = test.stats_std
    pn = (pred_phys - sm) / ss
    tn = (truth_phys - sm) / ss
    err = pn - tn
    nrmse_3d = float(np.sqrt(np.mean(err ** 2)))
    # The same normalised error, split by component: in-plane (u_x, u_y) and u_z.
    # u_z is orders of magnitude below the in-plane velocity, but after
    # normalisation its error dominates nrmse_3d.
    nrmse_inplane = float(np.sqrt(np.mean(err[..., :2] ** 2)))
    nrmse_z = float(np.sqrt(np.mean(err[..., 2] ** 2)))
    # per z-level normalised RMSE
    zid = test.z_level_id
    per_z = {}
    for z in np.unique(zid):
        msk = zid == z
        per_z[int(z)] = float(np.sqrt(np.mean(err[:, :, msk] ** 2)))
    # wafer-region RMSE; each case uses its own wafer mask
    wm = test.wafer_mask(test.case_ids)            # (n_case,Node)
    wsq = err ** 2
    wafer_rmse = float(np.sqrt(np.mean(wsq[wm[:, None, :].repeat(err.shape[1], 1)])))
    # cum-WSS relative L2 and the structure metrics of cmp_wss_metrics, both on the two
    # wall planes of the 3D prediction and truth. The cum-WSS relative L2 uses a radius cut
    # only (all in-plane nodes with r/R < 0.9); the structure metrics use _inner90_mask
    # (wafer_mask and r/R <= 0.9). Both use R = WAFER_RADIUS_DEFAULT.
    tg = test.t_grid
    bot = test.wss_bot_pos; top = test.wss_top_pos
    wc = test.wafer_centers(test.case_ids)
    xy_ip = test.node_xyz[bot, :2]                 # in-plane coords (z=0 plane)
    steady = tg >= 1.5
    struct_keys = ["rmse_steady_inner90", "var_ratio_steady", "spatial_corr_inner90",
                   "onset_latency_excess", "radial_profile_l2", "steady_drift_excess",
                   "wiwnu_excess"]
    struct_acc = {k: [] for k in struct_keys}
    rel_list = []
    for c in range(pred_phys.shape[0]):
        p4 = _planes_to_4ch(pred_phys[c], bot, top)
        t4 = _planes_to_4ch(truth_phys[c], bot, top)
        cw_p = velocity_to_cum_wss(p4, tg)
        cw_t = velocity_to_cum_wss(t4, tg)
        r = np.hypot(xy_ip[:, 0] - wc[c, 0], xy_ip[:, 1] - wc[c, 1]) / WAFER_RADIUS_DEFAULT
        inner = r < 0.9
        rel = np.linalg.norm(cw_p[inner] - cw_t[inner]) / (np.linalg.norm(cw_t[inner]) + 1e-12)
        rel_list.append(float(rel))
        im = _inner90_mask(xy_ip, wm[c][bot], wc[c], WAFER_RADIUS_DEFAULT)
        sm5 = case_metrics(p4, t4, steady=steady, t_grid=tg, fem_xy=xy_ip,
                           inner_mask=im, wafer_center=wc[c], truth_cum_ci=cw_t)
        for k in struct_keys:
            struct_acc[k].append(sm5[k])
    out = {
        "nrmse_3d": nrmse_3d,
        "nrmse_inplane": nrmse_inplane,
        "nrmse_z": nrmse_z,
        "wafer_rmse": wafer_rmse,
        "cumwss_rel_l2_inner_mean": float(np.mean(rel_list)),
        "cumwss_rel_l2_inner_per_case": rel_list,
        "rmse_per_zlevel": per_z,
    }
    for k in struct_keys:
        out[k] = float(np.mean(struct_acc[k]))
        out[k + "_per_case"] = struct_acc[k]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=None)
    ap.add_argument("--load-best", action="store_true",
                    help="skip training: load <run_dir>/best.pt and re-run inference")
    args = ap.parse_args()

    cfg = dict(read_yaml_or_json(args.config))
    cfg.setdefault("model_name", Path(args.config).stem)
    set_seed(args.seed)
    t0 = time.time()
    print(f"[gate2] config={args.config} seed={args.seed} adapter={cfg['adapter']}", flush=True)

    views, split_spec = make_cmp3d_views(cfg)
    train, val, test = views["train"], views["val"], views["test"]
    print(f"[gate2] train/val/test cases = {len(train.case_ids)}/{len(val.case_ids)}/{len(test.case_ids)}"
          f"  nodes={train.node_xyz.shape[0]}", flush=True)

    adapter = _get_adapter(cfg["adapter"])
    run_dir = Path(cfg.get("ckpt_dir", f"results/benchmark/runs/{cfg['model_name']}_seed{args.seed}"))
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg["ckpt_dir"] = str(run_dir)

    ms = adapter.build(cfg, args.seed)
    print(f"[gate2] model params = {ms['n_params']:,}", flush=True)
    if args.load_best:
        import torch, copy
        # Run fit() for one epoch into a temporary checkpoint dir, so that the adapter computes
        # its own statistics (z_mean_lat, th_mean, ...) without overwriting the real best.pt.
        # Then load best.pt over the network.
        cfg_s = copy.deepcopy(cfg)
        cfg_s["model"]["epochs"] = 1; cfg_s["model"]["n_epochs"] = 1  # statistics only
        cfg_s["ckpt_dir"] = str(run_dir / "_loadbest_tmp")
        ms, _ = adapter.fit(ms, train, val, cfg_s, args.seed)
        ck = run_dir / "best.pt"
        sd = torch.load(ck, map_location="cpu", weights_only=False)
        state = sd["state_dict"] if isinstance(sd, dict) and "state_dict" in sd else sd
        net = ms.get("model") if ms.get("model") is not None else ms.get("dyn")
        net.load_state_dict(state)
        ms.setdefault("run_meta", {})["total_epochs"] = -1
        print(f"[gate2] LOAD-BEST {ck} (stats via fit epochs=1) net={'model' if ms.get('model') is not None else 'dyn'}", flush=True)
    else:
        ms, hist = adapter.fit(ms, train, val, cfg, args.seed)
        np.savez(run_dir / "loss_curves.npz", train=np.array(hist["train"]), val=np.array(hist["val"]))

    print("[gate2] predicting full 3D field on test ...", flush=True)
    pred = adapter.predict_eval(ms, test, cfg)            # (n_case,T,Node,3) phys
    truth = test.states(test.case_ids) * test.stats_std + test.stats_mean

    met = gate2_metrics(pred, truth, test)
    met["model_name"] = cfg["model_name"]; met["adapter"] = cfg["adapter"]; met["seed"] = args.seed
    met["best_epoch"] = ms["run_meta"]["best_epoch"]; met["total_epochs"] = ms["run_meta"]["total_epochs"]
    met["n_params"] = ms["n_params"]; met["wallclock_s"] = round(time.time() - t0, 1)

    out = Path(args.out) if args.out else REPO / "reports" / f"cmp3d_gate2_{cfg['model_name']}_seed{args.seed}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(met, indent=2))
    print(f"[gate2] nrmse_3d={met['nrmse_3d']:.4f}  wafer_rmse={met['wafer_rmse']:.4f}  "
          f"cumwss_rel_l2_inner={met['cumwss_rel_l2_inner_mean']:.4f}  "
          f"spatial_corr={met['spatial_corr_inner90']:.4f}  drift_exc={met['steady_drift_excess']:.4f}"
          f"  -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
