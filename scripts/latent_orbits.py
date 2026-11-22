#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ============================================================================
# Purpose: re-infer a KVS AR/one-shot pair on the frozen interp encoder and
#   STORE the latent trajectories, which the benchmark runs do not keep
#   (predictions.npz holds decoded fields only). Default pair: GRU-branch DON
#   (latent-AR) against the GRU one-shot (latent one-shot); --pair s4d does the
#   same for the S4D pair. This is the raw material for what the two time
#   treatments do inside the shared latent space.
# Pipeline role: the latent-orbit numbers of the Discussion. Needs a GPU, the
#   run directories of the pair (gru_1phase4k_clean_interp_seed{1,2,3} and
#   latent_oneshot_gru_tuned_clean_interp_seed{1,2,3}, or for --pair s4d
#   s4d_1phase4k_clean_interp_seed{1,2,3} and
#   latent_oneshot_s4d_tuned_clean_interp_seed{1,2,3}) and the frozen interp
#   encoder. The run directories are not in this repository (--runs-root).
#   Writes results/latent_orbits_{gru,s4d}/interp/{latents.npz, stats.json,
#   verify.json}; scripts/latent_orbit_stats.py reads latents.npz. The model
#   classes come from each run's run_meta.json (build_dynamics via cfg["dynamics"],
#   LatentOneShotAdapter.build via cfg["branch_type"]), so the inference code
#   below serves both pairs unchanged.
# Eval-consistency: the inference paths are copied 1:1 from
#   latent_dynamics._predict_eval_s4d_branch_don and latent_oneshot.predict_eval.
#   --verify decodes the stored latents and compares the fields against the
#   run's predictions.npz (rel_rms 0.0 expected), so the latents are exactly
#   those behind the paper table. stats.json carries the latent power that
#   scripts/latent_orbit_stats.py must reproduce.
# ============================================================================
"""Latent trajectories of a KVS AR/one-shot pair (default: GRU pair).

Re-infers the trained GRU-DON (latent x AR) and the GRU one-shot (latent x one-shot)
on the frozen bc32 interp encoder and stores the LATENT trajectories, which the
benchmark runs do not keep (predictions.npz holds decoded fields only). --pair s4d
does the same for S4D-DON and the S4D one-shot. Inference paths are copied 1:1 from
latent_dynamics._predict_eval_s4d_branch_don and latent_oneshot.predict_eval, so the
latents are exactly those behind the paper tables.

Outputs (results/latent_orbits_gru/<split>/ for --pair gru (default),
         results/latent_orbits_s4d/<split>/ for --pair s4d):
  latents.npz   z_true_train/test, z_ar (seeds,N,T,d), z_os (seeds,N,T,d), theta, ids, t_grid
  stats.json    latent amplitude ratio (t >= 2 s), distance-to-time-mean ratios, PCA, winding
  verify.json   (--verify) decoded-field reproduction check against the stored predictions

  python3 -u scripts/latent_orbits.py --split interp --seeds 1 2 3 --device cuda:0 --verify \
      --runs-root <directory with the run directories>
  python3 -u scripts/latent_orbits.py --pair s4d --split interp --seeds 1 2 3 --device cuda:0 --verify \
      --runs-root <directory with the run directories>
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

T_STEADY = 2.0                                       # steady window t >= 2 s (as for P)

# The run directories are not in this repository; --runs-root points at them,
# --pair picks the AR/one-shot family (KVS interp runs of the paper).
RUNS_ROOT = _REPO / "results" / "benchmark" / "runs"
PAIRS = {
    "gru": {"ar": "gru_1phase4k_clean_interp_seed{seed}",
            "os": "latent_oneshot_gru_tuned_clean_interp_seed{seed}",
            "out": "results/latent_orbits_gru"},
    "s4d": {"ar": "s4d_1phase4k_clean_interp_seed{seed}",
            "os": "latent_oneshot_s4d_tuned_clean_interp_seed{seed}",
            "out": "results/latent_orbits_s4d"},
}
PAIR = "gru"


def _ar_run_dir(split: str, seed: int) -> Path:
    assert split == "interp", "the orbit analysis covers the interp split only"
    return RUNS_ROOT / PAIRS[PAIR]["ar"].format(seed=seed)


def _os_run_dir(split: str, seed: int) -> Path:
    assert split == "interp", "the orbit analysis covers the interp split only"
    return RUNS_ROOT / PAIRS[PAIR]["os"].format(seed=seed)


def _load_cfg(run_dir: Path, device: str) -> dict:
    cfg = json.load(open(run_dir / "run_meta.json"))["config"]
    cfg["model"]["device"] = device
    return cfg


def _theta_norm_train(train):
    th = np.asarray(train.theta, np.float32)
    return th.mean(0), th.std(0) + 1e-8


def _ds_span(s_t, dt):
    import torch
    ds_t = torch.zeros_like(s_t)
    T = s_t.shape[1]
    ds_t[:, 1:-1] = (s_t[:, 2:] - s_t[:, :-2]) / (2 * dt)
    ds_t[:, 0] = (s_t[:, 1] - s_t[:, 0]) / dt
    ds_t[:, -1] = (s_t[:, -1] - s_t[:, -2]) / dt
    return ds_t


def rollout_ar(cfg, seed, test, ae, x_mean, x_std, z_mean_lat, z_std_lat, th_mean, th_std, dev):
    """Branch-DON rollout (GRU-DON or S4D-DON), identical to
    latent_dynamics._predict_eval_s4d_branch_don."""
    import torch
    from benchmark.adapters._common import grid_to_channels_first
    from benchmark.adapters._dynamics_models import build_dynamics
    from benchmark.adapters.latent_dynamics import _control_seq_stack
    m = cfg["model"]
    dyn = build_dynamics(cfg["dynamics"], latent_dim=ae.latent_dim, theta_dim=6, hp=m)
    ck = torch.load(_ar_run_dir(cfg["split"], seed) / "best.pt", map_location="cpu", weights_only=False)
    dyn.load_state_dict(ck["state_dict"])
    dyn = dyn.to(dev).eval()
    x0_np = np.stack([test.grid.frames_for([cid], dtype=np.float16)[0, 0] for cid in test.case_ids], axis=0)
    n_test, T = x0_np.shape[0], len(test.t_grid)
    t_grid = np.asarray(test.t_grid, dtype=np.float64)
    dt = float(t_grid[1] - t_grid[0])
    th = (torch.from_numpy(test.theta).float().to(dev) - torch.tensor(th_mean, device=dev)) / torch.tensor(th_std, device=dev)
    s = _control_seq_stack(test.theta.astype(np.float32), t_grid, test.case_ids)[0]
    s_t = torch.from_numpy(s).to(dev)
    assert str(m.get("ds_fd", "span")) == "span", m.get("ds_fd")
    ds_t = _ds_span(s_t, dt)
    bf = (th, s_t.unsqueeze(-1))
    zm, zs = torch.tensor(z_mean_lat, device=dev), torch.tensor(z_std_lat, device=dev)
    with torch.no_grad():
        x0 = torch.from_numpy(grid_to_channels_first(x0_np)).to(dev).float()
        z_ar = (ae.encode((x0 - x_mean) / x_std) - zm) / zs
        preds = [z_ar]
        for t_step in range(T - 1):
            t_f = torch.full((n_test, 1), t_step / T, dtype=torch.float32, device=dev)
            trunk_in = torch.cat([z_ar, t_f, s_t[:, t_step].unsqueeze(1), ds_t[:, t_step].unsqueeze(1)], dim=1)
            z_ar = z_ar + dyn(bf, trunk_in)
            preds.append(z_ar)
        z_pred = torch.stack(preds, dim=1)
        z_phys = z_pred * zs + zm
    return z_phys.cpu().numpy().astype(np.float32), int(ck.get("epoch", -1))


def forward_oneshot(split, seed, test, dev):
    """One-shot forward (GRU or S4D), identical to latent_oneshot.predict_eval.
    Returns PHYSICAL latents."""
    import torch
    from benchmark.adapters._common import ramp_s_dsdt
    from benchmark.adapters.latent_oneshot import LatentOneShotAdapter
    run = _os_run_dir(split, seed)
    cfg = _load_cfg(run, str(dev))
    ad = LatentOneShotAdapter(cfg)
    ms = ad.build(cfg, seed)
    ck = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
    dyn = ms["dyn"]
    dyn.load_state_dict(ck["dyn_state_dict"])
    dyn = dyn.to(dev).eval()
    t_grid = np.asarray(test.t_grid, dtype=np.float64)
    n_test = len(test.case_ids)
    t_norm = (t_grid / max(float(t_grid[-1]), 1e-9)).astype(np.float32)
    th = torch.from_numpy(test.theta.astype(np.float32)).to(dev)
    s = np.stack([ramp_s_dsdt(test.theta[i, 1:], t_grid)[0] for i in range(n_test)], 0).astype(np.float32)
    with torch.no_grad():
        z_pred = dyn(th, torch.from_numpy(s).unsqueeze(-1).to(dev), torch.from_numpy(t_norm).to(dev))
        z_phys = z_pred * ms["z_std"] + ms["z_mean"]
    return z_phys.cpu().numpy().astype(np.float32), int(ck.get("epoch", -1)), ms


def verify_against_stored(run_dir, z_phys, ae, x_mean, x_std, test, split_spec, n_check=2):
    """Decode the re-inferred latents for n_check cases and compare with the stored
    predictions.npz of the same run (the paper's evaluated fields)."""
    import torch
    from benchmark.adapters._common import assemble_kvs_field_arrays, channels_first_to_grid
    stored = np.load(run_dir / "predictions.npz")
    ids = [str(c) for c in stored["case_ids"]]
    out = {}
    H, W = ae.grid_h, ae.grid_w
    with torch.no_grad():
        for cid in list(test.case_ids)[:n_check]:
            i = list(test.case_ids).index(cid)
            dec = ae.decode(torch.from_numpy(z_phys[i]).to(x_mean.device)) * x_std + x_mean
            grid = channels_first_to_grid(dec.cpu().numpy()[None])            # (1,T,H,W,3)
            arrays = assemble_kvs_field_arrays(
                grid, [cid], np.asarray(test.t_grid, np.float32),
                coords_p2=test.mesh["coords_p2"], coords_p1=test.mesh["coords_p1"],
                eval_idx_p2=split_spec.eval_idx_p2, eval_idx_p1=split_spec.eval_idx_p1,
                bbox=test.grid.bbox)
            u_new = np.asarray(arrays["u_eval"][0], np.float32)
            u_old = np.asarray(stored["u_eval"][ids.index(cid)], np.float32)
            out[cid] = {"max_abs_diff": float(np.abs(u_new - u_old).max()),
                        "rel_rms": float(np.sqrt(np.mean((u_new - u_old) ** 2)) / (np.sqrt(np.mean(u_old ** 2)) + 1e-12))}
    return out


def _steady_mask(t_grid):
    return np.asarray(t_grid) >= T_STEADY - 1e-9


def _amp_ratio(z_pred, z_true, mask):
    """Latent analogue of the shedding power P: pooled temporal variance in the steady window."""
    vp = z_pred[:, mask].var(axis=1).sum()
    vt = z_true[:, mask].var(axis=1).sum()
    return float(vp / vt)


def _dist_to_mean_ratio(z_pred, z_true, mask):
    """RMS distance of the prediction to the per-case steady-window time mean of the truth,
    normalised by the truth's own RMS excursion about that mean. ~0: prediction sits on the
    mean (collapse); ~1: prediction lives at the truth's amplitude (phase error or not)."""
    mu = z_true[:, mask].mean(axis=1, keepdims=True)
    num = np.sqrt(((z_pred[:, mask] - mu) ** 2).sum(axis=-1).mean(axis=1))
    den = np.sqrt(((z_true[:, mask] - mu) ** 2).sum(axis=-1).mean(axis=1))
    r = num / (den + 1e-12)
    return {"median": float(np.median(r)), "mean": float(r.mean()), "per_case": r.astype(float).tolist()}


def _pca_plane(z_train, mask):
    """Orbit plane: PCA of per-case-centred steady-window truth latents (train)."""
    zc = z_train[:, mask] - z_train[:, mask].mean(axis=1, keepdims=True)
    X = zc.reshape(-1, zc.shape[-1])
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    ev = (S ** 2) / (S ** 2).sum()
    return Vt[:2].astype(np.float32), ev[:4].astype(float).tolist()


def _phase(z, Vt2, mask):
    zc = z[:, mask] - z[:, mask].mean(axis=1, keepdims=True)
    p = zc @ Vt2.T                                            # (N, Ts, 2)
    return np.arctan2(p[..., 1], p[..., 0]), p


def _parse_ids(case_ids):
    um, tau = [], []
    for c in case_ids:
        parts = str(c).split("_")
        um.append(int(parts[1])); tau.append(int(parts[3]))
    return np.array(um), np.array(tau)


def _winding(z_all, ids_all, Vt2, t_grid, mask):
    """Phase winding across the U_max axis for each shared ramp schedule (tau index):
    for each steady time t*, the number of 2*pi wraps of the truth phase across the sorted
    U_max ladder, estimated from the per-case steady frequency (unwrapped phase slope)."""
    phi, _ = _phase(z_all, Vt2, mask)                          # (N, Ts)
    ts = np.asarray(t_grid)[mask]
    # per-case angular frequency from the unwrapped phase over the steady window
    omega = np.array([np.polyfit(ts, np.unwrap(phi[i]), 1)[0] for i in range(phi.shape[0])])
    um, tau = _parse_ids(ids_all)
    out = {"t_star": [float(t) for t in ts[::10]], "omega_per_case": omega.astype(float).tolist(),
           "umax_idx": um.tolist(), "tau_idx": tau.tolist(), "per_tau": {}}
    for tv in np.unique(tau):
        sel = np.where(tau == tv)[0]
        order = sel[np.argsort(um[sel])]
        om = omega[order]
        # neighbour phase gap grows linearly in t: |dphi| = t * |domega|
        dom = np.abs(np.diff(om))
        gaps = {float(t): float(np.mean(t * dom > np.pi)) for t in ts[::10]}   # fraction of aliased neighbours
        wraps = {float(t): float(t * (om.max() - om.min()) / (2 * np.pi)) for t in ts[::10]}
        out["per_tau"][int(tv)] = {"n_cases": int(len(order)), "frac_neighbours_aliased": gaps,
                                   "wraps_across_ladder": wraps,
                                   "phase_at_tstar": {float(ts[k]): phi[order, k].astype(float).tolist() for k in range(0, len(ts), 25)},
                                   "umax_idx_sorted": um[order].tolist()}
    return out


def main():
    global RUNS_ROOT, PAIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", choices=sorted(PAIRS), default="gru",
                    help="AR/one-shot run family: gru (default) or s4d")
    ap.add_argument("--split", choices=["interp", "extrap"], default="interp")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=None,
                    help="output root; default results/latent_orbits_gru (gru) or "
                         "results/latent_orbits_s4d (s4d)")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--runs-root", default=str(RUNS_ROOT),
                    help="directory holding the run directories of the pair (not in this repository)")
    args = ap.parse_args()
    RUNS_ROOT = Path(args.runs_root)
    PAIR = args.pair
    if args.out is None:
        args.out = PAIRS[PAIR]["out"]

    import torch
    from benchmark.runner import _make_views
    from benchmark.adapters.latent_dynamics import _load_frozen_encoder, _encode_split_streaming
    from tools.common.seeding import set_seed

    dev = torch.device(args.device)
    out_dir = _REPO / args.out / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    cfg = _load_cfg(_ar_run_dir(args.split, args.seeds[0]), args.device)
    set_seed(args.seeds[0])
    views, split_spec = _make_views(cfg, args.seeds[0])
    train, test = views["train"], views["test"]
    print(f"[latent-orbits] pair={PAIR} split={args.split} train={len(train.case_ids)} test={len(test.case_ids)} "
          f"encoder={cfg['encoder_ref']} dynamics={cfg['dynamics']}", flush=True)

    ae, (H, W), d, z_mean, z_std, x_mean, x_std = _load_frozen_encoder(cfg["encoder_ref"], dev)
    z_true_train = _encode_split_streaming(ae, train.grid, train.case_ids, dev, x_mean, x_std)
    z_true_test = _encode_split_streaming(ae, test.grid, test.case_ids, dev, x_mean, x_std)
    print(f"[latent-orbits] encoded truth: train {z_true_train.shape} test {z_true_test.shape} ({time.time()-t0:.0f}s)", flush=True)
    zf = z_true_train.reshape(-1, d)
    z_mean_lat, z_std_lat = zf.mean(0).astype(np.float32), (zf.std(0) + 1e-8).astype(np.float32)
    th_mean, th_std = _theta_norm_train(train)

    t_grid = np.asarray(test.t_grid, np.float32)
    mask = _steady_mask(t_grid)
    z_ar, z_os, meta, verify = [], [], {"ar": {}, "os": {}}, {"ar": {}, "os": {}}
    for seed in args.seeds:
        za, ep_a = rollout_ar(cfg, seed, test, ae, x_mean, x_std, z_mean_lat, z_std_lat, th_mean, th_std, dev)
        zo, ep_o, ms_os = forward_oneshot(args.split, seed, test, dev)
        z_ar.append(za); z_os.append(zo)
        meta["ar"][seed] = {"best_epoch": ep_a, "run": str(_ar_run_dir(args.split, seed))}
        meta["os"][seed] = {"best_epoch": ep_o, "run": str(_os_run_dir(args.split, seed))}
        print(f"[latent-orbits] seed{seed}: AR P_lat={_amp_ratio(za, z_true_test, mask):.3f}  "
              f"one-shot P_lat={_amp_ratio(zo, z_true_test, mask):.3f}  ({time.time()-t0:.0f}s)", flush=True)
        if args.verify:
            verify["ar"][seed] = verify_against_stored(_ar_run_dir(args.split, seed), za, ae, x_mean, x_std, test, split_spec)
            verify["os"][seed] = verify_against_stored(_os_run_dir(args.split, seed), zo, ae, x_mean, x_std, test, split_spec)
            print(f"[latent-orbits] verify seed{seed}: AR {verify['ar'][seed]}  OS {verify['os'][seed]}", flush=True)
    z_ar = np.stack(z_ar); z_os = np.stack(z_os)

    Vt2, ev = _pca_plane(z_true_train, mask)
    z_all = np.concatenate([z_true_train, z_true_test], axis=0)
    ids_all = list(train.case_ids) + list(test.case_ids)
    stats = {
        "split": args.split, "seeds": args.seeds, "t_steady": T_STEADY,
        "pca_explained_first4": ev,
        "P_latent": {"ar": [_amp_ratio(z, z_true_test, mask) for z in z_ar],
                     "os": [_amp_ratio(z, z_true_test, mask) for z in z_os]},
        "dist_to_truth_time_mean": {"ar": [_dist_to_mean_ratio(z, z_true_test, mask) for z in z_ar],
                                    "os": [_dist_to_mean_ratio(z, z_true_test, mask) for z in z_os]},
        "winding": _winding(z_all, ids_all, Vt2, t_grid, mask),
        "meta": meta,
    }
    np.savez_compressed(out_dir / "latents.npz",
                        z_true_train=z_true_train, z_true_test=z_true_test, z_ar=z_ar, z_os=z_os,
                        theta_train=np.asarray(train.theta, np.float32), theta_test=np.asarray(test.theta, np.float32),
                        ids_train=np.array(list(train.case_ids)), ids_test=np.array(list(test.case_ids)),
                        t_grid=t_grid, pca_Vt2=Vt2, z_mean_lat=z_mean_lat, z_std_lat=z_std_lat,
                        seeds=np.array(args.seeds))
    json.dump(stats, open(out_dir / "stats.json", "w"), indent=1)
    if args.verify:
        json.dump(verify, open(out_dir / "verify.json", "w"), indent=1)
    print(f"[latent-orbits] P_latent AR {stats['P_latent']['ar']}  OS {stats['P_latent']['os']}")
    print(f"[latent-orbits] dist-to-mean median AR {[round(s['median'],3) for s in stats['dist_to_truth_time_mean']['ar']]}"
          f"  OS {[round(s['median'],3) for s in stats['dist_to_truth_time_mean']['os']]}")
    print(f"[latent-orbits] PCA explained {ev}")
    print(f"[latent-orbits] done -> {out_dir} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
