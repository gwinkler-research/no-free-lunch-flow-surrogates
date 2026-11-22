# CMP-3D coordinate-based S-DeepONet trainer (branch: control sequence + theta, trunk:
# wafer-relative coordinates) predicting the 3-channel velocity field directly, without
# an encoder. Feeds the CMP sdeeponet_nowss rows. Driven by scripts/run_cmp3d_gate2.py
# through build/fit/predict_eval (not registered in the benchmark adapter registry).
# Normalisation: theta mean/std from the TRAIN split; velocities are the dataset's
# normalised states (stats_mean/stats_std stored in the npz).
# Seeds: the driver calls set_seed(seed); np.random.RandomState(seed) here draws, in
# order, the fixed validation nodes, then per epoch the case permutation and per batch
# the time window, the WSS-plane points and the per-case sample points.
# Checkpoint selection: fit() evaluates the held-out `val` split every epoch and saves
# best.pt only when the validation loss improves.
"""CMP-3D adapter: coordinate-based S-DeepONet on the full 3D velocity field.

The trunk queries (x, y, z) together with wafer-relative polar features; the output is
the 3-channel velocity (u_x, u_y, u_z). Training loss: field MSE on sampled points
(a fraction wafer_frac drawn from wafer-region nodes) in a random time window of
t_window steps, plus an optional WSS loss (weight wss_lambda) on the gap-difference
shear of (u_x, u_y) between the two WSS planes. The sdeeponet_nowss configs set
wss_lambda to 0. Validation: field MSE on the first t_window steps at val_pts fixed nodes.

Driven by scripts/run_cmp3d_gate2.py via build/fit/predict_eval; the training loop
lives here.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from tools.models.cmp3d_sdeeponet import SDeepONet3D

_MU = 1.0e-3   # dynamic viscosity (Pa s) in the shear of the WSS loss


def _dev(cfg):
    d = str(cfg["model"].get("device", "auto"))
    if d == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(d)


class Cmp3dSDeepONetAdapter:
    def build(self, cfg: dict, seed: int):
        m = cfg["model"]
        dev = _dev(cfg)
        model = SDeepONet3D(
            theta_dim=6,
            branch_hidden=int(m.get("branch_hidden", 128)),
            branch_layers=int(m.get("branch_layers", 2)),
            latent_dim=int(m.get("latent_dim", 128)),
            trunk_hidden=int(m.get("trunk_hidden", 256)),
            trunk_layers=int(m.get("trunk_layers", 4)),
            n_channels=3,
        ).to(dev)
        return {"model": model, "dev": dev,
                "n_params": sum(p.numel() for p in model.parameters())}

    def fit(self, ms, train, val, cfg, seed):
        m = cfg["model"]
        dev = ms["dev"]; model = ms["model"]
        n_ep = int(m.get("n_epochs", 4000))
        bs = int(m.get("batch_size", 4))
        tw = int(m.get("t_window", 32))
        pps = int(m.get("pts_per_sample", 3072))
        wfrac = float(m.get("wafer_frac", 0.5))
        lr = float(m.get("lr", 1e-3))
        wd = float(m.get("weight_decay", 1e-5))
        wl = float(m.get("wss_lambda", 1.0))
        wsc = float(m.get("wss_scale", 50.0))
        n_wss = int(m.get("wss_pts", 2048))
        val_pts = int(m.get("val_pts", 6144))
        rng = np.random.RandomState(seed)

        th_tr = train.theta
        th_mean = th_tr.mean(0); th_std = th_tr.std(0) + 1e-6
        ms["theta_mean"] = th_mean; ms["theta_std"] = th_std
        gap_z = train.gap_z

        def _load(view):
            ids = view.case_ids
            st = torch.tensor(view.states(ids), device=dev)             # (N,T,Node,3) norm
            ct = torch.tensor(view.controls(ids), device=dev).unsqueeze(-1)
            th = torch.tensor((view.theta - th_mean) / th_std, dtype=torch.float32, device=dev)
            wc = torch.tensor(view.wafer_centers(ids), device=dev)
            wm = view.wafer_mask(ids)
            return st, ct, th, wc, wm

        st_tr, ct_tr, th_tr_t, wc_tr, wm_tr = _load(train)
        st_va, ct_va, th_va_t, wc_va, _ = _load(val)
        xyz = torch.tensor(train.node_xyz, device=dev)                  # (Node,3)
        sm = torch.tensor(train.stats_mean, device=dev)                 # (3,)
        ss = torch.tensor(train.stats_std, device=dev)
        bot = torch.tensor(train.wss_bot_pos, device=dev, dtype=torch.long)
        top = torch.tensor(train.wss_top_pos, device=dev, dtype=torch.long)
        n_ip = bot.shape[0]
        N, T, Node, _ = st_tr.shape
        w_idx = [np.where(wm_tr[i])[0] for i in range(N)]
        nw_idx = [np.where(~wm_tr[i])[0] for i in range(N)]
        va_pt = rng.choice(Node, size=min(val_pts, Node), replace=False)
        xyz_va = xyz[va_pt]

        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_ep, eta_min=lr * 1e-3)
        hist = {"train": [], "val": []}
        best = float("inf"); best_ep = 0
        ck_dir = Path(cfg.get("ckpt_dir", f"results/benchmark/runs/{cfg['model_name']}_seed{seed}_ckpts"))
        ck_dir.mkdir(parents=True, exist_ok=True)
        best_path = ck_dir / "best.pt"
        t0 = time.time()
        for ep in range(1, n_ep + 1):
            model.train()
            perm = rng.permutation(N)
            el = 0.0; nb = 0
            for bi in range(0, N, bs):
                ci = perm[bi:bi + bs]
                t0w = rng.randint(0, max(1, T - tw))
                s_seq = ct_tr[ci, t0w:t0w + tw]
                th_b = th_tr_t[ci]; wc_b = wc_tr[ci]
                nwf = int(pps * wfrac); noth = pps - nwf
                tlist = []; plist = []
                wss_p = []; wss_t = []
                wss_ip = rng.choice(n_ip, size=min(n_wss, n_ip), replace=False)
                for k, c in enumerate(ci):
                    wp = w_idx[c]; nwp = nw_idx[c]
                    wi = rng.choice(wp, size=nwf, replace=len(wp) < nwf)
                    ni = rng.choice(nwp, size=noth, replace=len(nwp) < noth)
                    pid = np.concatenate([wi, ni])
                    tgt = st_tr[c, t0w:t0w + tw][:, pid].unsqueeze(0)
                    p = model(s_seq[k:k+1], th_b[k:k+1], xyz[pid], wc_b[k:k+1])
                    tlist.append(tgt); plist.append(p)
                    if wl > 0:
                        bpos = bot[wss_ip]; tpos = top[wss_ip]
                        xyz_bt = torch.cat([xyz[bpos], xyz[tpos]], 0)
                        pbt = model(s_seq[k:k+1], th_b[k:k+1], xyz_bt, wc_b[k:k+1])  # (1,T,2M,3)
                        wss_p.append(pbt)
                        tb = st_tr[c, t0w:t0w + tw][:, bpos]
                        tt = st_tr[c, t0w:t0w + tw][:, tpos]
                        wss_t.append(torch.cat([tb, tt], 1).unsqueeze(0))
                target = torch.cat(tlist, 0); pred = torch.cat(plist, 0)
                loss = F.mse_loss(pred, target)
                if wl > 0:
                    M = wss_ip.shape[0]
                    pb = torch.cat(wss_p, 0)              # (B,T,2M,3)
                    tbt = torch.cat(wss_t, 0)
                    ppb = pb * ss + sm; ttb = tbt * ss + sm
                    def shear(v):
                        vb = v[:, :, :M, :2]; vt = v[:, :, M:, :2]
                        return _MU * torch.sqrt(((vt - vb) / gap_z).pow(2).sum(-1) + 1e-12)
                    loss = loss + wl * F.mse_loss(shear(ppb) / wsc, shear(ttb) / wsc)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                el += loss.item(); nb += 1
            sch.step()
            hist["train"].append(el / max(nb, 1))
            if ep == 1:
                dt1 = time.time() - t0
                print(f"[cmp3d] ep1: {dt1:.1f}s -> est. total: {dt1 * n_ep / 60:.0f} min", flush=True)
            model.eval()
            with torch.no_grad():
                vp = model(ct_va[:, :tw], th_va_t, xyz_va, wc_va)
                vl = F.mse_loss(vp, st_va[:, :tw][:, :, va_pt]).item()
            hist["val"].append(vl)
            # Checkpoint selection on validation loss only.
            if vl < best:
                best = vl; best_ep = ep
                torch.save(model.state_dict(), best_path)
                if ep % 10 == 0 or ep == 1:
                    print(f"[cmp3d] Saved best.pt (val={vl:.6f}, ep={ep})", flush=True)
            if ep % 10 == 0 or ep == n_ep:
                print(f"[cmp3d] ep{ep} train={hist['train'][-1]:.5f} val={vl:.5f} best={best:.5f}@{best_ep}", flush=True)
        model.load_state_dict(torch.load(best_path, map_location=dev))
        ms["best_ckpt_path"] = str(best_path)
        ms["run_meta"] = {"best_epoch": best_ep, "total_epochs": n_ep}
        return ms, hist

    def predict_eval(self, ms, test, cfg, *, query_pos=None, chunk=20000):
        """Predict velocity at node positions `query_pos` (default: all nodes).

        The model output is normalised and is mapped back with the test view's
        stats_mean/stats_std. Returns vel (n_case, T, len(query_pos), 3) in physical units."""
        dev = ms["dev"]; model = ms["model"]; model.eval()
        ids = test.case_ids
        if query_pos is None:
            query_pos = np.arange(test.node_xyz.shape[0])
        xyz_all = torch.tensor(test.node_xyz[query_pos], device=dev)
        th = torch.tensor((test.theta - ms["theta_mean"]) / ms["theta_std"], dtype=torch.float32, device=dev)
        ct = torch.tensor(test.controls(ids), device=dev).unsqueeze(-1)
        wc = torch.tensor(test.wafer_centers(ids), device=dev)
        sm = test.stats_mean.astype(np.float32); ss = test.stats_std.astype(np.float32)
        out = []
        with torch.no_grad():
            for i in range(len(ids)):
                parts = []
                for s in range(0, xyz_all.shape[0], chunk):
                    p = model(ct[i:i+1], th[i:i+1], xyz_all[s:s+chunk], wc[i:i+1])
                    parts.append(p[0].cpu().numpy())
                out.append(np.concatenate(parts, 1))
        vel = np.stack(out, 0).astype(np.float32) * ss + sm
        return vel
