# Frozen 3D-POD encoder + Branch(theta,ramp).Trunk(t) -> z(t) one-shot latent map (no
# state feedback, no AR rollout); the CMP counterpart of the KVS latent_oneshot adapter.
# Driven via scripts/run_cmp3d_gate2.py (build/fit/predict_eval). Feeds the CMP
# latent_oneshot_{gru,s4d} rows, which compare one-shot against autoregressive time
# treatment (cmp3d_latent_ar) on the same POD latent.
# Normalisation: latent mean/std and theta mean/std from the TRAIN split.
# Seeds: set_seed(seed) in build() and fit(); np.random.default_rng(seed) permutes the
# training cases each epoch.
# Checkpoint selection: fit() computes the validation loss (MSE on the full latent
# trajectory) every val_every epochs and writes best.pt only when it improves.
"""CMP-3D latent x one-shot adapter: frozen 3D-POD encoder + a
Branch(theta,ramp).Trunk(t) -> z(t) one-shot map (no state, no AR feedback).

Counterpart of the KVS latent_oneshot cell. Reuses the POD encoder and the state-encode
helper from cmp3d_latent_ar; the dynamics model is LatentOneShotDON (branch_type gru | s4d).
The ramp input is the dataset control sequence (view.controls). Driven by
scripts/run_cmp3d_gate2.py: predict_eval returns (n_case,T,Node,3) in physical units,
scored by gate2_metrics. Loss: MSE on the normalised z-trajectory.
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from benchmark.adapters.cmp3d_latent_ar import _Pod3dEncoder, _device, _encode_states, REPO
from benchmark.adapters._dynamics_models import LatentOneShotDON
from tools.common.seeding import set_seed


def _branch_type(cfg: dict) -> str:
    return str(cfg.get("branch_type", cfg["model"].get("branch_type", "gru"))).strip().lower()


class Cmp3dLatentOneShotAdapter:
    def build(self, cfg: dict, seed: int) -> dict:
        set_seed(seed)
        m = cfg["model"]; dev = _device(m)
        p = Path(cfg["cmp3d_encoder_ref"]); p = p if p.is_absolute() else REPO / p
        if not p.exists():
            raise FileNotFoundError(f"cmp3d_encoder_ref missing: {p}")
        encoder = _Pod3dEncoder(p, k=cfg.get("pod_k"))
        k = encoder.latent_dim
        dyn = LatentOneShotDON(
            latent_dim=k, theta_dim=6, branch_type=_branch_type(cfg),
            basis_dim=int(m.get("basis_dim", 128)),
            branch_hidden=int(m.get("branch_hidden", 128)),
            branch_layers=int(m.get("branch_layers", 2)),
            branch_width=int(m.get("branch_width", 128)),
            branch_n_layers=int(m.get("branch_n_layers", 4)),
            branch_d_state=int(m.get("branch_d_state", 64)),
            dt_min=float(m.get("dt_min", 0.0001)), dt_max=float(m.get("dt_max", 0.01)),
            trunk_width=int(m.get("trunk_width", 256)),
            trunk_depth=int(m.get("trunk_depth", 4)),
        )
        return {"encoder": encoder, "dyn": dyn, "device": dev, "latent_dim": k,
                "dynamics_kind": f"latent_oneshot_{dyn.branch_type}",
                "n_params": sum(pp.numel() for pp in dyn.parameters()),
                "run_meta": {"encoder_ref": cfg["cmp3d_encoder_ref"],
                             "dynamics": f"latent_oneshot_{dyn.branch_type}", "latent_dim": k}}

    def fit(self, ms, train, val, cfg, seed):
        set_seed(seed)
        m = cfg["model"]; dev = ms["device"]; encoder = ms["encoder"]
        dyn = ms["dyn"].to(dev)
        z_tr = _encode_states(encoder, train)              # (N,T,k)
        z_va = _encode_states(encoder, val)
        N, T, d = z_tr.shape
        zf = z_tr.reshape(-1, d)
        z_mean = zf.mean(0).astype(np.float32); z_std = (zf.std(0) + 1e-8).astype(np.float32)
        z_tr_n = ((z_tr - z_mean) / z_std).astype(np.float32)
        z_va_n = ((z_va - z_mean) / z_std).astype(np.float32)
        th_raw = np.asarray(train.theta, np.float32)
        th_mean = th_raw.mean(0); th_std = th_raw.std(0) + 1e-8
        th_tr_n = (th_raw - th_mean) / th_std
        th_va_n = (np.asarray(val.theta, np.float32) - th_mean) / th_std
        ctrl_tr = np.asarray(train.controls(train.case_ids), np.float32)
        ctrl_va = np.asarray(val.controls(val.case_ids), np.float32)
        tg = np.asarray(train.t_grid, np.float64)
        t_norm = (tg / max(float(tg[-1]), 1e-9)).astype(np.float32)

        z_tr_t = torch.tensor(z_tr_n, device=dev); z_va_t = torch.tensor(z_va_n, device=dev)
        th_tr_t = torch.tensor(th_tr_n, device=dev); th_va_t = torch.tensor(th_va_n, device=dev)
        s_tr_t = torch.tensor(ctrl_tr, device=dev).unsqueeze(-1)   # (N,T,1)
        s_va_t = torch.tensor(ctrl_va, device=dev).unsqueeze(-1)
        t_q = torch.tensor(t_norm, device=dev)                     # (T,)

        n_epochs = int(m["epochs"]); bs = int(m.get("batch_size", 64))
        val_every = int(m.get("val_every", 25)); patience = int(m.get("early_stop_patience", 300))
        opt = torch.optim.AdamW(dyn.parameters(), lr=float(m.get("lr", 5e-4)),
                                weight_decay=float(m.get("weight_decay", 1e-6)))
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(n_epochs, 1))
        rng = np.random.default_rng(seed)
        ck_dir = Path(cfg.get("ckpt_dir", REPO / "results" / "benchmark" / "runs" /
                              f"{cfg['model_name']}_seed{seed}_ckpts"))
        ck_dir.mkdir(parents=True, exist_ok=True)
        best = float("inf"); best_ep = -1; best_state = None
        hist = {"train": [], "val": []}

        for epoch in range(1, n_epochs + 1):
            dyn.train()
            perm = rng.permutation(N); ep_loss, nb = 0.0, 0
            for i in range(0, N, bs):
                idx = torch.as_tensor(perm[i:i + bs], dtype=torch.long, device=dev)
                zt = z_tr_t.index_select(0, idx)
                zp = dyn(th_tr_t.index_select(0, idx), s_tr_t.index_select(0, idx), t_q)
                loss = F.mse_loss(zp, zt)
                opt.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(dyn.parameters(), 1.0)
                opt.step(); ep_loss += float(loss.item()); nb += 1
            sch.step()
            tr = ep_loss / max(nb, 1); vl = float("nan")
            if epoch % val_every == 0 or epoch == n_epochs:
                dyn.eval()
                with torch.no_grad():
                    vl = float(F.mse_loss(dyn(th_va_t, s_va_t, t_q), z_va_t).item())
                # Checkpoint selection on validation loss only, every val_every epochs.
                if vl < best:
                    best = vl; best_ep = epoch; best_state = copy.deepcopy(dyn.state_dict())
                    torch.save({"epoch": epoch, "state_dict": best_state, "loss": best}, ck_dir / "best.pt")
            hist["train"].append(tr); hist["val"].append(vl)
            if epoch == 1 or epoch % 100 == 0 or epoch == n_epochs:
                print(f"[cmp3d_oneshot] ep{epoch} train={tr:.5f} val={vl if vl==vl else tr:.5f} best={best:.5f}@{best_ep}", flush=True)
        torch.save({"epoch": n_epochs, "state_dict": dyn.state_dict()}, ck_dir / "last.pt")
        if best_state is None:
            # No finite validation loss was recorded: keep the final weights as best.pt.
            best = tr; best_ep = n_epochs
            torch.save({"epoch": n_epochs, "state_dict": dyn.state_dict(), "loss": best}, ck_dir / "best.pt")
        else:
            dyn.load_state_dict(best_state)
        ms.update({"dyn": dyn, "z_mean_lat": torch.tensor(z_mean), "z_std_lat": torch.tensor(z_std),
                   "th_mean": torch.tensor(th_mean), "th_std": torch.tensor(th_std),
                   "best_epoch": best_ep, "best_val": best,
                   "run_meta": {**ms.get("run_meta", {}), "best_epoch": best_ep, "best_val": best,
                                "total_epochs": n_epochs}})
        return ms, hist

    def predict_eval(self, ms, test, cfg, **kw):
        dev = ms["device"]; encoder = ms["encoder"]; dyn = ms["dyn"].to(dev).eval()
        z_mean = ms["z_mean_lat"].to(dev); z_std = ms["z_std_lat"].to(dev)
        ids = test.case_ids; n = len(ids)
        st = np.asarray(test.states(ids), np.float32)
        phys = st * test.stats_std + test.stats_mean
        T = st.shape[1]; Node = phys.shape[2]
        tg = np.asarray(test.t_grid, np.float64)
        t_q = torch.tensor((tg / max(float(tg[-1]), 1e-9)).astype(np.float32), device=dev)
        th = (torch.tensor(np.asarray(test.theta, np.float32), device=dev) - ms["th_mean"].to(dev)) / ms["th_std"].to(dev)
        s_t = torch.tensor(np.asarray(test.controls(ids), np.float32), device=dev).unsqueeze(-1)
        with torch.no_grad():
            z_pred = dyn(th, s_t, t_q)                       # (n,T,k) normalised
            z_phys = (z_pred * z_std + z_mean).cpu().numpy()
        vel = np.empty((n, T, Node, 3), np.float32)
        for ci in range(n):
            vel[ci] = encoder.decode(z_phys[ci])
        return vel
