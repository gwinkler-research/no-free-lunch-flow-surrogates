# =====================================================================
# CMP-3D counterpart of the KVS latent-dynamics adapter: a frozen 3D-POD encoder
# (_Pod3dEncoder, defined here) projects the physical velocity field to a latent,
# an autoregressive dynamics model (build_dynamics from _dynamics_models.py)
# integrates z(t+1) = z(t) + dz, and the POD basis maps back to the full 3D
# field. Driven by scripts/run_cmp3d_gate2.py.
#
# Feeds the CMP latent_ar_{mamba,s4d,gru,armlp}_k8 rows (interp + extrap) in the
# paper's results table.
#
# Normalisation: latent mean/std and theta mean/std come from the TRAIN split.
# Seeds: set_seed(seed) in build() and fit(); a local np.random.RandomState(seed)
# draws the case batch and the rollout start offsets of every epoch.
# Checkpoint selection: best.pt is the epoch with the lowest full-horizon
# validation rollout MSE (_val_rollout_loss), evaluated every val_every epochs
# and only from best_mature_epoch_phase1 on. The validation pass runs under
# no_grad and does not touch `rng`, so it does not change the training
# trajectory.
# =====================================================================
"""CMP-3D latent-AR adapter: frozen 3D-POD encoder + autoregressive latent dynamics.

The POD encoder projects the physical 3D velocity field (T,Node,3) -> latent z(T,k);
an AR dynamics model (build_dynamics, e.g. mamba_branch_don) rolls the latent forward;
the POD basis decodes back to the 3D field. The training loop (one-step dz MSE,
curriculum rollout MSE, variance penalty once the horizon reaches five steps) is the
same as in branch_don_training.train_phase1 for KVS.

Driven by scripts/run_cmp3d_gate2.py: predict_eval returns the full 3D field
(n_case,T,Node,3) in physical units (no resampling; T is already 201).
"""
from __future__ import annotations

import copy
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from benchmark.adapters._dynamics_models import build_dynamics
from tools.common.seeding import set_seed

REPO = Path(__file__).resolve().parents[2]


def _device(cfg_model: dict) -> torch.device:
    d = str(cfg_model.get("device", "auto"))
    if d == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(d)


class _Pod3dEncoder:
    """Frozen 3D-POD encoder read from an npz (basis, mean, n_node, n_channels).

    encode/decode map physical (T,Node,3) <-> (T,k). `k` truncates the
    (energy-ordered) basis to the first k modes (None = all)."""

    def __init__(self, npz_path: Path, k: int | None = None):
        z = np.load(npz_path, allow_pickle=False)
        basis = np.asarray(z["basis"], np.float64)           # (k_full, D)
        if k is not None and k < basis.shape[0]:
            basis = basis[:k]
        self.basis = basis
        self.mean = np.asarray(z["mean"], np.float64)        # (D,)
        self.latent_dim = basis.shape[0]
        self.n_node = int(z["n_node"]); self.n_channels = int(z["n_channels"])

    def encode(self, vel_phys: np.ndarray) -> np.ndarray:
        T = vel_phys.shape[0]
        X = vel_phys.reshape(T, -1).astype(np.float64) - self.mean
        return (X @ self.basis.T).astype(np.float32)         # (T,k)

    def decode(self, z: np.ndarray) -> np.ndarray:
        X = np.asarray(z, np.float64) @ self.basis + self.mean
        return X.reshape(z.shape[0], self.n_node, self.n_channels).astype(np.float32)


# De-normalise view.states() with the dataset statistics and POD-encode every case.
def _encode_states(encoder, view):
    ids = view.case_ids
    st = np.asarray(view.states(ids), np.float32)
    phys = st * view.stats_std + view.stats_mean
    return np.stack([encoder.encode(phys[i]) for i in range(st.shape[0])], 0).astype(np.float32)


class _ARMlpBranchTrunkShim(torch.nn.Module):
    """Adapts ARMlpStepper.forward(z, theta, t, s, ds) onto the branch-trunk
    call convention dyn(branch_in=(theta, ramp_seq), trunk_in=[z, t/T, s, ds])
    that fit/predict_eval use, so all dynamics kinds share one training and
    inference loop. ramp_seq is unused by the stepper."""

    def __init__(self, stepper: torch.nn.Module, latent_dim: int):
        super().__init__()
        self.stepper = stepper
        self.latent_dim = int(latent_dim)

    def forward(self, branch_in, trunk_in):
        theta = branch_in[0]
        k = self.latent_dim
        z, t, s, ds = (trunk_in[:, :k], trunk_in[:, k:k + 1],
                       trunk_in[:, k + 1:k + 2], trunk_in[:, k + 2:k + 3])
        return self.stepper(z, theta, t, s, ds)


class Cmp3dLatentARAdapter:
    def build(self, cfg: dict, seed: int) -> dict:
        set_seed(seed)
        m = cfg["model"]; dev = _device(m)
        enc_ref = cfg["cmp3d_encoder_ref"]
        p = Path(enc_ref); p = p if p.is_absolute() else REPO / p
        if not p.exists():
            raise FileNotFoundError(f"cmp3d_encoder_ref missing: {p}")
        encoder = _Pod3dEncoder(p, k=cfg.get("pod_k"))
        k = encoder.latent_dim
        dyn_kind = str(cfg.get("dynamics_kind", "mamba_branch_don"))
        dyn = build_dynamics(dyn_kind, latent_dim=k, theta_dim=6, hp=m)
        if dyn_kind == "ar_mlp":
            dyn = _ARMlpBranchTrunkShim(dyn, k)
        return {"encoder": encoder, "dyn": dyn, "device": dev, "latent_dim": k,
                "dynamics_kind": dyn_kind, "n_params": sum(p.numel() for p in dyn.parameters()),
                "run_meta": {"encoder_ref": enc_ref, "dynamics": dyn_kind, "latent_dim": k}}

    def fit(self, ms, train, val, cfg, seed):
        set_seed(seed)
        m = cfg["model"]; dev = ms["device"]; encoder = ms["encoder"]
        dyn = ms["dyn"].to(dev)
        z_tr = _encode_states(encoder, train)
        N, T, d = z_tr.shape
        zf = z_tr.reshape(-1, d)
        z_mean = zf.mean(0).astype(np.float32); z_std = (zf.std(0) + 1e-8).astype(np.float32)
        z_tr_n = (z_tr - z_mean) / z_std
        th_raw = np.asarray(train.theta, np.float32)
        th_mean = th_raw.mean(0); th_std = th_raw.std(0) + 1e-8
        th_n = (th_raw - th_mean) / th_std
        ctrl = np.asarray(train.controls(train.case_ids), np.float32)

        z_t = torch.tensor(z_tr_n, device=dev)
        th_t = torch.tensor(th_n, device=dev)
        ctrl_t = torch.tensor(ctrl, device=dev)

        n_epochs = int(m["epochs"])
        rollout_max = int(m.get("rollout_max_phase1", 50))
        warmup = int(m.get("rollout_warmup_phase1", 300))
        ramp_ep = int(m.get("rollout_ramp_phase1", 300))
        var_lambda = float(m.get("var_lambda_phase1", 0.3))
        batch_cases = int(m.get("batch_size", 64))
        lr = float(m.get("lr", 2.5e-4)); wd = float(m.get("weight_decay", 1e-4))
        grad_clip = float(m.get("grad_clip_phase1", 0.25))
        best_mature = int(m.get("best_mature_epoch_phase1", 700))
        dt = float(np.asarray(train.t_grid)[1] - np.asarray(train.t_grid)[0]) if T > 1 else 1.0

        ck_dir = Path(cfg.get("ckpt_dir", REPO / "results" / "benchmark" / "runs" /
                              f"{cfg['model_name']}_seed{seed}_ckpts"))
        ck_dir.mkdir(parents=True, exist_ok=True)
        opt = torch.optim.AdamW(dyn.parameters(), lr=lr, weight_decay=wd)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(n_epochs, 1), eta_min=1e-6)
        rng = np.random.RandomState(seed)
        ramp_seqs = {i: ctrl_t[i].unsqueeze(-1) for i in range(N)}
        best = float("inf"); best_state = None; best_ep = -1
        hist = {"train": [], "val": []}

        # ---------- Validation-based checkpoint selection ----------
        # best.pt is selected on a full-horizon validation rollout MSE, not on the
        # train loss. The val pass is torch-only, runs under no_grad and never
        # touches the local `rng`, so it leaves the training trajectory unchanged.
        z_va = _encode_states(encoder, val)
        z_va_t = torch.tensor((z_va - z_mean) / z_std, device=dev)
        th_va_t = torch.tensor((np.asarray(val.theta, np.float32) - th_mean) / th_std, device=dev)
        ctrl_va_t = torch.tensor(np.asarray(val.controls(val.case_ids), np.float32), device=dev)
        ramp_va = ctrl_va_t.unsqueeze(-1)
        val_every = int(m.get("val_every", 25))
        best_val = float("inf"); last_val = float("nan")

        def _val_rollout_loss() -> float:
            dyn.eval()
            with torch.no_grad():
                bf_va = (th_va_t, ramp_va)
                z_c = z_va_t[:, 0]; preds = []
                for tk in range(T - 1):
                    t_f = torch.full((z_c.shape[0], 1), tk / T, dtype=torch.float32, device=dev)
                    s_f = ctrl_va_t[:, tk].unsqueeze(1)
                    ds_f = ((ctrl_va_t[:, min(tk + 1, T - 1)] - ctrl_va_t[:, max(tk - 1, 0)]) / (2 * dt)).unsqueeze(1)
                    z_c = z_c + dyn(bf_va, torch.cat([z_c, t_f, s_f, ds_f], 1))
                    preds.append(z_c)
                vl = F.mse_loss(torch.stack(preds, 1), z_va_t[:, 1:]).item()
            dyn.train()
            return float(vl)

        t0 = time.time()
        for epoch in range(1, n_epochs + 1):
            dyn.train()
            rk = (1 if epoch < warmup else
                  max(1, int(1 + min(1.0, (epoch - warmup) / ramp_ep) * (rollout_max - 1))))
            cases = rng.permutation(N)[:batch_cases]
            max_start = max(2, T - rk - 2)
            ts = np.clip(rng.randint(1, max_start, size=len(cases)), 0, T - rk - 2)
            bf = (th_t[cases], torch.stack([ramp_seqs[c] for c in cases]))
            z_true_t = z_t[cases, ts]
            t_f = torch.tensor(ts / T, dtype=torch.float32, device=dev).unsqueeze(1)
            s_f = ctrl_t[cases, ts].unsqueeze(1)
            ds_f = torch.stack([(ctrl_t[c, min(t + 1, T - 1)] - ctrl_t[c, max(t - 1, 0)]) / (2 * dt)
                                for c, t in zip(cases, ts)]).unsqueeze(1)
            true_dz = z_t[cases, ts + 1] - z_t[cases, ts]
            total = F.mse_loss(dyn(bf, torch.cat([z_true_t, t_f, s_f, ds_f], 1)), true_dz)
            if rk > 1:
                z_c = z_t[cases, ts]; ps = [z_c]; tss = [z_c.detach()]
                for kstep in range(min(rk, T - ts.max() - 2)):
                    tk = ts + kstep; tk_safe = np.clip(tk, 0, T - 2)
                    t_f = torch.tensor(tk_safe / T, dtype=torch.float32, device=dev).unsqueeze(1)
                    s_f = ctrl_t[cases, tk_safe].unsqueeze(1)
                    ds_f = torch.stack([(ctrl_t[c, min(t + 1, T - 1)] - ctrl_t[c, max(t - 1, 0)]) / (2 * dt)
                                        for c, t in zip(cases, tk_safe)]).unsqueeze(1)
                    z_c = z_c + dyn(bf, torch.cat([z_c, t_f, s_f, ds_f], 1))
                    ps.append(z_c); tss.append(z_t[cases, np.clip(tk + 1, 0, T - 1)])
                ps_s = torch.stack(ps, 1); ts_s = torch.stack(tss, 1)
                total = total + F.mse_loss(ps_s, ts_s)
                if rk >= 5:
                    total = total + var_lambda * F.mse_loss(ps_s.var(1).mean(), ts_s.var(1).mean())
            opt.zero_grad(set_to_none=True); total.backward()
            torch.nn.utils.clip_grad_norm_(dyn.parameters(), grad_clip)
            opt.step(); sch.step()
            lv = float(total.item()); hist["train"].append(lv)
            if epoch == 1:
                print(f"[cmp3d_ar] ep1: {time.time()-t0:.1f}s -> est {(time.time()-t0)*n_epochs/60:.0f} min", flush=True)
            if epoch % val_every == 0 or epoch == n_epochs:
                last_val = _val_rollout_loss()
                if epoch >= best_mature and last_val < best_val:
                    best_val = last_val; best_ep = epoch; best_state = copy.deepcopy(dyn.state_dict())
                    torch.save({"epoch": epoch, "state_dict": best_state,
                                "loss": best_val, "val_loss": best_val}, ck_dir / "best.pt")
            hist["val"].append(last_val)
            if epoch % 50 == 0 or epoch == n_epochs:
                print(f"[cmp3d_ar] ep{epoch} loss={lv:.5f} val={last_val:.5f} best_val={best_val:.5f}@{best_ep} rk={rk}", flush=True)
        torch.save({"epoch": n_epochs, "state_dict": dyn.state_dict()}, ck_dir / "last.pt")
        if best_state is None:
            best_val = _val_rollout_loss(); best_ep = n_epochs
            torch.save({"epoch": n_epochs, "state_dict": dyn.state_dict(),
                        "loss": best_val, "val_loss": best_val}, ck_dir / "best.pt")
        else:
            dyn.load_state_dict(best_state)
        ms.update({"dyn": dyn, "z_mean_lat": torch.tensor(z_mean), "z_std_lat": torch.tensor(z_std),
                   "th_mean": torch.tensor(th_mean), "th_std": torch.tensor(th_std),
                   "run_meta": {**ms.get("run_meta", {}), "best_epoch": best_ep},
                   "best_epoch": best_ep})
        ms["run_meta"]["total_epochs"] = n_epochs
        return ms, hist

    def predict_eval(self, ms, test, cfg, **kw):
        dev = ms["device"]; encoder = ms["encoder"]; dyn = ms["dyn"].to(dev).eval()
        z_mean = ms["z_mean_lat"].to(dev); z_std = ms["z_std_lat"].to(dev)
        ids = test.case_ids; n = len(ids)
        st = np.asarray(test.states(ids), np.float32)
        phys = st * test.stats_std + test.stats_mean
        T = st.shape[1]
        tg = np.asarray(test.t_grid, np.float64); dt = float(tg[1] - tg[0]) if T > 1 else 1.0
        th = (torch.tensor(np.asarray(test.theta, np.float32), device=dev) - ms["th_mean"].to(dev)) / ms["th_std"].to(dev)
        s_np = np.asarray(test.controls(ids), np.float32)
        s_t = torch.tensor(s_np, device=dev); ds_t = torch.zeros_like(s_t)
        if T >= 3:
            ds_t[:, 1:-1] = (s_t[:, 2:] - s_t[:, :-2]) / (2 * dt)
            ds_t[:, 0] = (s_t[:, 1] - s_t[:, 0]) / dt; ds_t[:, -1] = (s_t[:, -1] - s_t[:, -2]) / dt
        bf = (th, s_t.unsqueeze(-1))
        z0 = np.stack([encoder.encode(phys[i, 0:1])[0] for i in range(n)], 0)
        with torch.no_grad():
            z_ar = (torch.tensor(z0, device=dev) - z_mean) / z_std
            preds = [z_ar]
            for t_step in range(T - 1):
                t_f = torch.full((n, 1), t_step / T, dtype=torch.float32, device=dev)
                trunk = torch.cat([z_ar, t_f, s_t[:, t_step].unsqueeze(1), ds_t[:, t_step].unsqueeze(1)], 1)
                z_ar = z_ar + dyn(bf, trunk); preds.append(z_ar)
            z_pred = (torch.stack(preds, 1) * z_std + z_mean).cpu().numpy()
        Node = phys.shape[2]
        vel = np.empty((n, T, Node, 3), np.float32)
        for ci in range(n):
            vel[ci] = encoder.decode(z_pred[ci])
        return vel
