# benchmark/adapters/branch_don_training.py
# ============================================================================
# Purpose: the training loop of the three KVS latent x AR rows with a recurrent
#   branch (S4D, Mamba and GRU branch DeepONet). The model predicts the latent
#   increment dz from a branch over (theta, s(t)) and a trunk over
#   [z, t/T, s, ds]. Each epoch draws a batch of training cases and combines a
#   one-step MSE on dz with a curriculum rollout MSE, whose horizon grows from 1
#   to rollout_max after a warmup, plus a variance-matching penalty once the
#   horizon reaches five steps.
# Pipeline role: called by benchmark/adapters/latent_dynamics.py, which encodes
#   the trajectories with encode_all, normalises latents and theta on the train
#   split, and hands the tensors to train_phase1. The best checkpoint is written
#   to <ckpt_dir>/best.pt and reloaded by the adapter for inference.
# Eval-consistency: the best checkpoint is selected on a full-horizon
#   validation rollout MSE, evaluated every val_every epochs and only from
#   best_mature_epoch on. The validation pass runs under no_grad and draws no
#   random numbers, so it does not change the training trajectory. The
#   trajectory itself is fixed by set_seed(seed) at the start and by the order
#   of the numpy draws for the case batch and the rollout start offsets.
# ============================================================================
"""Training loop of the branch-DeepONet latent dynamics (S4D, Mamba, GRU branch)."""
from __future__ import annotations

import copy
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from benchmark.adapters._common import grid_to_channels_first
from tools.common.seeding import set_seed

# Snapshot spacing of the KVS trajectories (201 snapshots over 4 s).
DT = 0.02


def ds_fd(ctrl, t: int, T: int):
    """Finite difference of the control at snapshot t: central inside, one-sided at the ends."""
    hi, lo = min(t + 1, T - 1), max(t - 1, 0)
    return (ctrl[..., hi] - ctrl[..., lo]) / ((hi - lo) * DT)


def log(msg: str) -> None:
    print(f"{datetime.now().isoformat(timespec='seconds')} {msg}", flush=True)


def encode_all(ae, frames: np.ndarray, x_mean: torch.Tensor, x_std: torch.Tensor,
               dev: torch.device, latent_dim: int) -> np.ndarray:
    """frames (N, T, H, W, 3) physical fp16 -> z (N, T, d) fp32.

    Encodes case by case on the device; applies (x - x_mean) / x_std before the encoder.
    """
    N, T, _, _, _ = frames.shape
    x = torch.from_numpy(grid_to_channels_first(frames))           # (N, T, 3, H, W) fp16
    out = np.empty((N, T, latent_dim), dtype=np.float32)
    with torch.no_grad():
        for i in range(N):
            xc = x[i].to(dev, non_blocking=True).float()            # (T, 3, H, W)
            xc_norm = (xc - x_mean) / x_std
            out[i] = ae.encode(xc_norm).cpu().numpy()
    return out


def train_phase1(model, data: dict, n_epochs: int, rollout_max: int, warmup: int, ramp_ep: int,
                 var_lambda: float, batch_cases: int, lr: float, weight_decay: float,
                 grad_clip: float, best_mature_epoch: int, dev: torch.device, ckpt_dir: Path,
                 seed: int, val_every: int = 25):
    """Train `model` on the normalised latents in `data`; returns (best_val, train_time).

    `data` holds z_tr_n / z_va_n (N, T, d), th_tr / th_va (N, theta_dim), ctrl_tr / ctrl_va
    (N, T) on the device, and the trajectory length T.
    """
    set_seed(seed)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    z_gpu = data["z_tr_n"]
    th_gpu = data["th_tr"]
    ctrl_gpu = data["ctrl_tr"]
    z_va_n = data["z_va_n"]
    th_va = data["th_va"]
    ctrl_va = data["ctrl_va"]
    T = data["T"]
    n_train = z_gpu.shape[0]
    train_idx = np.arange(n_train)

    # Ramp sequence of each training case as a (T, 1) tensor.
    ramp_seqs = {i: ctrl_gpu[i].unsqueeze(-1) for i in range(n_train)}

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs, eta_min=1e-6)

    best_state: dict | None = None
    best_val = float("inf")
    best_val_ep = -1
    last_val = float("nan")
    t_start = time.time()

    def _val_rollout_loss() -> float:
        """Full-horizon autoregressive rollout MSE on the validation latents (no grad)."""
        model.eval()
        with torch.no_grad():
            bf_va = (th_va, ctrl_va.unsqueeze(-1))
            z_c = z_va_n[:, 0]
            preds = []
            for tk in range(T - 1):
                t_f = torch.full((z_c.shape[0], 1), tk / T, dtype=torch.float32, device=dev)
                s_f = ctrl_va[:, tk].unsqueeze(1)
                ds_f = ds_fd(ctrl_va, tk, T).unsqueeze(1)
                z_c = z_c + model(bf_va, torch.cat([z_c, t_f, s_f, ds_f], dim=1))
                preds.append(z_c)
            vl = F.mse_loss(torch.stack(preds, dim=1), z_va_n[:, 1:]).item()
        model.train()
        return float(vl)

    for epoch in range(1, n_epochs + 1):
        model.train()
        rk = 1 if epoch < warmup else max(1, int(1 + min(1.0, (epoch - warmup) / ramp_ep) * (rollout_max - 1)))
        cases = train_idx[np.random.permutation(n_train)[:batch_cases]]
        max_start = max(2, T - rk - 2)
        ts = np.random.randint(1, max_start, size=len(cases))
        ts = np.clip(ts, 0, T - rk - 2)
        total = torch.tensor(0.0, device=dev)

        bf = (th_gpu[cases], torch.stack([ramp_seqs[c] for c in cases]))

        # One-step loss on dz.
        z_true_t = z_gpu[cases, ts]
        t_f = torch.tensor(ts / T, dtype=torch.float32, device=dev).unsqueeze(1)
        s_f = ctrl_gpu[cases, ts].unsqueeze(1)
        ds_f = torch.stack([ds_fd(ctrl_gpu[c], t, T) for c, t in zip(cases, ts)]).unsqueeze(1)
        true_dz = z_gpu[cases, ts + 1] - z_gpu[cases, ts]
        dz_pred_1step = model(bf, torch.cat([z_true_t, t_f, s_f, ds_f], dim=1))
        total = total + F.mse_loss(dz_pred_1step, true_dz)

        # Rollout loss over the current curriculum horizon, plus the variance penalty.
        if rk > 1:
            z_c = z_gpu[cases, ts]
            ps = [z_c]
            tss = [z_c.detach()]
            for k in range(min(rk, T - ts.max() - 2)):
                tk = ts + k
                tk_safe = np.clip(tk, 0, T - 2)
                t_f = torch.tensor(tk_safe / T, dtype=torch.float32, device=dev).unsqueeze(1)
                s_f = ctrl_gpu[cases, tk_safe].unsqueeze(1)
                ds_f = torch.stack([ds_fd(ctrl_gpu[c], t, T) for c, t in zip(cases, tk_safe)]).unsqueeze(1)
                z_c = z_c + model(bf, torch.cat([z_c, t_f, s_f, ds_f], dim=1))
                ps.append(z_c)
                tss.append(z_gpu[cases, np.clip(tk + 1, 0, T - 1)])
            ps_s = torch.stack(ps, 1)
            ts_s = torch.stack(tss, 1)
            total = total + F.mse_loss(ps_s, ts_s)
            if rk >= 5:
                total = total + var_lambda * F.mse_loss(
                    ps_s.var(dim=1).mean(), ts_s.var(dim=1).mean())

        opt.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        sch.step()

        loss_val = float(total.item())
        if epoch % val_every == 0 or epoch == n_epochs:
            last_val = _val_rollout_loss()
            if epoch >= best_mature_epoch and last_val < best_val:
                best_val = last_val
                best_val_ep = epoch
                best_state = copy.deepcopy(model.state_dict())
                torch.save({"epoch": epoch, "state_dict": best_state,
                            "loss": best_val, "val_loss": best_val},
                           ckpt_dir / "best.pt")

        if epoch == 1:
            log(f"  [phase1] ep1: {time.time() - t_start:.1f}s (no ETA, the rollout grows over time)")
        if epoch % 10 == 0:
            _disp_best = best_val if best_state else float("nan")
            log(f"  [phase1] ep {epoch}/{n_epochs} k={rk} loss={loss_val:.6f} "
                f"val={_disp_best:.6f} last_val={last_val:.6f} "
                f"lr={sch.get_last_lr()[0]:.2e} [{time.time() - t_start:.0f}s]")

    train_time = time.time() - t_start
    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        # Never reached the selection window: keep the final weights.
        best_val = _val_rollout_loss()
        torch.save({"epoch": n_epochs, "state_dict": model.state_dict(), "loss": best_val},
                   ckpt_dir / "best.pt")
    log(f"[phase1] done in {train_time:.0f}s, best_val={best_val:.6f} @ep{best_val_ep}")
    return best_val, train_time
