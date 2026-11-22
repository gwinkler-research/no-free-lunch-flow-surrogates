# benchmark/adapters/latent_dynamics.py
# ============================================================================
# Purpose: the latent x AR adapter. It loads the frozen ConvAE named by
#   cfg["encoder_ref"] (requires_grad False, eval), builds the dz-stepper chosen
#   by cfg["dynamics"], encodes the rasterised trajectories to z-sequences,
#   integrates z(t+1) = z(t) + dz autoregressively from the encoded t=0 frame,
#   and decodes back to the field. Four dynamics kinds are dispatched: ar_mlp,
#   s4d_branch_don, mamba_branch_don and gru_branch_don; anything else raises.
#   AR-MLP is trained here (_fit_ar_mlp); the three branch DeepONets are trained
#   by branch_don_training.train_phase1.
# Pipeline role: the middle stage of the latent chain (frozen encoder -> this
#   adapter -> predictions.npz -> eval_harness). It produces the S4D, Mamba and
#   GRU branch DeepONet rows and the AR-MLP row of the KVS table.
#   _load_frozen_encoder and _encode_traj are reused by latent_oneshot, dmdc
#   and several scripts.
# Eval-consistency: the two fit paths normalise differently. ar_mlp uses the
#   z_mean/z_std stored in the frozen encoder file and feeds theta
#   unnormalised. The branch-DeepONet path computes latent mean/std and theta
#   mean/std from the TRAIN split and keeps them in the run state; its predict
#   path uses them and falls back to the encoder's latent stats only if they
#   are missing. Train, val and test come from the runner's frozen views.
#   set_seed(seed) runs in build() and in both fit paths; a separate
#   default_rng(seed) drives the ar_mlp case permutation and rollout offsets.
#   Both paths select best.pt on a full-horizon validation rollout MSE, and only
#   from a maturity epoch on (best_mature_epoch for ar_mlp,
#   best_mature_epoch_phase1 for the branch DeepONets).
#   ds/dt input: ar_mlp uses the analytic ramp slope from ramp_s_dsdt. The branch
#   DeepONets use central differences of s(t), one-sided at the ends, with
#   dt = 0.02 in training (branch_don_training.DT) and dt from t_grid (0.02 for
#   KVS) at inference.
#   Checkpoints go to results/benchmark/runs/<model_name>_seed<seed>_ckpts/
#   (the split is not part of the directory name).
# ============================================================================
"""latent_dynamics adapter: latent x AR rows, parametrised by cfg["dynamics"].

build:        load the frozen grid_convae checkpoint (cfg["encoder_ref"]); rebuild
              the ConvAutoencoder and freeze it (requires_grad=False, eval). Build
              the dynamics model via _dynamics_models.build_dynamics.
fit:          encode the rasterised TRAIN/VAL trajectories with the frozen encoder
              -> z-sequences. Then dispatch by `dynamics`:
                ar_mlp         -> _fit_ar_mlp: latent stats from the encoder file;
                                  rollout curriculum k=1..rollout_max (warmup
                                  rollout_warmup_epochs at k=1, then ramp over
                                  rollout_ramp_epochs); loss = rollout MSE + lambda_var
                                  * max(0, var_true - var_pred); batched over cases;
                                  early stopping after early_stop_patience epochs
                                  without improvement (from epoch 100 on).
                s4d/mamba/gru_branch_don
                               -> _fit_s4d_branch_don: latent and theta stats from
                                  TRAIN; branch_don_training.train_phase1 (one-step
                                  dz MSE + curriculum rollout MSE + variance penalty
                                  once the horizon reaches five steps, AdamW +
                                  CosineAnnealingLR).
              Saves best + last checkpoints.
predict_eval: encode the t=0 frame of each test case and roll out T-1 steps;
              ar_mlp steps with (z, theta, t, s, ds), the branch DeepONets with
              trunk_in = [z, t/T, s, ds] and branch input (theta, s(t)).
              Decode -> grid -> grid_to_nodes -> arrays.

Note on T: The kvs/field schema (validated by benchmark.prediction_io.validate)
requires exactly T=201 timesteps and t_grid.shape==(201,). The KVS cases have 201
saved steps (checked in tools.data.karman_h5_dataset), so no resampling is needed.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from benchmark.adapters import BenchmarkAdapter, register_adapter
from benchmark.adapters._common import (EpochLogger, assemble_kvs_field_arrays,
                                        channels_first_to_grid, control_features,
                                        grid_to_channels_first, save_ckpt)
from benchmark.adapters._dynamics_models import (ARMlpStepper, S4DBranchDeepONet,
                                                  build_dynamics)
from benchmark.adapters.branch_don_training import encode_all, train_phase1
from tools.common.seeding import set_seed
from tools.models.conv_autoencoder import ConvAutoencoder

REPO = Path(__file__).resolve().parents[2]


def _device(cfg_model: dict) -> torch.device:
    d = str(cfg_model.get("device", "cpu"))
    if d == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(d)


def _load_frozen_encoder(enc_ref: str, dev: torch.device):
    """Load a grid_convae encoder file and freeze it.

    Returns (ae, (H, W), latent_dim, z_mean, z_std, x_mean, x_std); x_mean/x_std are
    (1,3,1,1) tensors on `dev` (zeros/ones if the file has none)."""
    p = Path(enc_ref)
    if not p.is_absolute():
        p = REPO / p
    ck = torch.load(p, map_location="cpu", weights_only=False)
    H, W = int(ck["grid_hw"][0]), int(ck["grid_hw"][1])
    ae = ConvAutoencoder(in_channels=3, latent_dim=int(ck["latent_dim"]),
                         base_channels=int(ck["base_channels"]), n_downsample=int(ck["n_downsample"]),
                         grid_h=H, grid_w=W)
    ae.load_state_dict(ck["model_state_dict"])
    ae.to(dev).eval()
    for prm in ae.parameters():
        prm.requires_grad_(False)
    z_mean = torch.tensor(np.asarray(ck["z_mean"]), dtype=torch.float32, device=dev)
    z_std = torch.tensor(np.asarray(ck["z_std"]), dtype=torch.float32, device=dev)
    x_mean_arr = np.asarray(ck.get("x_mean", np.zeros(3, dtype=np.float32)))
    x_std_arr = np.asarray(ck.get("x_std", np.ones(3, dtype=np.float32)))
    x_mean = torch.tensor(x_mean_arr, dtype=torch.float32, device=dev).reshape(1, 3, 1, 1)
    x_std = torch.tensor(x_std_arr, dtype=torch.float32, device=dev).reshape(1, 3, 1, 1)
    return ae, (H, W), int(ck["latent_dim"]), z_mean, z_std, x_mean, x_std


def _encode_traj(ae: ConvAutoencoder, frames: np.ndarray, dev: torch.device,
                 x_mean: torch.Tensor, x_std: torch.Tensor) -> np.ndarray:
    """frames (N, T, H, W, 3) PHYSICAL -> z (N, T, d) using the frozen encoder.

    Applies per-channel input normalisation (x_mean, x_std are (1,3,1,1) tensors
    on `dev`). Keeps frames at their incoming dtype (fp16 if passed) to limit peak host RAM;
    casts each case to fp32 on the device before encoding."""
    N, T, H, W, _ = frames.shape
    x = torch.from_numpy(grid_to_channels_first(frames))    # (N, T, 3, H, W), incoming dtype
    out = np.empty((N, T, ae.latent_dim), dtype=np.float32)
    with torch.no_grad():
        for i in range(N):
            xc = x[i].to(dev, non_blocking=True).float()    # fp32 cast on the device, not the host
            xc_norm = (xc - x_mean) / x_std
            out[i] = ae.encode(xc_norm).cpu().numpy()
    return out


def _control_seq_stack(theta_arr: np.ndarray, t_grid: np.ndarray, case_ids):
    """(N, T) stacks of the ramp s(t) and ds/dt reconstructed from theta."""
    feats = [control_features(theta_arr[i], t_grid) for i in range(len(case_ids))]
    f = np.stack([a for a, _ in feats], axis=0).astype(np.float32)
    fd = np.stack([b for _, b in feats], axis=0).astype(np.float32)
    return f, fd


def _theta_inputs(theta_np: np.ndarray, dev) -> torch.Tensor:
    """theta model input: the full theta rows as a float tensor on `dev`."""
    return torch.from_numpy(theta_np).float().to(dev)


def _encode_split_streaming(ae: ConvAutoencoder, grid, case_ids, dev: torch.device,
                            x_mean: torch.Tensor, x_std: torch.Tensor) -> np.ndarray:
    """Encode case by case straight from the memory-mapped raster cache -> z (N, T, d) float32.

    Holds the frames of one case in host RAM (~42 MB) instead of the whole split
    (~27 GB fp16 for the 644 interp training cases). The result is the same as
    encoding the whole split at once.
    """
    out = None
    for i, cid in enumerate(case_ids):
        frames = grid.frames_for([cid], dtype=np.float16)          # (1, T, H, W, 3)
        z = _encode_traj(ae, frames, dev, x_mean, x_std)[0]        # (T, d)
        if out is None:
            out = np.empty((len(case_ids),) + z.shape, dtype=np.float32)
        out[i] = z
    return out


def _rollout_k(ms_seq, ds_seq, t_norm, model: ARMlpStepper, z0, theta, k: int):
    """AR rollout: z[0]=z0, z[t+1] = z[t] + model(z[t], theta, t, s, ds), for k steps.
    Returns (B, k+1, d)."""
    preds = [z0]
    z = z0
    for step in range(k):
        z = z + model(z, theta, t=t_norm[:, step:step + 1], s=ms_seq[:, step:step + 1],
                      ds=ds_seq[:, step:step + 1])
        preds.append(z)
    return torch.stack(preds, dim=1)


@register_adapter("latent_dynamics")
class LatentDynamicsAdapter(BenchmarkAdapter):

    def build(self, cfg: dict, seed: int) -> dict:
        set_seed(seed)
        m = cfg["model"]
        dev = _device(m)
        ae, (H, W), latent_dim, z_mean, z_std, x_mean, x_std = _load_frozen_encoder(cfg["encoder_ref"], dev)
        dyn_kind = str(cfg["dynamics"])
        dyn = build_dynamics(dyn_kind, latent_dim=latent_dim, theta_dim=6, hp=m)
        n_params = sum(p.numel() for p in dyn.parameters())
        return {"ae": ae, "dyn": dyn, "device": dev, "grid_hw": [H, W],
                "latent_dim": latent_dim, "z_mean": z_mean, "z_std": z_std,
                "x_mean": x_mean, "x_std": x_std, "n_params": int(n_params),
                "run_meta": {"encoder_ref": cfg["encoder_ref"], "dynamics": dyn_kind,
                             "latent_dim": latent_dim}}

    def fit(self, ms: dict, train, val, cfg: dict, seed: int):
        dyn_kind = str(cfg["dynamics"])
        if dyn_kind == "ar_mlp":
            return self._fit_ar_mlp(ms, train, val, cfg, seed)
        if dyn_kind in ("s4d_branch_don", "mamba_branch_don", "gru_branch_don"):
            # S4D/Mamba/GRU branch DONs share the trunk/branch interface and the
            # training loop in branch_don_training.py.
            return self._fit_s4d_branch_don(ms, train, val, cfg, seed)
        raise ValueError(f"latent_dynamics: unknown dynamics {dyn_kind!r}")

    def _fit_ar_mlp(self, ms: dict, train, val, cfg: dict, seed: int):
        set_seed(seed)
        m = cfg["model"]
        dev = ms["device"]
        ae: ConvAutoencoder = ms["ae"]
        dyn = ms["dyn"].to(dev)
        z_mean, z_std = ms["z_mean"], ms["z_std"]
        x_mean, x_std = ms["x_mean"], ms["x_std"]
        n_epochs = int(m["epochs"])
        log = EpochLogger("latent_dynamics:ar_mlp", n_epochs)
        z_tr = (_encode_split_streaming(ae, train.grid, train.case_ids, dev, x_mean, x_std)
                - z_mean.cpu().numpy()) / z_std.cpu().numpy()
        z_va = (_encode_split_streaming(ae, val.grid, val.case_ids, dev, x_mean, x_std)
                - z_mean.cpu().numpy()) / z_std.cpu().numpy()
        N, T, d = z_tr.shape
        th_tr = _theta_inputs(train.theta, dev)
        th_va = _theta_inputs(val.theta, dev)
        t_grid = np.asarray(train.t_grid, dtype=np.float64)
        s_tr, ds_tr = _control_seq_stack(train.theta, t_grid, train.case_ids)
        s_va, ds_va = _control_seq_stack(val.theta, t_grid, val.case_ids)
        t_norm = torch.tensor((t_grid / max(float(t_grid[-1]), 1e-9)).astype(np.float32), device=dev)
        z_tr_t = torch.from_numpy(z_tr).float().to(dev)
        z_va_t = torch.from_numpy(z_va).float().to(dev)
        s_tr_t = torch.from_numpy(s_tr).to(dev)
        ds_tr_t = torch.from_numpy(ds_tr).to(dev)
        s_va_t = torch.from_numpy(s_va).to(dev)
        ds_va_t = torch.from_numpy(ds_va).to(dev)
        bs = int(m["batch_size"])
        val_every = int(m.get("val_every", 25))
        patience = int(m.get("early_stop_patience", 100))
        # Counterpart of best_mature_epoch_phase1 in the branch-DeepONet path: before
        # this epoch neither best.pt nor the patience counter is updated. During the
        # curriculum ramp the full-rollout val loss is not comparable (models trained
        # on short horizons give spurious, mean-regression-like values that later
        # epochs would first have to beat).
        mature_ep = int(m.get("best_mature_epoch", 0))
        opt = torch.optim.AdamW(dyn.parameters(), lr=float(m["lr"]),
                                weight_decay=float(m.get("weight_decay", 1e-6)))
        warmup = int(m.get("rollout_warmup_epochs", 0))
        ramp_ep = int(m.get("rollout_ramp_epochs", 0))
        k_max = int(m.get("rollout_max", 2))
        lam_var = float(m.get("lambda_var", 0.3))
        rng = np.random.default_rng(seed)
        best_val, best_ep, no_imp = float("inf"), -1, 0
        hist = {"train": [], "val": []}
        ck_dir = REPO / "results" / "benchmark" / "runs" / f"{cfg['model_name']}_seed{seed}_ckpts"
        best_ckpt = ck_dir / "best.pt"
        last_ckpt = ck_dir / "last.pt"

        def _curr_k(epoch: int) -> int:
            if epoch <= warmup or ramp_ep <= 0:
                return 1
            frac = min(1.0, (epoch - warmup) / ramp_ep)
            return max(1, int(1 + frac * (k_max - 1)))

        for epoch in range(1, n_epochs + 1):
            log.epoch_start()
            dyn.train()
            perm = rng.permutation(N)
            ep_loss, nb = 0.0, 0
            k = _curr_k(epoch)
            for bi, i in enumerate(range(0, N, bs)):
                idx = perm[i:i + bs]
                z_b = z_tr_t[idx]
                th_b = th_tr[idx]
                if k + 1 < T:
                    start = int(rng.integers(0, T - k))
                else:
                    start = 0
                z0 = z_b[:, start]
                seg = z_b[:, start:start + k + 1]
                s_seg = s_tr_t[idx, start:start + k + 1]
                ds_seg = ds_tr_t[idx, start:start + k + 1]
                t_seg = t_norm[start:start + k + 1].unsqueeze(0).expand(len(idx), -1)
                pred = _rollout_k(s_seg, ds_seg, t_seg, dyn, z0, th_b, k)
                mse = F.mse_loss(pred, seg)
                var_pred = pred.var(dim=1).mean()
                var_true = seg.var(dim=1).mean()
                loss = mse + lam_var * (var_true - var_pred).clamp(min=0.0)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(dyn.parameters(), 1.0)
                opt.step()
                ep_loss += float(loss.item())
                nb += 1
            tr = ep_loss / max(nb, 1)
            vl = float("nan")
            if epoch % val_every == 0 or epoch == n_epochs:
                dyn.eval()
                with torch.no_grad():
                    z0 = z_va_t[:, 0]
                    t_full = t_norm.unsqueeze(0).expand(z_va_t.shape[0], -1)
                    pred = _rollout_k(s_va_t, ds_va_t, t_full, dyn, z0, th_va, T - 1)
                    vl = float(F.mse_loss(pred, z_va_t).item())
                if epoch < mature_ep:
                    pass  # before the maturity epoch: neither best.pt nor patience
                elif vl < best_val:
                    best_val, best_ep, no_imp = vl, epoch, 0
                    save_ckpt(ck_dir, best_ckpt.name, {"epoch": epoch, "dyn_state_dict": dyn.state_dict()})
                    log.saved(best_ckpt.name, vl, epoch)
                else:
                    no_imp += val_every
            hist["train"].append(tr)
            hist["val"].append(vl)
            log.epoch_end(epoch, tr, vl if vl == vl else tr,
                          best_val if best_val < float("inf") else tr, best_ep)
            if no_imp >= patience and epoch > 100:
                print(f"[latent_dynamics] early stop at epoch {epoch}", flush=True)
                break
        save_ckpt(ck_dir, last_ckpt.name, {"epoch": epoch, "dyn_state_dict": dyn.state_dict()})
        if best_ckpt.exists():
            dyn.load_state_dict(torch.load(best_ckpt, map_location=dev, weights_only=False)["dyn_state_dict"])
        ms.update({"dyn": dyn.cpu(), "best_ckpt_path": str(best_ckpt) if best_ckpt.exists() else None,
                   "last_ckpt_path": str(last_ckpt), "best_epoch": best_ep,
                   "run_meta": {**ms.get("run_meta", {}), "best_epoch": best_ep, "best_val": best_val}})
        return ms, hist

    def _fit_s4d_branch_don(self, ms: dict, train, val, cfg: dict, seed: int):
        """Encode, normalise on the train split, and train with branch_don_training.train_phase1."""
        set_seed(seed)
        m = cfg["model"]
        dev = ms["device"]
        ae: ConvAutoencoder = ms["ae"]
        dyn = ms["dyn"].to(dev)
        x_mean, x_std = ms["x_mean"], ms["x_std"]

        # Encode case by case straight from the raster cache (bounded host RAM).
        def _enc_stream(case_ids, grid):
            out = None
            for i, cid in enumerate(case_ids):
                fr = grid.frames_for([cid], dtype=np.float16)          # (1, T, H, W, 3)
                z1 = encode_all(ae, fr, x_mean, x_std, dev, ms["latent_dim"])[0]
                if out is None:
                    out = np.empty((len(case_ids),) + z1.shape, dtype=np.float32)
                out[i] = z1
            return out

        z_tr = _enc_stream(train.case_ids, train.grid)
        z_va = _enc_stream(val.case_ids, val.grid)
        N, T, d = z_tr.shape

        # Latent normalisation: per-dimension mean/std of the TRAIN latents
        zf = z_tr.reshape(-1, d)
        z_mean_lat, z_std_lat = zf.mean(0), zf.std(0) + 1e-8
        z_tr_n = (z_tr - z_mean_lat) / z_std_lat
        z_va_n = (z_va - z_mean_lat) / z_std_lat
        z_mean_lat_t = torch.tensor(z_mean_lat, dtype=torch.float32, device=dev)
        z_std_lat_t = torch.tensor(z_std_lat, dtype=torch.float32, device=dev)

        # Theta normalisation from TRAIN only.
        th_tr_raw = train.theta.astype(np.float32)
        th_va_raw = val.theta.astype(np.float32)
        th_mean = th_tr_raw.mean(0)
        th_std = th_tr_raw.std(0) + 1e-8
        th_tr_n = (th_tr_raw - th_mean) / th_std
        th_va_n = (th_va_raw - th_mean) / th_std

        # Controls: piecewise-linear ramp s(t) from the raw tau
        t_grid = np.asarray(train.t_grid, dtype=np.float64)

        # The trunk input ds is formed inside the training loop by finite differences of s(t).
        ctrl_tr = _control_seq_stack(train.theta.astype(np.float32), t_grid, train.case_ids)[0]
        ctrl_va = _control_seq_stack(val.theta.astype(np.float32), t_grid, val.case_ids)[0]

        data = {
            "T": int(T),
            "z_tr_n": torch.tensor(z_tr_n, dtype=torch.float32, device=dev),
            "z_va_n": torch.tensor(z_va_n, dtype=torch.float32, device=dev),
            "th_tr": torch.tensor(th_tr_n, dtype=torch.float32, device=dev),
            "th_va": torch.tensor(th_va_n, dtype=torch.float32, device=dev),
            "ctrl_tr": torch.tensor(ctrl_tr, dtype=torch.float32, device=dev),
            "ctrl_va": torch.tensor(ctrl_va, dtype=torch.float32, device=dev),
        }

        # ---------- Training (train_phase1) ----------
        ck_dir = REPO / "results" / "benchmark" / "runs" / f"{cfg['model_name']}_seed{seed}_ckpts"
        ck_dir.mkdir(parents=True, exist_ok=True)
        rollout_max = int(m.get("rollout_max_phase1", 100))
        warmup = int(m.get("rollout_warmup_phase1", 150))
        ramp_ep = int(m.get("rollout_ramp_phase1", 300))
        var_lambda = float(m.get("var_lambda_phase1", 0.3))
        batch_cases = int(m.get("batch_size", 64))
        lr = float(m.get("lr", 1e-3))
        weight_decay = float(m.get("weight_decay", 1e-4))
        grad_clip = float(m.get("grad_clip_phase1", 1.0))
        best_mature = int(m.get("best_mature_epoch_phase1", 500))

        best_loss, train_time = train_phase1(
            dyn, data, n_epochs=int(m["epochs"]),
            rollout_max=rollout_max, warmup=warmup, ramp_ep=ramp_ep,
            var_lambda=var_lambda, batch_cases=batch_cases,
            lr=lr, weight_decay=weight_decay, grad_clip=grad_clip,
            best_mature_epoch=best_mature, dev=dev,
            ckpt_dir=ck_dir, seed=seed)
        best_ckpt = ck_dir / "best.pt"
        best_ep = -1

        # ---------- Save final ----------
        last_ckpt = ck_dir / "last.pt"
        torch.save({"state_dict": dyn.state_dict()}, last_ckpt)
        if best_ckpt.exists():
            ckpt_payload = torch.load(best_ckpt, map_location=dev, weights_only=False)
            dyn.load_state_dict(ckpt_payload["state_dict"])
            best_ep = int(ckpt_payload.get("epoch", -1))

        hist = {"train": [float(best_loss)], "val": [float("nan")]}
        ms.update({"dyn": dyn.cpu(),
                   "best_ckpt_path": str(best_ckpt) if best_ckpt.exists() else None,
                   "last_ckpt_path": str(last_ckpt), "best_epoch": best_ep,
                   "th_mean": torch.tensor(th_mean, dtype=torch.float32),
                   "th_std": torch.tensor(th_std, dtype=torch.float32),
                   "z_mean_lat": z_mean_lat_t.detach().cpu(),
                   "z_std_lat": z_std_lat_t.detach().cpu(),
                   "run_meta": {**ms.get("run_meta", {}), "best_epoch": best_ep,
                                "best_loss": float(best_loss)}})
        return ms, hist

    def predict_eval(self, ms: dict, test, split_spec, cfg: dict):
        dyn_kind = str(cfg["dynamics"])
        if dyn_kind == "ar_mlp":
            return self._predict_eval_ar_mlp(ms, test, split_spec, cfg)
        if dyn_kind in ("s4d_branch_don", "mamba_branch_don", "gru_branch_don"):
            return self._predict_eval_s4d_branch_don(ms, test, split_spec, cfg)
        raise ValueError(f"latent_dynamics.predict_eval: unknown dynamics {dyn_kind!r}")

    def _predict_eval_ar_mlp(self, ms: dict, test, split_spec, cfg: dict):
        dev = ms["device"]
        ae: ConvAutoencoder = ms["ae"].to(dev).eval()
        dyn = ms["dyn"].to(dev).eval()
        z_mean, z_std = ms["z_mean"], ms["z_std"]
        x_mean, x_std = ms["x_mean"], ms["x_std"]
        H, W = ms["grid_hw"]
        # load only the t=0 frames, case by case from the memory-mapped cache
        x0_np = np.stack([test.grid.frames_for([cid], dtype=np.float16)[0, 0]
                          for cid in test.case_ids], axis=0)      # (n_test, H, W, 3)
        n_test, T = x0_np.shape[0], len(test.t_grid)
        t_grid = np.asarray(test.t_grid, dtype=np.float64)
        t_norm = torch.tensor((t_grid / max(float(t_grid[-1]), 1e-9)).astype(np.float32), device=dev)
        m = cfg["model"]
        th = _theta_inputs(test.theta, dev)
        s, ds = _control_seq_stack(test.theta, t_grid, test.case_ids)
        s_t = torch.from_numpy(s).to(dev)
        ds_t = torch.from_numpy(ds).to(dev)
        with torch.no_grad():
            x0 = torch.from_numpy(grid_to_channels_first(x0_np)).to(dev, non_blocking=True).float()
            x0_norm = (x0 - x_mean) / x_std
            z0 = (ae.encode(x0_norm) - z_mean) / z_std
            t_full = t_norm.unsqueeze(0).expand(n_test, -1)
            z_pred = _rollout_k(s_t, ds_t, t_full, dyn, z0, th, T - 1)
            z_phys = z_pred * z_std + z_mean
            pred_cfhw = np.empty((n_test, T, 3, H, W), dtype=np.float32)
            for ci in range(n_test):
                dec_norm = ae.decode(z_phys[ci])
                dec_phys = dec_norm * x_std + x_mean
                pred_cfhw[ci] = dec_phys.cpu().numpy()
        pred_grid = channels_first_to_grid(pred_cfhw)
        arrays = assemble_kvs_field_arrays(
            pred_grid, test.case_ids, np.asarray(t_grid, dtype=np.float32),
            coords_p2=test.mesh["coords_p2"], coords_p1=test.mesh["coords_p1"],
            eval_idx_p2=split_spec.eval_idx_p2, eval_idx_p1=split_spec.eval_idx_p1,
            bbox=test.grid.bbox)
        meta = {"encoder_ref": cfg["encoder_ref"], "dynamics": cfg["dynamics"],
                "best_epoch": ms.get("best_epoch", -1)}
        return arrays, meta

    def _predict_eval_s4d_branch_don(self, ms: dict, test, split_spec, cfg: dict):
        dev = ms["device"]
        ae: ConvAutoencoder = ms["ae"].to(dev).eval()
        dyn: S4DBranchDeepONet = ms["dyn"].to(dev).eval()
        # Prefer the TRAIN latent stats set by _fit_s4d_branch_don.
        z_mean = ms.get("z_mean_lat", ms["z_mean"]).to(dev)
        z_std = ms.get("z_std_lat", ms["z_std"]).to(dev)
        x_mean, x_std = ms["x_mean"], ms["x_std"]
        H, W = ms["grid_hw"]
        m = cfg["model"]
        # load only the t=0 frames, case by case
        x0_np = np.stack([test.grid.frames_for([cid], dtype=np.float16)[0, 0]
                          for cid in test.case_ids], axis=0)
        n_test, T = x0_np.shape[0], len(test.t_grid)
        t_grid = np.asarray(test.t_grid, dtype=np.float64)
        dt = float(t_grid[1] - t_grid[0]) if T > 1 else 1.0
        # Theta: apply the training-time normalisation if present (must match fit).
        th_raw = torch.from_numpy(test.theta).float().to(dev)
        if "th_mean" in ms and "th_std" in ms:
            th = (th_raw - ms["th_mean"].to(dev)) / ms["th_std"].to(dev)
        else:
            th = th_raw
        s = _control_seq_stack(test.theta.astype(np.float32), t_grid, test.case_ids)[0]
        s_t = torch.from_numpy(s).to(dev)
        ds_t = torch.zeros_like(s_t)
        if T >= 3:
            ds_t[:, 1:-1] = (s_t[:, 2:] - s_t[:, :-2]) / (2 * dt)
            ds_t[:, 0] = (s_t[:, 1] - s_t[:, 0]) / dt
            ds_t[:, -1] = (s_t[:, -1] - s_t[:, -2]) / dt
        elif T == 2:
            ds_t[:, 0] = (s_t[:, 1] - s_t[:, 0]) / dt
            ds_t[:, 1] = ds_t[:, 0]
        ramp_seq = s_t.unsqueeze(-1)
        bf = (th, ramp_seq)
        with torch.no_grad():
            x0 = torch.from_numpy(grid_to_channels_first(x0_np)).to(dev, non_blocking=True).float()
            x0_norm = (x0 - x_mean) / x_std
            z_ar = (ae.encode(x0_norm) - z_mean) / z_std
            preds = [z_ar]
            for t_step in range(T - 1):
                t_f = torch.full((n_test, 1), t_step / T, dtype=torch.float32, device=dev)
                s_f = s_t[:, t_step].unsqueeze(1)
                ds_f = ds_t[:, t_step].unsqueeze(1)
                trunk_in = torch.cat([z_ar, t_f, s_f, ds_f], dim=1)
                z_ar = z_ar + dyn(bf, trunk_in)
                preds.append(z_ar)
            z_pred = torch.stack(preds, dim=1)
            z_phys = z_pred * z_std + z_mean
            pred_cfhw = np.empty((n_test, T, 3, H, W), dtype=np.float32)
            for ci in range(n_test):
                dec_norm = ae.decode(z_phys[ci])
                dec_phys = dec_norm * x_std + x_mean
                pred_cfhw[ci] = dec_phys.cpu().numpy()
        pred_grid = channels_first_to_grid(pred_cfhw)
        arrays = assemble_kvs_field_arrays(
            pred_grid, test.case_ids, np.asarray(t_grid, dtype=np.float32),
            coords_p2=test.mesh["coords_p2"], coords_p1=test.mesh["coords_p1"],
            eval_idx_p2=split_spec.eval_idx_p2, eval_idx_p1=split_spec.eval_idx_p1,
            bbox=test.grid.bbox)
        meta = {"encoder_ref": cfg["encoder_ref"], "dynamics": cfg["dynamics"],
                "best_epoch": ms.get("best_epoch", -1)}
        return arrays, meta
