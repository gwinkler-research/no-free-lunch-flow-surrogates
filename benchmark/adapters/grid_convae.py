# benchmark/adapters/grid_convae.py
# ============================================================================
# Purpose: train the 2D convolutional autoencoder on the rasterised KVS fields
#   (u, v, p) and write the frozen encoder that every latent row depends on.
#   fit() is reconstructive only: no dynamics, no time model. Besides best/last
#   checkpoints it writes one encoder file that bundles the weights with
#   everything a later run needs to reproduce the same latent space: grid_hw,
#   latent_dim, base_channels, n_downsample, the channel statistics
#   x_mean/x_std and the latent statistics z_mean/z_std. predict_eval() runs
#   the plain encode-decode roundtrip, the representation floor of every
#   latent model in the KVS table.
# Pipeline role: the encoder stage at the head of the latent chain. The encoder
#   file is what latent_dynamics._load_frozen_encoder loads for the AR rows,
#   the DMDc row and the one-shot rows; configs point to it via encoder_ref.
#   Its n_params and training wallclock enter the cost table.
#   Paper: Sec. 3 (frozen encoder) and the cost table.
# Eval-consistency: the per-channel statistics x_mean/x_std are computed from
#   the TRAIN cases only and applied to train and val alike; the latent
#   statistics z_mean/z_std likewise come from TRAIN latents only (encoded with
#   the best checkpoint). Both are stored in the encoder file, so every
#   downstream row uses the same normalisation. set_seed(seed) runs in build()
#   and again in fit(); a separate np.random.default_rng(seed) drives the epoch
#   permutation and the subsequence offsets. The best checkpoint is selected on
#   the validation reconstruction loss, evaluated every val_every epochs
#   (default 25), so only those epochs can be selected. The encoder file name
#   is built in _encoder_path(): kvs_<split>_convae_<H>x<W>_bc<base_channels>.pt,
#   with _dim<latent_dim> appended when latent_dim != 32.
# ============================================================================
"""grid_convae adapter: frozen 2D convolutional autoencoder on rasterised KVS.

build:        ConvAutoencoder (tools/models/conv_autoencoder.py), in_channels=3
              ([u,v,p]), grid_h/grid_w from cfg, latent/base_channels/n_downsample
              from cfg["model"].
fit:          reconstructive training on the rasterised TRAIN cases (cache view);
              AdamW + cosine LR schedule, AMP on CUDA, early stopping after
              early_stop_patience epochs without improvement (from epoch 100 on).
              Writes best+last checkpoints and the frozen encoder file
              <encoder_out_dir>/kvs_<split>_convae_<H>x<W>_bc<bc>[_dim<d>].pt
              (default dir results/benchmark/encoders) holding the full
              state_dict + grid_hw + channel and latent normalisation stats.
predict_eval: encode->decode roundtrip on the test cases -> grid_to_nodes ->
              kvs/field arrays. Also writes <encoder stem>_roundtrip.json with
              rel_l2 (all channels) and fft_freq_err mean+median and power_ratio
              (v channel) on the test cases: the representation floor.

Note on T: The kvs/field schema (validated by benchmark.prediction_io.validate)
requires exactly T=201 timesteps and t_grid.shape==(201,). The KVS cases have 201
saved steps (checked in tools.data.karman_h5_dataset), so no resampling is needed.
"""
from __future__ import annotations

import json as _json
from pathlib import Path

import numpy as np
import torch

from benchmark.adapters import BenchmarkAdapter, register_adapter
from benchmark.adapters._common import (EpochLogger, assemble_kvs_field_arrays,
                                        channels_first_to_grid, grid_to_channels_first,
                                        latent_norm_stats, save_ckpt)
from benchmark.metrics_kvs import _fft_freq_err, _power_ratio
from tools.common.seeding import set_seed
from tools.models.conv_autoencoder import ConvAutoencoder

REPO = Path(__file__).resolve().parents[2]
_DEFAULT_ENC_DIR = REPO / "results" / "benchmark" / "encoders"


def _device(cfg_model: dict) -> torch.device:
    d = str(cfg_model.get("device", "cpu"))
    if d == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(d)


def _encoder_path(cfg) -> Path:
    enc_dir = Path(cfg.get("encoder_out_dir", _DEFAULT_ENC_DIR))
    H, W = cfg["grid_hw"]
    bc = int(cfg["model"]["base_channels"])
    dim = int(cfg["model"]["latent_dim"])
    # The file name carries base_channels and, unless it is 32, latent_dim.
    if dim == 32:
        return enc_dir / f"kvs_{cfg['split']}_convae_{H}x{W}_bc{bc}.pt"
    return enc_dir / f"kvs_{cfg['split']}_convae_{H}x{W}_bc{bc}_dim{dim}.pt"


@register_adapter("grid_convae")
class GridConvAEAdapter(BenchmarkAdapter):

    def build(self, cfg: dict, seed: int) -> dict:
        set_seed(seed)
        m = cfg["model"]
        H, W = cfg["grid_hw"]
        model = ConvAutoencoder(in_channels=3, latent_dim=int(m["latent_dim"]),
                                base_channels=int(m["base_channels"]),
                                n_downsample=int(m["n_downsample"]), grid_h=H, grid_w=W)
        n_params = sum(p.numel() for p in model.parameters())
        return {"model": model, "n_params": int(n_params), "device": _device(m),
                "run_meta": {"latent_dim": int(m["latent_dim"]), "grid_hw": [H, W]}}

    def fit(self, ms: dict, train, val, cfg: dict, seed: int):
        set_seed(seed)
        m = cfg["model"]
        dev = ms["device"]
        model: ConvAutoencoder = ms["model"].to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=float(m["lr"]),
                                weight_decay=float(m.get("weight_decay", 1e-4)))
        amp_dtype = torch.float16 if dev.type == "cuda" else torch.float32
        use_amp = dev.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        n_epochs = int(m["epochs"])
        log = EpochLogger("grid_convae", n_epochs)
        cache_dtype = np.float16 if use_amp else np.float32
        tr_frames = train.grid.frames_for(train.case_ids, dtype=cache_dtype)   # (N,T,H,W,3) fp16/fp32
        va_frames = val.grid.frames_for(val.case_ids, dtype=cache_dtype)
        N, T, H, W, _ = tr_frames.shape
        tr_x = torch.from_numpy(grid_to_channels_first(tr_frames))   # (N,T,3,H,W) fp16/fp32
        va_x = torch.from_numpy(grid_to_channels_first(va_frames))
        del tr_frames, va_frames
        # Per-channel input normalisation with stats from TRAIN only (no leak from val/test).
        # Stored in the encoder file; predict_eval and the latent adapters normalise and
        # de-normalise with these stats, so the encoder always sees the same input scaling.
        tr_f32 = tr_x.float()                                 # (N,T,3,H,W) fp32 temporary
        x_mean = tr_f32.mean(dim=(0, 1, 3, 4), keepdim=True)  # (1,1,3,1,1)
        x_std = tr_f32.std(dim=(0, 1, 3, 4), keepdim=True).clamp(min=1e-8)
        del tr_f32
        tr_x = ((tr_x.float() - x_mean) / x_std).to(tr_x.dtype)
        va_x = ((va_x.float() - x_mean) / x_std).to(va_x.dtype)
        print(f"[grid_convae] input normalised per-channel: "
              f"mean={[round(v, 4) for v in x_mean.flatten().tolist()]}, "
              f"std={[round(v, 4) for v in x_std.flatten().tolist()]}", flush=True)
        bs = int(m["batch_size"])
        val_every = int(m.get("val_every", 25))
        patience = int(m.get("early_stop_patience", 100))
        lr_lambda_recon = float(m.get("lambda_recon", 1.0))
        lr_lambda_temporal = float(m.get("lambda_temporal", 0.0))
        # Optional sub-sequence and case subsampling (default: full T, all cases)
        sub = m.get("subsample", {}) or {}
        case_ratio = float(sub.get("case_ratio", 1.0))
        subseq_len = int(sub.get("subseq_len", T))
        if not (0.0 < case_ratio <= 1.0):
            raise ValueError(f"subsample.case_ratio must be in (0,1], got {case_ratio}")
        if not (2 <= subseq_len <= T):
            raise ValueError(f"subsample.subseq_len must be in [2,{T}], got {subseq_len}")
        n_use = max(bs, int(round(N * case_ratio)))
        if (case_ratio < 1.0) or (subseq_len < T):
            print(f"[grid_convae] subsample: case_ratio={case_ratio} -> {n_use}/{N} cases/ep, "
                  f"subseq_len={subseq_len}/{T} frames", flush=True)
        # Optional GPU preloading (train+val resident on the GPU in fp16)
        gpu_preload = bool(m.get("gpu_preload", False)) and dev.type == "cuda"
        if gpu_preload:
            tr_gb = tr_x.element_size() * tr_x.numel() / 1e9
            va_gb = va_x.element_size() * va_x.numel() / 1e9
            print(f"[grid_convae] GPU-preloading {tr_x.dtype} data: {tr_gb:.1f} GB train + {va_gb:.1f} GB val "
                  f"-> {dev}", flush=True)
            tr_x = tr_x.to(dev)
            va_x = va_x.to(dev)
        # Cosine learning-rate schedule over all epochs
        sched_cfg = m.get("lr_schedule", {}) or {}
        if sched_cfg.get("kind", "cosine") != "cosine":
            raise ValueError(f"grid_convae: only the cosine schedule is supported, got {sched_cfg}")
        eta_min = float(sched_cfg.get("eta_min", float(m["lr"]) / 100.0))
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs, eta_min=eta_min)
        print(f"[grid_convae] CosineAnnealingLR enabled: T_max={n_epochs}, eta_min={eta_min:.2e}", flush=True)
        rng = np.random.default_rng(seed)
        best_val, best_ep, no_imp = float("inf"), -1, 0
        hist = {"train": [], "val": []}
        enc_dir = Path(cfg.get("encoder_out_dir", _DEFAULT_ENC_DIR))
        enc_dir.mkdir(parents=True, exist_ok=True)
        # The split is part of the checkpoint names, so interp and extrap runs with the
        # same model_name do not overwrite each other.
        best_ckpt = enc_dir / f"{cfg['model_name']}_{cfg['split']}_seed{seed}_best.pt"
        last_ckpt = enc_dir / f"{cfg['model_name']}_{cfg['split']}_seed{seed}_last.pt"
        TOTAL_KEY = "total"   # ConvAutoencoder.losses returns {"recon", "temporal", "total"}
        cur_lr = float(m["lr"])
        for epoch in range(1, n_epochs + 1):
            log.epoch_start()
            model.train()
            perm = rng.permutation(N)[:n_use]
            ep_loss, nb = 0.0, 0
            for i in range(0, n_use, bs):
                idx = perm[i:i + bs]
                if subseq_len < T:
                    t0 = int(rng.integers(0, T - subseq_len + 1))
                    x = tr_x[idx, t0:t0 + subseq_len]
                else:
                    x = tr_x[idx]
                if x.device != dev:
                    x = x.to(dev, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                    losses = model.losses(x, mask=None, lambda_recon=lr_lambda_recon,
                                          lambda_temporal=lr_lambda_temporal)
                scaler.scale(losses[TOTAL_KEY]).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                ep_loss += float(losses[TOTAL_KEY].item()); nb += 1
            tr = ep_loss / max(nb, 1)
            # Cosine schedule steps every epoch
            sched.step()
            vl = float("nan")
            if epoch % val_every == 0 or epoch == n_epochs:
                model.eval()
                with torch.no_grad(), torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                    vl_sum, vl_nb = 0.0, 0
                    for j in range(0, va_x.shape[0], bs):
                        vx = va_x[j:j + bs]
                        if vx.device != dev:
                            vx = vx.to(dev, non_blocking=True)
                        vl_sum += float(model.losses(vx, mask=None,
                                                     lambda_recon=lr_lambda_recon,
                                                     lambda_temporal=lr_lambda_temporal)[TOTAL_KEY].item())
                        vl_nb += 1
                    vl = vl_sum / max(vl_nb, 1)
                if vl < best_val:
                    best_val, best_ep, no_imp = vl, epoch, 0
                    save_ckpt(enc_dir, best_ckpt.name, {"epoch": epoch, "model_state_dict": model.state_dict()})
                    log.saved(best_ckpt.name, vl, epoch)
                else:
                    no_imp += val_every
            # Log any LR change
            new_lr = opt.param_groups[0]["lr"]
            if abs(new_lr - cur_lr) / max(cur_lr, 1e-12) > 0.05:
                print(f"[grid_convae] LR {cur_lr:.2e} -> {new_lr:.2e} at ep{epoch}", flush=True)
                cur_lr = new_lr
            hist["train"].append(tr); hist["val"].append(vl)
            log.epoch_end(epoch, tr, vl if vl == vl else tr, best_val if best_val < float("inf") else tr, best_ep)
            if no_imp >= patience and epoch > 100:
                print(f"[grid_convae] early stop at epoch {epoch}", flush=True); break
        save_ckpt(enc_dir, last_ckpt.name, {"epoch": epoch, "model_state_dict": model.state_dict()})
        if best_ckpt.exists():
            model.load_state_dict(torch.load(best_ckpt, map_location=dev, weights_only=True)["model_state_dict"])
        model.eval()
        latent_dim = int(m["latent_dim"])
        zt = np.empty((N * T, latent_dim), dtype=np.float32)
        enc_bs = max(1, bs * 32)
        flat = tr_x.reshape(N * T, 3, H, W)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            for j in range(0, flat.shape[0], enc_bs):
                zt[j:j + enc_bs] = model.encode(flat[j:j + enc_bs].to(dev, non_blocking=True)).float().cpu().numpy()
        z_mean, z_std = latent_norm_stats(zt)
        x_mean_np = x_mean.flatten().cpu().numpy()        # (3,) per-channel
        x_std_np = x_std.flatten().cpu().numpy()
        enc_path = _encoder_path(cfg)
        save_ckpt(enc_path.parent, enc_path.name, {
            "model_state_dict": model.state_dict(), "grid_hw": list(cfg["grid_hw"]),
            "latent_dim": int(m["latent_dim"]), "base_channels": int(m["base_channels"]),
            "n_downsample": int(m["n_downsample"]), "z_mean": z_mean, "z_std": z_std,
            "x_mean": x_mean_np, "x_std": x_std_np,
            "split": cfg["split"], "model_name": cfg["model_name"],
        })
        print(f"[grid_convae] Saved frozen encoder {enc_path}", flush=True)
        ms.update({"model": model.cpu(), "best_ckpt_path": str(best_ckpt) if best_ckpt.exists() else None,
                   "last_ckpt_path": str(last_ckpt), "encoder_ckpt_path": str(enc_path),
                   "z_mean": z_mean, "z_std": z_std, "x_mean": x_mean_np, "x_std": x_std_np,
                   "best_epoch": best_ep,
                   "run_meta": {**ms.get("run_meta", {}), "best_epoch": best_ep, "best_val": best_val}})
        return ms, hist

    def predict_eval(self, ms: dict, test, split_spec, cfg: dict):
        dev = ms["device"]
        model: ConvAutoencoder = ms["model"].to(dev).eval()
        H, W = cfg["grid_hw"]
        frames = test.grid.frames_for(test.case_ids)             # (n_test, T, H, W, 3) fp32 PHYSICAL
        n_test, T = frames.shape[0], frames.shape[1]
        x = torch.from_numpy(grid_to_channels_first(frames)).float()    # (n_test, T, 3, H, W)
        # Apply train-stats normalisation before encoder (encoder operates in normalised space)
        x_mean = torch.tensor(np.asarray(ms["x_mean"]), dtype=torch.float32).reshape(1, 3, 1, 1).to(dev)
        x_std = torch.tensor(np.asarray(ms["x_std"]), dtype=torch.float32).reshape(1, 3, 1, 1).to(dev)
        pred_cfhw = np.empty((n_test, T, 3, H, W), dtype=np.float32)
        with torch.no_grad():
            for ci in range(n_test):
                xc = x[ci].to(dev)                       # (T, 3, H, W) physical
                xc_norm = (xc - x_mean) / x_std
                pred_norm = model(xc_norm)
                pred_phys = pred_norm * x_std + x_mean   # denormalise back to physical for metrics
                pred_cfhw[ci] = pred_phys.cpu().numpy()
        pred_grid = channels_first_to_grid(pred_cfhw)            # (n_test, T, H, W, 3) PHYSICAL

        # Floor metrics on the original time axis, pred and truth in physical units.
        # fft_freq_err and power_ratio use the v channel; rel_l2 uses all channels.
        true_v = frames[..., 1].reshape(n_test, T, -1)
        pred_v = pred_grid[..., 1].reshape(n_test, T, -1)
        t_grid = np.asarray(test.t_grid, dtype=np.float64)
        fft_mean, fft_median = _fft_freq_err(pred_v, true_v, t_grid)
        power_ratio = _power_ratio(pred_v, true_v, t_grid)
        rel_l2 = float(np.linalg.norm(pred_grid - frames) / max(float(np.linalg.norm(frames)), 1e-12))

        # Handle NaN from short sequences (fall back to 0.0 / 1.0)
        if not np.isfinite(fft_mean):
            fft_mean = 0.0
        if not np.isfinite(fft_median):
            fft_median = 0.0
        if not np.isfinite(power_ratio):
            power_ratio = 1.0

        floor = {"rel_l2": rel_l2, "fft_freq_err_mean": fft_mean, "fft_freq_err_median": fft_median,
                 "power_ratio": power_ratio, "n_test_cases": int(n_test), "grid_hw": list(cfg["grid_hw"])}
        rj_path = _encoder_path(cfg).with_name(_encoder_path(cfg).stem + "_roundtrip.json")
        rj_path.parent.mkdir(parents=True, exist_ok=True)
        rj_path.write_text(_json.dumps(floor, indent=2))
        print(f"[grid_convae] roundtrip floor {floor} -> {rj_path}", flush=True)

        arrays = assemble_kvs_field_arrays(
            pred_grid, test.case_ids, np.asarray(test.t_grid, dtype=np.float32),
            coords_p2=test.mesh["coords_p2"], coords_p1=test.mesh["coords_p1"],
            eval_idx_p2=split_spec.eval_idx_p2, eval_idx_p1=split_spec.eval_idx_p1,
            bbox=test.grid.bbox)
        meta = {"encoder_ref": ms["encoder_ckpt_path"], "roundtrip_floor": floor,
                "best_epoch": ms.get("best_epoch", -1)}
        ms["roundtrip_json_path"] = str(rj_path)
        return arrays, meta
