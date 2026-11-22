# benchmark/adapters/latent_oneshot.py
# ============================================================================
# Purpose: the latent x one-shot cell of the model matrix. The same frozen
#   ConvAE as the AR rows, but the latent trajectory is produced in a single
#   forward pass: a branch reads the ramp and theta into per-latent
#   coefficients, a trunk reads only the scalar time t, and their product gives
#   z(t) directly. No hidden state and no autoregressive feedback, so an error
#   at one instant cannot compound into the next. Two branch types are
#   available, 'gru' and 's4d'. With the branch architecture held fixed, the
#   comparison with the AR rows isolates the effect of the recursion.
# Pipeline role: same chain position as latent_dynamics (frozen encoder ->
#   this adapter -> predictions.npz -> eval_harness), producing the
#   oneshot_gru / oneshot_s4d rows of the KVS table.
#   Paper: KVS table (time-axis control, latent x one-shot).
# Eval-consistency: no statistics are fitted here. The latent statistics
#   z_mean/z_std and the field statistics x_mean/x_std come from the frozen
#   encoder file, so this row works in the same latent space as the AR rows,
#   with the same latent normalisation as the AR-MLP row; theta is fed
#   unnormalised. Splits come from the runner's frozen views.
#   set_seed(seed) runs in build() and fit(); a separate default_rng(seed) only
#   permutes the cases per epoch. Best-checkpoint selection is on the validation
#   loss (MSE on the full latent trajectory), evaluated every val_every epochs
#   (default 25). Checkpoints go to
#   results/benchmark/runs/<model_name>_<split>_seed<seed>_ckpts/.
# ============================================================================
"""latent_oneshot adapter: Branch(theta, ramp) . Trunk(t) -> z(t) in one forward.

The latent x one-shot cell of the model matrix: no state, no AR feedback. The branch
reads the ramp s(t) (+theta) into per-latent coefficients; the trunk reads only the
scalar time t. Two branch variants via cfg["branch_type"]: 'gru' (same construction as
the field DeepONet branch, pairs with the field DeepONet along the representation
axis) and 's4d' (same branch as the S4D AR row, pairs with the latent-AR rows along
the time axis). Reuses the frozen-encoder load / encode helpers from
latent_dynamics, so the cell uses the same KVS latent pipeline.

Loss: MSE on the normalised z-trajectory. AdamW + CosineAnnealingLR, early stopping
after early_stop_patience epochs without improvement (from epoch 100 on).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from benchmark.adapters import BenchmarkAdapter, register_adapter
from benchmark.adapters._common import (EpochLogger, assemble_kvs_field_arrays,
                                        channels_first_to_grid, ramp_s_dsdt, save_ckpt)
from benchmark.adapters._dynamics_models import LatentOneShotDON
from benchmark.adapters.latent_dynamics import (REPO, _device, _encode_traj,
                                                _load_frozen_encoder)
from tools.common.seeding import set_seed
from tools.models.conv_autoencoder import ConvAutoencoder


def _branch_type(cfg: dict) -> str:
    return str(cfg.get("branch_type", cfg["model"].get("branch_type", "gru"))).strip().lower()


@register_adapter("latent_oneshot")
class LatentOneShotAdapter(BenchmarkAdapter):

    def build(self, cfg: dict, seed: int) -> dict:
        set_seed(seed)
        m = cfg["model"]
        dev = _device(m)
        ae, (H, W), latent_dim, z_mean, z_std, x_mean, x_std = _load_frozen_encoder(cfg["encoder_ref"], dev)
        dyn = LatentOneShotDON(
            latent_dim=latent_dim, theta_dim=6, branch_type=_branch_type(cfg),
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
        n_params = sum(p.numel() for p in dyn.parameters())
        return {"ae": ae, "dyn": dyn, "device": dev, "grid_hw": [H, W],
                "latent_dim": latent_dim, "z_mean": z_mean, "z_std": z_std,
                "x_mean": x_mean, "x_std": x_std, "n_params": int(n_params),
                "run_meta": {"encoder_ref": cfg["encoder_ref"], "adapter": "latent_oneshot",
                             "branch_type": dyn.branch_type, "latent_dim": latent_dim}}

    def fit(self, ms: dict, train, val, cfg: dict, seed: int):
        set_seed(seed)
        m = cfg["model"]
        dev = ms["device"]
        ae: ConvAutoencoder = ms["ae"]
        dyn = ms["dyn"].to(dev)
        z_mean, z_std = ms["z_mean"], ms["z_std"]
        x_mean, x_std = ms["x_mean"], ms["x_std"]
        n_epochs = int(m["epochs"])
        log = EpochLogger("latent_oneshot", n_epochs)

        # Encode train/val trajectories with the frozen encoder; latent-normalise.
        tr_frames = train.grid.frames_for(train.case_ids, dtype=np.float16)
        va_frames = val.grid.frames_for(val.case_ids, dtype=np.float16)
        zmn, zsd = z_mean.cpu().numpy(), z_std.cpu().numpy()
        z_tr = ((_encode_traj(ae, tr_frames, dev, x_mean, x_std) - zmn) / zsd).astype(np.float32)
        z_va = ((_encode_traj(ae, va_frames, dev, x_mean, x_std) - zmn) / zsd).astype(np.float32)
        N, T, d = z_tr.shape
        n_val = len(val.case_ids)

        # Controls s(t) over the full grid + normalised time query.
        t_grid = np.asarray(train.t_grid, dtype=np.float64)
        s_tr = np.stack([ramp_s_dsdt(train.theta[i, 1:], t_grid)[0] for i in range(N)], 0).astype(np.float32)
        s_va = np.stack([ramp_s_dsdt(val.theta[i, 1:], t_grid)[0] for i in range(n_val)], 0).astype(np.float32)
        t_norm = (t_grid / max(float(t_grid[-1]), 1e-9)).astype(np.float32)

        z_tr_t = torch.from_numpy(z_tr).to(dev)
        z_va_t = torch.from_numpy(z_va).to(dev)
        th_tr = torch.from_numpy(train.theta.astype(np.float32)).to(dev)
        th_va = torch.from_numpy(val.theta.astype(np.float32)).to(dev)
        s_tr_t = torch.from_numpy(s_tr).unsqueeze(-1).to(dev)      # (N, T, 1)
        s_va_t = torch.from_numpy(s_va).unsqueeze(-1).to(dev)
        t_q = torch.from_numpy(t_norm).to(dev)                    # (T,)

        bs = int(m.get("batch_size", 64))
        val_every = int(m.get("val_every", 25))
        patience = int(m.get("early_stop_patience", 200))
        opt = torch.optim.AdamW(dyn.parameters(), lr=float(m.get("lr", 5e-4)),
                                weight_decay=float(m.get("weight_decay", 1e-6)))
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(n_epochs, 1))
        rng = np.random.default_rng(seed)
        best_val, best_ep, no_imp = float("inf"), -1, 0
        hist = {"train": [], "val": []}
        # The split is part of the directory name, so interp and extrap runs with the
        # same model_name and seed do not overwrite each other's best.pt/last.pt.
        ck_dir = REPO / "results" / "benchmark" / "runs" / f"{cfg['model_name']}_{cfg['split']}_seed{seed}_ckpts"
        best_ckpt = ck_dir / "best.pt"
        last_ckpt = ck_dir / "last.pt"

        for epoch in range(1, n_epochs + 1):
            log.epoch_start()
            dyn.train()
            perm = rng.permutation(N)
            ep_loss, nb = 0.0, 0
            for i in range(0, N, bs):
                idx_t = torch.as_tensor(perm[i:i + bs], dtype=torch.long, device=dev)
                zt = z_tr_t.index_select(0, idx_t)                # (B, T, d)
                zp = dyn(th_tr.index_select(0, idx_t),
                         s_tr_t.index_select(0, idx_t), t_q)      # (B, T, d)
                loss = F.mse_loss(zp, zt)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(dyn.parameters(), 1.0)
                opt.step()
                ep_loss += float(loss.item()); nb += 1
            sched.step()
            tr = ep_loss / max(nb, 1)
            vl = float("nan")
            if epoch % val_every == 0 or epoch == n_epochs:
                dyn.eval()
                with torch.no_grad():
                    vl = float(F.mse_loss(dyn(th_va, s_va_t, t_q), z_va_t).item())
                if vl < best_val:
                    best_val, best_ep, no_imp = vl, epoch, 0
                    save_ckpt(ck_dir, best_ckpt.name, {"epoch": epoch, "dyn_state_dict": dyn.state_dict()})
                    log.saved(best_ckpt.name, vl, epoch)
                else:
                    no_imp += val_every
            hist["train"].append(tr); hist["val"].append(vl)
            log.epoch_end(epoch, tr, vl if vl == vl else tr,
                          best_val if best_val < float("inf") else tr, best_ep)
            if no_imp >= patience and epoch > 100:
                print(f"[latent_oneshot] early stop at epoch {epoch}", flush=True)
                break
        save_ckpt(ck_dir, last_ckpt.name, {"epoch": epoch, "dyn_state_dict": dyn.state_dict()})
        if best_ckpt.exists():
            dyn.load_state_dict(torch.load(best_ckpt, map_location=dev, weights_only=False)["dyn_state_dict"])
        ms.update({"dyn": dyn.cpu(), "best_ckpt_path": str(best_ckpt) if best_ckpt.exists() else None,
                   "last_ckpt_path": str(last_ckpt), "best_epoch": best_ep,
                   "run_meta": {**ms.get("run_meta", {}), "best_epoch": best_ep, "best_val": best_val}})
        return ms, hist

    def predict_eval(self, ms: dict, test, split_spec, cfg: dict):
        dev = ms["device"]
        ae: ConvAutoencoder = ms["ae"].to(dev).eval()
        dyn = ms["dyn"].to(dev).eval()
        z_mean, z_std = ms["z_mean"], ms["z_std"]
        x_mean, x_std = ms["x_mean"], ms["x_std"]
        H, W = ms["grid_hw"]
        n_test = len(test.case_ids)
        t_grid = np.asarray(test.t_grid, dtype=np.float64)
        T = len(t_grid)
        t_norm = (t_grid / max(float(t_grid[-1]), 1e-9)).astype(np.float32)
        th = torch.from_numpy(test.theta.astype(np.float32)).to(dev)
        s = np.stack([ramp_s_dsdt(test.theta[i, 1:], t_grid)[0] for i in range(n_test)], 0).astype(np.float32)
        s_t = torch.from_numpy(s).unsqueeze(-1).to(dev)
        t_q = torch.from_numpy(t_norm).to(dev)
        with torch.no_grad():
            z_pred = dyn(th, s_t, t_q)                            # (n_test, T, d) normalised
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
        meta = {"encoder_ref": cfg["encoder_ref"], "branch_type": ms["run_meta"].get("branch_type"),
                "best_epoch": ms.get("best_epoch", -1)}
        return arrays, meta
