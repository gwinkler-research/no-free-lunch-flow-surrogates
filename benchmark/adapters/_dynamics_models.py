# =====================================================================
# nn.Module definitions of the latent-dynamics models, shared by
# benchmark/adapters/latent_dynamics.py and latent_oneshot.py (KVS) and by
# cmp3d_latent_ar.py and cmp3d_latent_oneshot.py (CMP). Contains the
# feedforward ARMlpStepper, the branch-trunk dot-product DeepONets with an
# S4D, GRU or Mamba branch, and the one-shot LatentOneShotDON.
#
# These classes define the {s4d,mamba,gru}_branch_don, ar_mlp and
# latent_oneshot_* rows of the KVS and CMP results tables.
#
# All AR models read the same per-step features (z, theta, t, s(t), ds/dt).
# =====================================================================
"""Latent-dynamics models used by the latent adapters.

ar_mlp           -> ARMlpStepper: feedforward MLP (z, theta, t, s, ds) -> dz, rolled
                    out autoregressively z(t+1) = z(t) + dz.
s4d_branch_don   -> S4DBranchDeepONet: Branch_S4D(theta, ramp_seq) . Trunk_MLP([z, t/T, s, ds])
                    -> dz, rolled out the same way.
mamba_branch_don -> MambaBranchDeepONet: as S4D, with Mamba layers in the branch.
gru_branch_don   -> GRUBranchDeepONet: as S4D, with a GRU branch.
LatentOneShotDON -> Branch(theta, ramp_seq) . Trunk(t) -> z(t) for all t in one forward.
"""
from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn

from tools.models.ssm import S4DLayer

_ACT = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh, "silu": nn.SiLU}


def _mlp(in_dim: int, hidden: Iterable[int], out_dim: int, act: str = "silu", dropout: float = 0.0) -> nn.Sequential:
    layers, prev = [], int(in_dim)
    for h in hidden:
        layers += [nn.Linear(prev, int(h)), _ACT[act]()]
        if dropout > 0.0:
            layers.append(nn.Dropout(float(dropout)))
        prev = int(h)
    layers.append(nn.Linear(prev, int(out_dim)))
    return nn.Sequential(*layers)


# Plain concat-MLP, not a DeepONet: no branch/trunk split and no dot product.
class ARMlpStepper(nn.Module):
    """MLP step: input = concat(z, theta, t, s, ds) -> dz (same dim as z)."""

    def __init__(self, latent_dim: int, theta_dim: int, hidden, activation: str = "silu", dropout: float = 0.0):
        super().__init__()
        self.latent_dim = int(latent_dim)
        in_dim = int(latent_dim) + int(theta_dim) + 3   # +t, +s, +ds
        self.net = _mlp(in_dim, hidden, latent_dim, act=activation, dropout=dropout)

    def forward(self, z: torch.Tensor, theta: torch.Tensor, t: torch.Tensor, s: torch.Tensor, ds: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z, theta, t, s, ds], dim=-1)
        return self.net(x)


# `(hb * ht).sum(dim=-1) + bias` in forward() (same pattern in GRUBranchDeepONet and
# MambaBranchDeepONet below) is the branch-trunk dot product over `basis_dim`, the
# defining DeepONet operation, as opposed to a concat-then-MLP merge.
class S4DBranchDeepONet(nn.Module):
    """Branch_S4D(theta, ramp_seq) . Trunk_MLP([z, t/T, s, ds]) -> dz.

    The branch runs the S4D stack over the ramp sequence, mean-pools over time
    and concatenates a linear projection of theta. The caller integrates dz
    autoregressively: z(t+1) = z(t) + dz.

    Interface:
        forward(branch_in=(theta, ramp_seq), trunk_in) -> dz
            theta:    (B, theta_dim)
            ramp_seq: (B, T, 1)
            trunk_in: (B, latent_dim + 3)  -> [z, t/T, s, ds]
            dz:       (B, latent_dim)
    """

    def __init__(self, latent_dim: int, theta_dim: int = 6,
                 branch_width: int = 128, branch_n_layers: int = 4,
                 branch_d_state: int = 64,
                 dt_min: float = 0.0001, dt_max: float = 0.01,
                 trunk_width: int = 512, trunk_depth: int = 5,
                 basis_dim: int = 128):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.basis_dim = int(basis_dim)
        # Branch
        self.theta_proj = nn.Linear(int(theta_dim), int(branch_width))
        self.ramp_proj = nn.Linear(1, int(branch_width))
        self.s4d_layers = nn.ModuleList([
            S4DLayer(int(branch_width), d_state=int(branch_d_state),
                     dt_min=float(dt_min), dt_max=float(dt_max))
            for _ in range(int(branch_n_layers))
        ])
        self.branch_pool = nn.AdaptiveAvgPool1d(1)
        self.branch_out = nn.Linear(int(branch_width) * 2, int(latent_dim) * int(basis_dim))
        # Trunk
        trunk_layers: list[nn.Module] = [
            nn.Linear(int(latent_dim) + 3, int(trunk_width)),
            nn.LayerNorm(int(trunk_width)),
            nn.SiLU(),
        ]
        for _ in range(int(trunk_depth) - 2):
            trunk_layers += [nn.Linear(int(trunk_width), int(trunk_width)),
                             nn.LayerNorm(int(trunk_width)),
                             nn.SiLU()]
        trunk_layers.append(nn.Linear(int(trunk_width), int(latent_dim) * int(basis_dim)))
        self.trunk = nn.Sequential(*trunk_layers)
        self.bias = nn.Parameter(torch.zeros(int(latent_dim)))

    def forward(self, branch_in, trunk_in: torch.Tensor) -> torch.Tensor:
        theta, ramp_seq = branch_in
        x = self.ramp_proj(ramp_seq)
        for layer in self.s4d_layers:
            x = layer(x)
        h_pool = self.branch_pool(x.transpose(1, 2)).squeeze(-1)
        hb = self.branch_out(torch.cat([h_pool, self.theta_proj(theta)], dim=1))
        hb = hb.view(-1, self.latent_dim, self.basis_dim)
        ht = self.trunk(trunk_in).view(-1, self.latent_dim, self.basis_dim)
        return (hb * ht).sum(dim=-1) + self.bias


class GRUBranchDeepONet(nn.Module):
    """Branch_GRU(theta, ramp_seq) . Trunk_MLP([z, t/T, s, ds]) -> dz.

    Classical-RNN counterpart of S4DBranchDeepONet / MambaBranchDeepONet. The GRU
    branch is structurally identical to LatentOneShotDON's gru branch (and to the
    field-space KvsFieldDeepONet branch), but the trunk reads the autoregressive
    state [z, t/T, s, ds] and the output is a per-step increment dz (caller
    integrates z(t+1) = z(t) + dz). Trunk + dot-product are identical to
    S4DBranchDeepONet, so along the architecture axis the only difference is the
    branch mechanism (gated recurrence vs structured/selective state-space).

    Interface (identical to S4DBranchDeepONet, so both share the fit/predict path):
        forward(branch_in=(theta, ramp_seq), trunk_in) -> dz
            theta:    (B, theta_dim)
            ramp_seq: (B, T, 1)
            trunk_in: (B, latent_dim + 3)  -> [z, t/T, s, ds]
            dz:       (B, latent_dim)
    """

    def __init__(self, latent_dim: int, theta_dim: int = 6,
                 branch_hidden: int = 128, branch_layers: int = 2,
                 trunk_width: int = 512, trunk_depth: int = 5,
                 basis_dim: int = 128):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.basis_dim = int(basis_dim)
        # Branch: GRU(ramp_seq concatenated with broadcast theta) -> coeffs.
        self.branch_gru = nn.GRU(input_size=1 + int(theta_dim),
                                 hidden_size=int(branch_hidden),
                                 num_layers=int(branch_layers),
                                 batch_first=True)
        self.branch_head = nn.Linear(int(branch_hidden), int(latent_dim) * int(basis_dim))
        # Trunk: identical to S4DBranchDeepONet ([z, t/T, s, ds] -> coeffs).
        trunk_layers: list[nn.Module] = [
            nn.Linear(int(latent_dim) + 3, int(trunk_width)),
            nn.LayerNorm(int(trunk_width)),
            nn.SiLU(),
        ]
        for _ in range(int(trunk_depth) - 2):
            trunk_layers += [nn.Linear(int(trunk_width), int(trunk_width)),
                             nn.LayerNorm(int(trunk_width)),
                             nn.SiLU()]
        trunk_layers.append(nn.Linear(int(trunk_width), int(latent_dim) * int(basis_dim)))
        self.trunk = nn.Sequential(*trunk_layers)
        self.bias = nn.Parameter(torch.zeros(int(latent_dim)))

    def forward(self, branch_in, trunk_in: torch.Tensor) -> torch.Tensor:
        theta, ramp_seq = branch_in
        B, Tr, _ = ramp_seq.shape
        th_exp = theta.unsqueeze(1).expand(B, Tr, theta.shape[-1])
        gru_in = torch.cat([ramp_seq, th_exp], dim=-1)               # (B, T_ramp, 1+theta_dim)
        _, h = self.branch_gru(gru_in)
        hb = self.branch_head(h[-1]).view(-1, self.latent_dim, self.basis_dim)
        ht = self.trunk(trunk_in).view(-1, self.latent_dim, self.basis_dim)
        return (hb * ht).sum(dim=-1) + self.bias


class MambaBranchDeepONet(nn.Module):
    """Mamba-branch DeepONet: selective-scan variant of S4DBranchDeepONet.

    Identical interface to S4DBranchDeepONet; replaces S4DLayer with mamba_ssm.Mamba.
    Mamba performs an input-dependent state-space scan (A, B, C selected per token),
    whereas S4D uses a fixed A. Used with torch 2.10 (cu130) and mamba-ssm 2.3.1.

    CUDA-only: the Mamba kernel requires a GPU and fp32 or bf16.
    """

    def __init__(self, latent_dim: int, theta_dim: int = 6,
                 branch_width: int = 128, branch_n_layers: int = 4,
                 mamba_d_state: int = 16, mamba_d_conv: int = 4, mamba_expand: int = 2,
                 trunk_width: int = 512, trunk_depth: int = 5,
                 basis_dim: int = 128):
        super().__init__()
        from mamba_ssm import Mamba
        self.latent_dim = int(latent_dim)
        self.basis_dim = int(basis_dim)
        # Branch
        self.theta_proj = nn.Linear(int(theta_dim), int(branch_width))
        self.ramp_proj = nn.Linear(1, int(branch_width))
        self.mamba_layers = nn.ModuleList([
            Mamba(d_model=int(branch_width),
                  d_state=int(mamba_d_state),
                  d_conv=int(mamba_d_conv),
                  expand=int(mamba_expand))
            for _ in range(int(branch_n_layers))
        ])
        self.branch_pool = nn.AdaptiveAvgPool1d(1)
        self.branch_out = nn.Linear(int(branch_width) * 2, int(latent_dim) * int(basis_dim))
        # Trunk (identical to the S4D variant)
        trunk_layers: list[nn.Module] = [
            nn.Linear(int(latent_dim) + 3, int(trunk_width)),
            nn.LayerNorm(int(trunk_width)),
            nn.SiLU(),
        ]
        for _ in range(int(trunk_depth) - 2):
            trunk_layers += [nn.Linear(int(trunk_width), int(trunk_width)),
                             nn.LayerNorm(int(trunk_width)),
                             nn.SiLU()]
        trunk_layers.append(nn.Linear(int(trunk_width), int(latent_dim) * int(basis_dim)))
        self.trunk = nn.Sequential(*trunk_layers)
        self.bias = nn.Parameter(torch.zeros(int(latent_dim)))

    def forward(self, branch_in, trunk_in: torch.Tensor) -> torch.Tensor:
        theta, ramp_seq = branch_in
        x = self.ramp_proj(ramp_seq)
        for layer in self.mamba_layers:
            x = layer(x)
        h_pool = self.branch_pool(x.transpose(1, 2)).squeeze(-1)
        hb = self.branch_out(torch.cat([h_pool, self.theta_proj(theta)], dim=1))
        hb = hb.view(-1, self.latent_dim, self.basis_dim)
        ht = self.trunk(trunk_in).view(-1, self.latent_dim, self.basis_dim)
        return (hb * ht).sum(dim=-1) + self.bias


def build_dynamics(name: str, *, latent_dim: int, theta_dim: int, hp: dict) -> nn.Module:
    """Instantiate an AR dynamics model by name; hyperparameters come from hp (cfg["model"])."""
    name = str(name).strip().lower()
    if name == "ar_mlp":
        return ARMlpStepper(latent_dim, theta_dim, hp.get("mlp_hidden", [128, 128]),
                            activation=hp.get("activation", "silu"), dropout=float(hp.get("dropout", 0.0)))
    if name == "s4d_branch_don":
        return S4DBranchDeepONet(
            latent_dim, int(hp.get("theta_dim", theta_dim)),
            branch_width=int(hp.get("branch_width", 128)),
            branch_n_layers=int(hp.get("branch_n_layers", 4)),
            branch_d_state=int(hp.get("branch_d_state", 64)),
            dt_min=float(hp.get("dt_min", 0.0001)),
            dt_max=float(hp.get("dt_max", 0.01)),
            trunk_width=int(hp.get("trunk_width", 512)),
            trunk_depth=int(hp.get("trunk_depth", 5)),
            basis_dim=int(hp.get("basis_dim", 128)),
        )
    if name == "mamba_branch_don":
        return MambaBranchDeepONet(
            latent_dim, int(hp.get("theta_dim", theta_dim)),
            branch_width=int(hp.get("branch_width", 128)),
            branch_n_layers=int(hp.get("branch_n_layers", 4)),
            mamba_d_state=int(hp.get("mamba_d_state", 16)),
            mamba_d_conv=int(hp.get("mamba_d_conv", 4)),
            mamba_expand=int(hp.get("mamba_expand", 2)),
            trunk_width=int(hp.get("trunk_width", 512)),
            trunk_depth=int(hp.get("trunk_depth", 5)),
            basis_dim=int(hp.get("basis_dim", 128)),
        )
    if name == "gru_branch_don":
        return GRUBranchDeepONet(
            latent_dim, int(hp.get("theta_dim", theta_dim)),
            branch_hidden=int(hp.get("branch_hidden", 128)),
            branch_layers=int(hp.get("branch_layers", 2)),
            trunk_width=int(hp.get("trunk_width", 512)),
            trunk_depth=int(hp.get("trunk_depth", 5)),
            basis_dim=int(hp.get("basis_dim", 128)),
        )
    raise ValueError(f"unknown dynamics {name!r}")


class LatentOneShotDON(nn.Module):
    """Latent one-shot Branch.Trunk DeepONet: Branch(theta, ramp_seq) . Trunk(t) -> z(t).

    The ENTIRE latent trajectory z(t) is produced in ONE forward (no state, no AR
    feedback). The branch reads the ramp sequence (+theta) into `basis_dim` coefficients
    per latent dimension; the trunk reads ONLY the scalar time t into a `basis_dim` basis;
    the dot-product over the basis yields z at every queried t. This is the latent x
    one-shot cell of the model matrix.

    branch_type:
      'gru' -> nn.GRU(1+theta_dim, branch_hidden, branch_layers) + Linear head,
               structurally identical to the KvsFieldDeepONet branch. Pairs with the
               field DeepONet along the representation axis (latent vs field).
      's4d' -> the 4xS4D stack (identical to S4DBranchDeepONet's branch). Pairs with
               the latent-AR rows along the time axis (one-shot vs AR).
               gru vs s4d isolates the branch.

    Interface:
        forward(theta, ramp_seq, t_query) -> z
            theta:    (B, theta_dim)
            ramp_seq: (B, T_ramp, 1)
            t_query:  (T,) or (T, 1)   -- normalised time in [0, 1]
            z:        (B, T, latent_dim)
    """

    def __init__(self, latent_dim: int, theta_dim: int = 6, branch_type: str = "gru",
                 basis_dim: int = 128,
                 # GRU-branch hyperparams (same construction as the field DeepONet branch)
                 branch_hidden: int = 128, branch_layers: int = 2,
                 # S4D-branch hyperparams (identical to S4DBranchDeepONet)
                 branch_width: int = 128, branch_n_layers: int = 4, branch_d_state: int = 64,
                 dt_min: float = 0.0001, dt_max: float = 0.01,
                 # trunk (t -> basis)
                 trunk_width: int = 256, trunk_depth: int = 4):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.basis_dim = int(basis_dim)
        self.branch_type = str(branch_type).strip().lower()
        out_coeff = self.latent_dim * self.basis_dim
        if self.branch_type == "gru":
            self.branch_gru = nn.GRU(input_size=1 + int(theta_dim),
                                     hidden_size=int(branch_hidden),
                                     num_layers=int(branch_layers),
                                     batch_first=True)
            self.branch_head = nn.Linear(int(branch_hidden), out_coeff)
        elif self.branch_type == "s4d":
            self.theta_proj = nn.Linear(int(theta_dim), int(branch_width))
            self.ramp_proj = nn.Linear(1, int(branch_width))
            self.s4d_layers = nn.ModuleList([
                S4DLayer(int(branch_width), d_state=int(branch_d_state),
                         dt_min=float(dt_min), dt_max=float(dt_max))
                for _ in range(int(branch_n_layers))
            ])
            self.branch_pool = nn.AdaptiveAvgPool1d(1)
            self.branch_out = nn.Linear(int(branch_width) * 2, out_coeff)
        else:
            raise ValueError(f"branch_type must be 'gru'|'s4d', got {branch_type!r}")
        # Trunk: scalar t -> (latent_dim * basis_dim) basis coefficients.
        trunk_layers: list[nn.Module] = [nn.Linear(1, int(trunk_width)), nn.SiLU()]
        for _ in range(int(trunk_depth) - 1):
            trunk_layers += [nn.Linear(int(trunk_width), int(trunk_width)), nn.SiLU()]
        trunk_layers.append(nn.Linear(int(trunk_width), out_coeff))
        self.trunk = nn.Sequential(*trunk_layers)
        self.bias = nn.Parameter(torch.zeros(self.latent_dim))

    def _branch(self, theta: torch.Tensor, ramp_seq: torch.Tensor) -> torch.Tensor:
        if self.branch_type == "gru":
            B, Tr, _ = ramp_seq.shape
            th_exp = theta.unsqueeze(1).expand(B, Tr, theta.shape[-1])
            gru_in = torch.cat([ramp_seq, th_exp], dim=-1)            # (B, T_ramp, 1+theta_dim)
            _, h = self.branch_gru(gru_in)
            hb = self.branch_head(h[-1])                              # (B, latent*basis)
        else:
            x = self.ramp_proj(ramp_seq)
            for layer in self.s4d_layers:
                x = layer(x)
            h_pool = self.branch_pool(x.transpose(1, 2)).squeeze(-1)
            hb = self.branch_out(torch.cat([h_pool, self.theta_proj(theta)], dim=1))
        return hb.view(-1, self.latent_dim, self.basis_dim)          # (B, d, p)

    def forward(self, theta: torch.Tensor, ramp_seq: torch.Tensor,
                t_query: torch.Tensor) -> torch.Tensor:
        if t_query.dim() == 1:
            t_query = t_query.unsqueeze(-1)                           # (T, 1)
        hb = self._branch(theta, ramp_seq)                           # (B, d, p)
        ht = self.trunk(t_query).view(-1, self.latent_dim, self.basis_dim)  # (T, d, p)
        return torch.einsum("bdp,tdp->btd", hb, ht) + self.bias      # (B, T, d)
