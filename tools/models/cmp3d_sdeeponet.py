# CMP field DeepONet (S-DeepONet; direct, one-shot). A GRU branch over the control
#   sequence and theta and an MLP trunk over 6 spatial features (x, y, z/gap,
#   r/R, sin phi, cos phi around the wafer centre) are combined by a dot
#   product (einsum in forward()). It predicts the full 3D velocity
#   (u_x, u_y, u_z) directly, without a latent encoder.
# Pipeline role: direct x one-shot cell of the CMP model matrix. Built by
#   benchmark/adapters/cmp3d_sdeeponet.py and scored by
#   scripts/run_cmp3d_gate2.py.
"""CMP 3D coordinate-based S-DeepONet.

The trunk takes an (x, y, z) query point together with wafer-relative polar features and
z/gap, so the model predicts the 3D velocity (u_x, u_y, u_z) anywhere in the gap. The branch
is a GRU over the control sequence and theta. Mesh-free: trained on sampled points and
evaluable at any coordinate (e.g. the two WSS planes).
"""
from __future__ import annotations

import torch
import torch.nn as nn

C = 3
WAFER_RADIUS = 0.149  # m
GAP_Z = 4.0e-5        # m (z in [0, GAP_Z])


class SDeepONet3D(nn.Module):
    """Branch (GRU + theta) x trunk (MLP on lab, polar and z features) -> 3-channel dot product.

    Trunk input: (x, y, z/gap, r/R, sin(phi), cos(phi)), 6-dim: lab-frame (x, y),
    vertical z/gap and polar geometry relative to the wafer centre.
    forward() takes xyz and the per-case wafer centre and computes the polar
    features internally.
    """

    def __init__(self, theta_dim: int = 6, branch_hidden: int = 128, branch_layers: int = 2,
                 latent_dim: int = 128, trunk_hidden: int = 256, trunk_layers: int = 4,
                 n_channels: int = C, wafer_radius: float = WAFER_RADIUS, gap_z: float = GAP_Z):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_channels = n_channels
        self.wafer_radius = wafer_radius
        self.gap_z = gap_z

        self.branch_gru = nn.GRU(
            input_size=1 + theta_dim,
            hidden_size=branch_hidden,
            num_layers=branch_layers,
            batch_first=True,
        )
        self.branch_head = nn.Linear(branch_hidden, latent_dim * n_channels)

        # Trunk: 6-dim (x, y, z/gap, r/R, sin(phi), cos(phi))
        layers = [nn.Linear(6, trunk_hidden), nn.SiLU()]
        for _ in range(trunk_layers - 1):
            layers += [nn.Linear(trunk_hidden, trunk_hidden), nn.SiLU()]
        layers += [nn.Linear(trunk_hidden, latent_dim)]
        self.trunk = nn.Sequential(*layers)

        self.bias = nn.Parameter(torch.zeros(n_channels))

    # Polar geometry (r/R, sin/cos phi) around the wafer centre wc. wc places the
    # case's wafer offset at the trunk's polar origin; theta itself enters the
    # branch (see forward()).
    def _trunk_features(self, xyz: torch.Tensor, wc: torch.Tensor) -> torch.Tensor:
        """xyz: (B, M, 3) or (M, 3); wc: (B, 2). Returns (B, M, 6)."""
        if xyz.dim() == 2:
            xyz = xyz.unsqueeze(0).expand(wc.shape[0], -1, -1)
        xy = xyz[..., :2]
        z = xyz[..., 2:3]
        rel = xy - wc.unsqueeze(1)
        r = torch.sqrt(rel[..., 0] ** 2 + rel[..., 1] ** 2 + 1e-12)
        phi = torch.atan2(rel[..., 1], rel[..., 0])
        return torch.cat([
            xy,
            z / self.gap_z,
            (r / self.wafer_radius).unsqueeze(-1),
            torch.sin(phi).unsqueeze(-1),
            torch.cos(phi).unsqueeze(-1),
        ], dim=-1)

    def forward(self, s_seq: torch.Tensor, theta: torch.Tensor,
                xyz: torch.Tensor, wc: torch.Tensor) -> torch.Tensor:
        """s_seq:(B,T,1) theta:(B,theta_dim) xyz:(B,M,3)|(M,3) wc:(B,2) -> (B,T,M,3)."""
        B, T, _ = s_seq.shape
        theta_seq = theta.unsqueeze(1).expand(-1, T, -1)
        gru_in = torch.cat([s_seq, theta_seq], dim=-1)
        gru_out, _ = self.branch_gru(gru_in)
        b = self.branch_head(gru_out).view(B, T, self.n_channels, self.latent_dim)
        feat = self._trunk_features(xyz, wc)        # (B, M, 6)
        t = self.trunk(feat)                        # (B, M, latent)
        # Branch . trunk dot product: einsum contraction over the latent dim d
        # (not a concat-then-MLP merge).
        out = torch.einsum("btcd,bmd->btmc", b, t)  # (B, T, M, C)
        return out + self.bias
