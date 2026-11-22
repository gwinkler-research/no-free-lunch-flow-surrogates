# ============================================================================
# Purpose: network definition of the KVS field DeepONet. A GRU branch reads
#   the control sequence together with theta and emits one coefficient vector
#   per output channel; an MLP trunk reads the normalised coordinates
#   (x, y, t) and emits a basis vector; the prediction is their dot product
#   plus a per-channel bias. No encoder, no autoregression, no file I/O, no
#   training loop.
# Pipeline role: built only by benchmark/adapters/kvs_field_deeponet.py, which
#   owns data handling and training. It is the direct x one-shot cell of the
#   KVS model matrix. Its GRU branch has the same construction as the GRU
#   branch of the latent one-shot model, so the direct-versus-latent
#   comparison differs in the representation, not in the branch.
# Eval-consistency: geometry-free. It receives coordinates and time already
#   normalised and computes or stores no bbox, mean or std. Weight init
#   follows the torch seed set before the model is built.
# ============================================================================
"""KVS field DeepONet (GRU branch x 3-dim trunk -> u, v, p).

Direct branch-trunk map, without encoder and without autoregression. The branch has the
same GRU design as the CMP field DeepONet (tools/models/cmp3d_sdeeponet.py); only the
trunk differs: CMP uses wafer-relative spatial features, KVS uses (x_n, y_n, t_n).

The model is geometry-free: it receives already-normalised coordinates and t_norm; the
bbox scaling is done in the adapter.
"""
from __future__ import annotations

import torch
import torch.nn as nn

C = 3  # KVS channels: u, v, p


class KvsFieldDeepONet(nn.Module):
    """Branch (GRU + theta) x trunk (MLP on normalised x, y, t) -> n-channel dot product.

    The trunk input is (x_n, y_n, t_n), 3-dim and already normalised. The
    branch is GRU(1 + theta_dim) followed by a linear branch_head ->
    latent_dim * n_channels. The trunk is a SiLU MLP
    3 -> trunk_hidden (trunk_layers layers) -> latent_dim.

    Branch and trunk are evaluated separately, so the model can be queried on
    any set of xy points (the adapter calls forward twice: P2 points for u, v
    and P1 points for p).
    """

    def __init__(self, theta_dim: int = 6, branch_hidden: int = 128,
                 branch_layers: int = 2, latent_dim: int = 128,
                 trunk_hidden: int = 256, trunk_layers: int = 4,
                 n_channels: int = C):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_channels = n_channels

        # Branch: same structure as in the CMP field DeepONet (SDeepONet3D).
        self.branch_gru = nn.GRU(
            input_size=1 + theta_dim,
            hidden_size=branch_hidden,
            num_layers=branch_layers,
            batch_first=True,
        )
        self.branch_head = nn.Linear(branch_hidden, latent_dim * n_channels)

        # Trunk: 3-dim (x_n, y_n, t_n), already normalized by the adapter.
        layers = [nn.Linear(3, trunk_hidden), nn.SiLU()]
        for _ in range(trunk_layers - 1):
            layers += [nn.Linear(trunk_hidden, trunk_hidden), nn.SiLU()]
        layers += [nn.Linear(trunk_hidden, latent_dim)]
        self.trunk = nn.Sequential(*layers)

        self.bias = nn.Parameter(torch.zeros(n_channels))

    @property
    def n_params(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _trunk_features(self, xy: torch.Tensor, t_norm: torch.Tensor,
                        B: int, T: int) -> torch.Tensor:
        """Build the (B, T, M, 3) trunk input from normalized xy + t_norm.

        xy:     (B, M, 2) or (M, 2), already-normalised query coords.
        t_norm: (T,) or (B, T), already-normalised time.
        Returns: (B, T, M, 3) with channels (x_n, y_n, t_n).
        """
        if xy.dim() == 2:
            xy = xy.unsqueeze(0).expand(B, -1, -1)
        M = xy.shape[1]
        # (B, T, M, 2)
        xy_e = xy.unsqueeze(1).expand(B, T, M, 2)
        if t_norm.dim() == 1:
            t_norm = t_norm.unsqueeze(0).expand(B, -1)
        # (B, T, M, 1)
        t_e = t_norm.unsqueeze(-1).unsqueeze(-1).expand(B, T, M, 1)
        return torch.cat([xy_e, t_e], dim=-1)

    def forward(self, s_seq: torch.Tensor, theta: torch.Tensor,
                xy: torch.Tensor, t_norm: torch.Tensor) -> torch.Tensor:
        """
        s_seq:  (B, T, 1)
        theta:  (B, theta_dim)
        xy:     (B, M, 2) or (M, 2), already-normalised query points
        t_norm: (T,) or (B, T), already-normalised time
        Returns: (B, T, M, n_channels)
        """
        B, T, _ = s_seq.shape
        theta_seq = theta.unsqueeze(1).expand(-1, T, -1)
        gru_in = torch.cat([s_seq, theta_seq], dim=-1)
        gru_out, _ = self.branch_gru(gru_in)
        b = self.branch_head(gru_out)
        b = b.view(B, T, self.n_channels, self.latent_dim)

        feat = self._trunk_features(xy, t_norm, B, T)   # (B, T, M, 3)
        t = self.trunk(feat)                            # (B, T, M, latent)
        out = torch.einsum("btcd,btmd->btmc", b, t)     # (B, T, M, C)
        out = out + self.bias
        return out
