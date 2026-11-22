# 2D convolutional autoencoder for KVS full-field snapshots (u, v, p) on the
#   83 x 442 grid.
# Pipeline role: the frozen encoder E and decoder D of the KVS latent chain
#   E -> F (latent dynamics) -> D. Trained by benchmark/adapters/grid_convae.py;
#   the paper configs use base_channels=32 and latent_dim=32.
# Every KVS latent model (AR-MLP, DMDc, the S4D/Mamba/GRU branch DeepONets and
#   the latent one-shot models) decodes through it, so its encode/decode round
#   trip is the accuracy floor of all of them.
"""Convolutional autoencoder for KVS full-field data.

Operates on the 2D grid (H=83, W=442, C=3 for u, v, p). The cylinder interior is filled
by interpolation when the grid is built, so the reconstruction loss runs over every grid
cell without a mask. The encoder downsamples with strided convolutions; the decoder uses
nearest-neighbour upsampling followed by a convolution.

Architecture:
  grid (83, 442, 3) -> zero-pad to a multiple of 2^n_downsample -> Conv2d blocks
  (one with stride 1, then n_downsample with stride 2) -> flatten -> Linear -> z in R^d
  z in R^d -> Linear + SiLU -> unflatten -> (upsample + Conv2d) blocks -> Conv2d
  -> crop -> grid (83, 442, 3)
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Conv2d + BatchNorm + activation + optional stride for downsampling."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride,
                              padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class UpsampleBlock(nn.Module):
    """Upsample (nearest) + Conv2d + BatchNorm + activation.

    Matches output sizes more reliably than ConvTranspose2d.
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.act(self.bn(self.conv(x)))


class ConvAutoencoder(nn.Module):
    """2D convolutional autoencoder for KVS full-field snapshots.

    Parameters
    ----------
    in_channels : int
        Number of input fields (3 for u, v, p).
    latent_dim : int
        Latent vector dimension.
    base_channels : int
        Channels after first conv layer (doubled at each downsampling).
    n_downsample : int
        Number of stride-2 downsampling steps (3: padded 88 x 448 -> bottleneck 11 x 56).
    grid_h, grid_w : int
        Input grid dimensions (83, 442).
    """

    def __init__(
        self,
        *,
        in_channels: int = 3,
        latent_dim: int = 16,
        base_channels: int = 32,
        n_downsample: int = 3,
        grid_h: int = 83,
        grid_w: int = 442,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.n_downsample = n_downsample

        # Pad input to make dimensions divisible by 2^n_downsample
        factor = 2 ** n_downsample
        self.pad_h = (factor - grid_h % factor) % factor
        self.pad_w = (factor - grid_w % factor) % factor
        padded_h = grid_h + self.pad_h
        padded_w = grid_w + self.pad_w

        # ---- Encoder ----
        enc_layers = []
        # Initial conv (no downsampling)
        enc_layers.append(ConvBlock(in_channels, base_channels, stride=1))

        ch = base_channels
        for i in range(n_downsample):
            ch_next = ch * 2
            enc_layers.append(ConvBlock(ch, ch_next, stride=2))
            ch = ch_next

        self.encoder_conv = nn.Sequential(*enc_layers)

        # Compute bottleneck spatial dims
        self.bottleneck_h = padded_h // factor
        self.bottleneck_w = padded_w // factor
        self.bottleneck_ch = ch
        bottleneck_flat = ch * self.bottleneck_h * self.bottleneck_w

        self.encoder_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(bottleneck_flat, latent_dim),
        )

        # ---- Decoder ----
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, bottleneck_flat),
            nn.SiLU(),
        )

        dec_layers = []
        for i in range(n_downsample):
            ch_next = ch // 2
            dec_layers.append(UpsampleBlock(ch, ch_next))
            ch = ch_next

        # Final conv to output channels (no activation, raw output)
        dec_layers.append(nn.Conv2d(ch, in_channels, kernel_size=3, padding=1))
        self.decoder_conv = nn.Sequential(*dec_layers)

    def _pad(self, x: torch.Tensor) -> torch.Tensor:
        """Pad input to make dims divisible by 2^n_downsample."""
        if self.pad_h > 0 or self.pad_w > 0:
            x = F.pad(x, (0, self.pad_w, 0, self.pad_h), mode="constant", value=0)
        return x

    def _unpad(self, x: torch.Tensor) -> torch.Tensor:
        """Crop to original grid size (handles both padding and upsample overshoot)."""
        return x[:, :, :self.grid_h, :self.grid_w]

    # Encoder entry point; the latent adapters in benchmark/adapters/ use it with
    # frozen weights.
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) -> (B, latent_dim)"""
        x = self._pad(x)
        h = self.encoder_conv(x)
        return self.encoder_fc(h)

    # Decoder entry point; decode(encode(x)) is the round trip that sets the
    # accuracy floor named in the module header.
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """(B, latent_dim) -> (B, C, H, W)"""
        h = self.decoder_fc(z)
        h = h.view(-1, self.bottleneck_ch, self.bottleneck_h, self.bottleneck_w)
        h = self.decoder_conv(h)
        return self._unpad(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct. (B, C, H, W) -> (B, C, H, W)"""
        return self.decode(self.encode(x))

    def losses(
        self,
        x_seq: torch.Tensor,
        mask: torch.Tensor | None = None,
        lambda_recon: float = 1.0,
        lambda_temporal: float = 0.1,
    ) -> Dict[str, torch.Tensor]:
        """Compute the reconstruction and temporal-smoothness losses.

        Each frame is encoded and decoded on its own. The temporal term is the
        mean squared difference of consecutive latent vectors (zero for T = 1).

        Parameters
        ----------
        x_seq : (B, T, C, H, W)
            Sequence of grid snapshots.
        mask : (H, W) or None
            Boolean mask of the fluid region. If given, the reconstruction loss
            uses fluid points only.
        lambda_recon, lambda_temporal : float
            Loss weights.

        Returns
        -------
        dict with "recon", "temporal" and "total" = lambda_recon * recon
        + lambda_temporal * temporal.
        """
        B, T, C, H, W = x_seq.shape
        x_in = x_seq.reshape(B * T, C, H, W)
        target = x_in
        Tprime = T

        z_flat = self.encode(x_in)
        x_hat = self.decode(z_flat)

        if mask is not None:
            mask_4d = mask.unsqueeze(0).unsqueeze(0)
            recon = F.mse_loss(x_hat * mask_4d, target * mask_4d)
        else:
            recon = F.mse_loss(x_hat, target)

        z_seq = z_flat.view(B, Tprime, -1)
        if Tprime > 1:
            dz = z_seq[:, 1:, :] - z_seq[:, :-1, :]
            temporal = (dz ** 2).mean()
        else:
            temporal = torch.zeros((), device=x_seq.device, dtype=z_flat.dtype)

        total = lambda_recon * recon + lambda_temporal * temporal
        return {"recon": recon, "temporal": temporal, "total": total}
