# =====================================================================
# Diagonal S4D state-space layer in pure PyTorch (no custom CUDA kernel).
# Pipeline role: the branch block of S4DBranchDeepONet and of
#   LatentOneShotDON(branch_type='s4d') in
#   benchmark/adapters/_dynamics_models.py; used by the S4D rows of the KVS
#   and CMP tables.
# Discretisation: exact zero-order hold (ZOH), A_d = exp(dt*A),
#   B_d = (A_d - 1)/A * B. With this form the DC gain of the discrete
#   system equals that of the continuous system, independent of dt.
# =====================================================================
"""Diagonal state-space model (S4D) layer.

Pure-PyTorch implementation of S4D (diagonal state-space model). No custom CUDA kernels.

References
----------
- S4D: "On the Parameterization and Initialization of Diagonal State Space Models"
  Gu et al. 2022 (https://arxiv.org/abs/2206.11893)
- S4:  "Efficiently Modeling Long Sequences with Structured State Spaces"
  Gu et al. 2021 (https://arxiv.org/abs/2111.00396)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# S4D kernel (diagonal, pure PyTorch)
# ---------------------------------------------------------------------------

class S4DKernel(nn.Module):
    """Diagonal S4 convolution kernel.

    Computes the SSM convolution filter K = (CB, CAB, CA^2B, ...) for
    a diagonal A matrix. Builds K as a Vandermonde product and applies it
    by FFT convolution.

    State: z_t = A z_{t-1} + B x_t
    Output: y_t = Re(C z_t) + D x_t

    A is diagonal with complex eigenvalues: a_n = exp(-dt * (alpha_n + i*beta_n))
    where alpha_n > 0 ensures stability.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.d_state = int(d_state)

        # Log step size (learnable), shape (d_model,)
        log_dt = torch.rand(d_model) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        self.log_dt = nn.Parameter(log_dt)

        # S4D-Lin initialisation: imaginary part = pi * n.
        # Real part of A is -exp(A_real) < 0 (stable); A_real starts at -0.5.
        # Difference to S4D-Lin, whose real part starts at -0.5: here it starts at
        # -exp(-0.5) = -0.61. A is learned, the difference concerns the initial value only.
        n = torch.arange(d_state, dtype=torch.float32)
        A_real = -0.5 * torch.ones(d_model, d_state)
        A_imag = math.pi * n.unsqueeze(0).expand(d_model, -1)
        self.A_real = nn.Parameter(A_real)          # (d_model, d_state)
        self.A_imag = nn.Parameter(A_imag)          # (d_model, d_state)

        # B, C: complex, shape (d_model, d_state)
        self.B_real = nn.Parameter(torch.randn(d_model, d_state) * 0.01)
        self.B_imag = nn.Parameter(torch.randn(d_model, d_state) * 0.01)
        self.C_real = nn.Parameter(torch.randn(d_model, d_state) * 0.01)
        self.C_imag = nn.Parameter(torch.randn(d_model, d_state) * 0.01)

        # D: skip connection (feedthrough)
        self.D = nn.Parameter(torch.ones(d_model))

    def forward_conv(self, u: torch.Tensor, L: int) -> torch.Tensor:
        """Compute SSM output via convolution (training, parallel over time).

        Parameters
        ----------
        u : Tensor (B, L, d_model)
        L : int, sequence length

        Returns
        -------
        y : Tensor (B, L, d_model)
        """
        dt = torch.exp(self.log_dt)  # (d_model,)
        A = torch.complex(-torch.exp(self.A_real), self.A_imag)  # (d_model, d_state)
        B = torch.complex(self.B_real, self.B_imag)               # (d_model, d_state)
        C = torch.complex(self.C_real, self.C_imag)               # (d_model, d_state)

        # Discretise (exact ZOH): A_d = exp(dt*A), B_d = A^{-1} (A_d - I) B.
        # The DC gain of the discrete system then equals that of the
        # continuous one and does not depend on dt.
        dtA = dt.unsqueeze(-1) * A  # (d_model, d_state)
        A_d = torch.exp(dtA)        # (d_model, d_state)
        B_d = (A_d - 1.0) / A * B   # (d_model, d_state)

        # Vandermonde convolution kernel: K[l] = 2 * Re(C * A_d^l * B_d)
        # shape: (d_model, L)
        l = torch.arange(L, device=u.device, dtype=torch.float32)  # (L,)
        # A_d^l: (d_model, d_state, L)
        powers = A_d.unsqueeze(-1) ** l.unsqueeze(0).unsqueeze(0)
        K = 2.0 * (C.unsqueeze(-1) * B_d.unsqueeze(-1) * powers).real.sum(dim=1)  # (d_model, L)

        # Causal convolution via FFT
        K_f = torch.fft.rfft(K, n=2 * L, dim=-1)             # (d_model, L+1)
        u_t = u.transpose(1, 2)                                # (B, d_model, L)
        u_f = torch.fft.rfft(u_t, n=2 * L, dim=-1)
        y_f = K_f.unsqueeze(0) * u_f                          # (B, d_model, L+1)
        y = torch.fft.irfft(y_f, n=2 * L, dim=-1)[..., :L]   # (B, d_model, L)
        y = y.transpose(1, 2)                                  # (B, L, d_model)

        # Skip connection
        y = y + u * self.D.unsqueeze(0).unsqueeze(0)
        return y


class S4DLayer(nn.Module):
    """S4D layer: SSM + LayerNorm + residual + optional dropout."""

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        dropout: float = 0.0,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
    ) -> None:
        super().__init__()
        self.ssm = S4DKernel(d_model, d_state, dt_min, dt_max)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(p=float(dropout)) if dropout > 0.0 else nn.Identity()

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """u: (B, L, d_model) -> (B, L, d_model)"""
        L = u.shape[1]
        y = self.ssm.forward_conv(u, L)
        return self.drop(self.norm(y + u))
