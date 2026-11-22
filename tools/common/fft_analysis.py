# ============================================================================
# Dominant-frequency peak of a sampled signal.
# Pipeline role: fft_peak() is the frequency estimator of the Strouhal error
#   in benchmark/metrics_kvs.py. This module applies no time window and no
#   component selection; metrics_kvs.py passes the v component for
#   t >= 2.0 s (steps 100..200).
# ============================================================================
"""Dominant-frequency extraction for the Strouhal error."""

from __future__ import annotations

import math
from typing import Optional

import numpy as np


# Frequency estimator of the Strouhal error (benchmark/metrics_kvs.py, _fft_freq_err).
def fft_peak(
    signal: np.ndarray,
    dt: float,
    fmin: float = 0.0,
    fmax: Optional[float] = None,
    pad_factor: int = 4,
) -> float:
    """Find the dominant FFT frequency of *signal* within [fmin, fmax].

    The signal is de-meaned and Hann-windowed; zero-padding and parabolic
    interpolation of the log power around the peak give sub-bin accuracy.

    Parameters
    ----------
    signal : 1-D array
    dt : sampling interval
    fmin, fmax : frequency band (Hz). *fmax=None* means no upper bound.
    pad_factor : zero-padding factor (default 4x). Higher = finer frequency grid.

    Returns
    -------
    Dominant frequency (Hz), or NaN if undetermined.
    """
    sig = np.asarray(signal, dtype=np.float64).ravel()
    if sig.size < 4 or not math.isfinite(dt) or dt <= 0.0:
        return float("nan")
    sig = sig - np.mean(sig)
    win = np.hanning(sig.size)
    n_padded = sig.size * pad_factor
    freq = np.fft.rfftfreq(n_padded, d=float(dt))
    psd = np.abs(np.fft.rfft(sig * win, n=n_padded)) ** 2
    mask = freq >= float(fmin)
    if fmax is not None:
        mask &= freq <= float(fmax)
    if not np.any(mask):
        return float("nan")

    # Find peak bin in masked region
    masked_psd = psd[mask]
    masked_freq = freq[mask]
    peak_local = int(np.argmax(masked_psd))

    # Parabolic interpolation around peak for sub-bin accuracy
    if 0 < peak_local < len(masked_psd) - 1:
        alpha = float(np.log(masked_psd[peak_local - 1] + 1e-30))
        beta = float(np.log(masked_psd[peak_local] + 1e-30))
        gamma = float(np.log(masked_psd[peak_local + 1] + 1e-30))
        denom = alpha - 2 * beta + gamma
        if abs(denom) > 1e-12:
            delta = 0.5 * (alpha - gamma) / denom  # fractional bin offset
            df = float(masked_freq[1] - masked_freq[0]) if len(masked_freq) > 1 else 0.0
            return float(masked_freq[peak_local] + delta * df)

    return float(masked_freq[peak_local])

