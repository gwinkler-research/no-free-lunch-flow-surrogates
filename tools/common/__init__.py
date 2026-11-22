"""Shared utility functions: FFT peak analysis (fft_analysis), seeding (seeding) and the
ratio-based case split (splitting).

Explicit submodule imports are preferred::

    from tools.common.seeding import set_seed
    from tools.common.splitting import split_runs
    from tools.common.fft_analysis import fft_peak
"""

# Re-exported for convenience.
from .seeding import set_seed
from .splitting import split_runs

__all__ = [
    "set_seed",
    "split_runs",
]
