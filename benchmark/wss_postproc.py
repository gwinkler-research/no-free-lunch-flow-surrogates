# ============================================================================
# Purpose: cumulative wall shear stress (cum-WSS) from the CMP velocity on two
#   z-planes. This is the post-processing step behind every CMP cum-WSS metric.
# Pipeline role: used by scripts/run_cmp3d_gate2.py and
#   benchmark/cmp_wss_metrics.py, identically for prediction and truth.
# Convention: lab frame, WSS = mu * (u_top - u_bot) / dz (finite difference
#   across the two planes). No radial or angular averaging happens here; the
#   radial binning is in cmp_wss_metrics.radial_profile. mu is a placeholder
#   value; relative metrics do not depend on it because it scales prediction
#   and truth alike.
# ============================================================================
"""CMP post-processing: velocity field -> cumulative wall shear stress (cum-WSS).

WSS(x, t) = mu * (u_top - u_bot) / dz, from the two z-planes z = 0 and z = 4e-5 m of the CMP
data; cum_wss(x) = integral_t |WSS(x, t)| dt, evaluated as sum_t |WSS| * dt with dt the median
time step. Applied identically to prediction and truth, so relative errors do not depend on mu.
"""
from __future__ import annotations

import numpy as np

DZ_DEFAULT = 4.0e-5      # distance [m] between the two planes, z = 0 and z = 4e-5
MU_DEFAULT = 1.0e-3      # placeholder dynamic viscosity [Pa s]; relative metrics do not depend on it


# Time integral of |WSS| per node (sum * dt); applied identically to prediction and truth.
def velocity_to_cum_wss(vel_fem, t_grid, *, mu=MU_DEFAULT, dz=DZ_DEFAULT):
    """vel_fem: (T, n_fem, 4), channels [u_x_top, u_y_top, u_x_bot, u_y_bot]. Returns cum_wss (n_fem,)."""
    vel = np.asarray(vel_fem, dtype=np.float64)
    if vel.ndim != 3 or vel.shape[-1] != 4:
        raise ValueError(f"vel_fem must have shape (T, n_fem, 4), got {vel.shape}")
    wss_vec = mu * (vel[..., 0:2] - vel[..., 2:4]) / dz       # (T, n_fem, 2)
    wss_mag = np.linalg.norm(wss_vec, axis=-1)                # (T, n_fem)
    dt = float(np.median(np.diff(np.asarray(t_grid, dtype=np.float64))))
    return np.sum(wss_mag, axis=0) * dt                       # (n_fem,)
