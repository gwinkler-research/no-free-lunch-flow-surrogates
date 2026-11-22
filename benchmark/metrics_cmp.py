# ============================================================================
# Purpose: the inner-90 node subset of the wafer (r/R <= 0.9 from the wafer
#   centre) and its reference radius R.
# Pipeline role: scripts/run_cmp3d_gate2.py imports _inner90_mask and
#   WAFER_RADIUS_DEFAULT for the inner-90 subset of the structure metrics
#   (benchmark/cmp_wss_metrics.py).
# ============================================================================
"""Inner-90 node mask of the CMP wafer at the eval FEM nodes.

inner-90 keeps the wafer nodes with r/R <= 0.9 from the wafer centre (R = wafer_radius). It is
the primary node subset because the reference solution is least certain near the wafer edge.
"""
from __future__ import annotations

import numpy as np

# Reference wafer radius R [m] of the inner-90 mask (0.9 R). 0.149 m is the radius of the
# outermost FEM node inside the wafer disk (top-plane node spacing about 3.5 mm); the
# simulated wafer has R = 0.15 m (300 mm). The same R is applied to the truth and to every
# model, and the published metrics are defined with this value.
WAFER_RADIUS_DEFAULT = 0.149


def _radius_norm(fem_xy, wafer_center, wafer_radius):
    r = np.linalg.norm(np.asarray(fem_xy, dtype=np.float64) - np.asarray(wafer_center, dtype=np.float64)[None, :], axis=1)
    return r / float(wafer_radius)


# Inner-90 mask: wafer_mask and r/R <= 0.9.
def _inner90_mask(fem_xy, wafer_mask_case, wafer_center, wafer_radius):
    rn = _radius_norm(fem_xy, wafer_center, wafer_radius)
    return np.asarray(wafer_mask_case, dtype=bool) & (rn <= 0.9)
