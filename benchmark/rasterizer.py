# ============================================================================
# Purpose: linear interpolation between the FEM node cloud and a uniform
#   raster grid in both directions: grid_to_nodes, and a precomputed
#   barycentric operator (RasterWeights) for rasterising many frames on one
#   node set (920 cases x 201 frames for KVS).
# Pipeline role: builds the fixed-size grid (83 x 442 for KVS) used by the
#   ConvAE and the latent adapters; grid_to_nodes maps predicted grids back
#   onto the FEM eval nodes for the metrics.
# Eval-consistency: KVS_BBOX is the physical extent mapped onto the grid.
#   Training and evaluation must use the same bbox and grid_hw, otherwise
#   predictions land at the wrong coordinates without any shape error. The
#   83 x 442 grid keeps the aspect ratio of the bbox (about 5 mm spacing in
#   x and y).
# ============================================================================
"""Linear interpolation between FEM nodes and a uniform grid, in both directions.

grid_to_nodes: (H,W) or (H,W,C) grid -> (M,) or (M,C) at arbitrary nodes (bilinear; points
               outside the bbox are clipped to its boundary).
bbox = (xmin, ymin, xmax, ymax). Default: the KVS channel.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import Delaunay, cKDTree

# Physical extent (xmin, ymin, xmax, ymax) [m] of the KVS channel mapped onto the grid.
# Training and evaluation must use the same bbox.
KVS_BBOX = (0.0, 0.0, 2.2, 0.41)


def _grid_axes(grid_hw, bbox):
    xmin, ymin, xmax, ymax = bbox
    H, W = int(grid_hw[0]), int(grid_hw[1])
    return np.linspace(ymin, ymax, H), np.linspace(xmin, xmax, W)  # ys (H,), xs (W,)


def grid_to_nodes(grid, coords_xy, *, bbox=KVS_BBOX):
    g = np.asarray(grid, dtype=np.float64)
    squeeze = g.ndim == 2
    if squeeze:
        g = g[..., None]
    H, W, C = g.shape
    ys, xs = _grid_axes((H, W), bbox)
    xmin, ymin, xmax, ymax = bbox
    pts = np.asarray(coords_xy, dtype=np.float64).copy()
    pts[:, 0] = np.clip(pts[:, 0], xmin, xmax)
    pts[:, 1] = np.clip(pts[:, 1], ymin, ymax)
    query_yx = np.column_stack([pts[:, 1], pts[:, 0]])  # RegularGridInterpolator expects (y, x)
    out = np.empty((pts.shape[0], C), dtype=np.float64)
    for c in range(C):
        rgi = RegularGridInterpolator((ys, xs), g[..., c], method="linear", bounds_error=False, fill_value=None)
        out[:, c] = rgi(query_yx)
    return out[:, 0] if squeeze else out


@dataclass(frozen=True)
class RasterWeights:
    """Precomputed barycentric interpolation operator: FEM nodes -> uniform grid.

    For each of the H*W grid points: either (a) the index of the containing
    Delaunay simplex of `coords_xy` plus the 3 barycentric weights, or
    (b) -1 simplex + the index of the nearest source node (convex-hull fallback).

    apply: grid_value[g] = sum_k bary[g,k] * values[tri[simplex[g], k]]   (inside)
           grid_value[g] = values[nn_idx[g]]                              (fallback)
    """
    grid_hw: tuple
    n_nodes: int
    simplex: np.ndarray   # (H*W,) int64, -1 = use nn fallback
    tri: np.ndarray       # (n_simplex, 3) int64 vertex indices into coords_xy
    bary: np.ndarray      # (H*W, 3) float64 barycentric weights (zeros where simplex == -1)
    nn_idx: np.ndarray    # (H*W,) int64 nearest-node index (used where simplex == -1)


def precompute_raster_weights(coords_xy, grid_hw, bbox=KVS_BBOX) -> RasterWeights:
    """Build the Delaunay triangulation and barycentric weights once for a fixed
    node set and grid. O(N log N); reused for every frame (920 x 201 for KVS)."""
    # The weights are reused and not re-validated later: callers must pass the same
    # (coords_xy, grid_hw, bbox) at training and evaluation time.
    coords_xy = np.ascontiguousarray(np.asarray(coords_xy, dtype=np.float64))
    n_nodes = coords_xy.shape[0]
    H, W = int(grid_hw[0]), int(grid_hw[1])
    ys, xs = _grid_axes((H, W), bbox)
    gx, gy = np.meshgrid(xs, ys)            # (H, W)
    gpts = np.column_stack([gx.ravel(), gy.ravel()])   # (H*W, 2)

    n_gpts = gpts.shape[0]
    bary = np.zeros((n_gpts, 3), dtype=np.float64)
    try:
        dt = Delaunay(coords_xy)
        simplex = dt.find_simplex(gpts)         # (H*W,) -1 outside hull
        tri = np.ascontiguousarray(dt.simplices.astype(np.int64))   # (n_simplex, 3)
        inside = simplex >= 0
        if np.any(inside):
            si = simplex[inside]
            T = dt.transform[si]                # (m, 3, 2): first 2 rows = transform, last row = offset
            b = np.einsum("mij,mj->mi", T[:, :2, :], gpts[inside] - T[:, 2, :])  # (m, 2)
            bary[inside, :2] = b
            bary[inside, 2] = 1.0 - b.sum(axis=1)
    except Exception:
        # Degenerate point set (e.g. collinear): use nearest-neighbour everywhere
        simplex = np.full(n_gpts, -1, dtype=np.int64)
        tri = np.empty((0, 3), dtype=np.int64)

    nn_idx = cKDTree(coords_xy).query(gpts, k=1)[1].astype(np.int64)
    return RasterWeights(
        grid_hw=(H, W), n_nodes=int(n_nodes),
        simplex=simplex.astype(np.int64), tri=tri,
        bary=bary, nn_idx=nn_idx,
    )


def apply_raster_weights(weights: RasterWeights, values):
    """Apply a precomputed RasterWeights to node values.

    values shape (N,) -> grid (H, W)
           (N, C) -> grid (H, W, C)
           (..., N) or (..., N, C) leading batch dims -> (..., H, W) / (..., H, W, C)
    """
    H, W = weights.grid_hw
    vals = np.asarray(values, dtype=np.float64)
    # Normalise to (B, N, C): figure out where the node axis is.
    if vals.shape[-1] == weights.n_nodes:                 # (..., N)
        batch_shape = vals.shape[:-1]
        v = vals.reshape(-1, weights.n_nodes, 1)
        squeeze_c = True
    elif vals.ndim >= 2 and vals.shape[-2] == weights.n_nodes:   # (..., N, C)
        batch_shape = vals.shape[:-2]
        v = vals.reshape(-1, weights.n_nodes, vals.shape[-1])
        squeeze_c = False
    else:
        raise ValueError(f"values shape {vals.shape} has no axis of length n_nodes={weights.n_nodes}")
    B, _, C = v.shape
    out = np.empty((B, H * W, C), dtype=np.float64)
    inside = weights.simplex >= 0
    si = weights.simplex[inside]              # (m,)
    vert = weights.tri[si]                    # (m, 3) node indices
    bary_in = weights.bary[inside]            # (m, 3)
    nn_out = weights.nn_idx[~inside]          # (k,)
    for bi in range(B):
        col = v[bi]                          # (N, C)
        gp = np.empty((H * W, C), dtype=np.float64)
        # inside: barycentric blend
        gp[inside] = np.einsum("mk,mkc->mc", bary_in, col[vert])
        # outside: nearest fill
        gp[~inside] = col[nn_out]
        out[bi] = gp
    out = out.reshape(*batch_shape, H, W, C) if batch_shape else out.reshape(H, W, C)
    if squeeze_c:
        out = out[..., 0]
    return out
