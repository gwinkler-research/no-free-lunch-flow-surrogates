# -*- coding: utf-8 -*-
"""Does the gap difference quotient agree with the resolved wafer-side wall shear?

The paper evaluates the film shear as mu |u_top - u_bot| / h (Datasets section). This script
compares that quotient with the one-sided second-order wall gradient on the wafer side, built
from the three uppermost z-levels of the training npz, on the 4,000 columns that are resolved on
every level, restricted to r/R <= 0.9 under the wafer, in the steady window t in [1, 2] s.
Needs the CMP FEM training tensor, which is not in this repository (--npz).

Usage:
  python scripts/cmp_couette_gap_quotient_check.py \
      --npz <data dir>/cmp3d_full100fix_fem_training.npz \
      --out results/benchmark/cmp_couette_gap_quotient_fix.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
REPO = Path(_REPO_ROOT)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--npz", default=str(REPO / "data" / "cmp" / "cmp3d_full100fix_fem_training.npz"))
    ap.add_argument("--out", default="results/benchmark/cmp_couette_gap_quotient_fix.json")
    ap.add_argument("--t-step", type=int, default=20, help="stride over the steady window (201 steps, dt 0.01 s)")
    a = ap.parse_args(argv)
    z = np.load(a.npz, allow_pickle=True)
    info = json.loads(str(z["info"]))
    zl = np.array(info["z_levels"])
    h = float(zl[-1])
    lev, ip = z["z_level_id"], z["in_plane_id"]
    m, s = z["stats_mean"], z["stats_std"]
    top, bot = np.where(lev == 28)[0], np.where(lev == 0)[0]
    l27, l26 = np.where(lev == 27)[0], np.where(lev == 26)[0]
    assert np.array_equal(np.sort(ip[l27]), np.sort(ip[l26]))
    o27, o26 = np.argsort(ip[l27]), np.argsort(ip[l26])
    cols = ip[l27][o27]
    top_by = np.empty(info["n_inplane"], int); top_by[ip[top]] = top
    bot_by = np.empty(info["n_inplane"], int); bot_by[ip[bot]] = bot
    S, xyz, wc, R = z["states_norm"], z["node_xyz"], z["wafer_centers"], info["wafer_radius"]
    tsel = np.arange(100, 201, a.t_step)
    dz = float(zl[28] - zl[27])
    res = []
    for c in range(S.shape[0]):
        u = S[c][tsel].astype(np.float32) * s + m
        ut, ub = u[:, top_by[cols], :2], u[:, bot_by[cols], :2]
        u27, u26 = u[:, l27[o27], :2], u[:, l26[o26], :2]
        q = np.linalg.norm(ut - ub, axis=-1) / h
        g2 = np.linalg.norm(1.5 * ut - 2.0 * u27 + 0.5 * u26, axis=-1) / dz
        xy = xyz[top_by[cols], :2]
        r = np.hypot(xy[:, 0] - wc[c, 0], xy[:, 1] - wc[c, 1])
        wm = r <= 0.9 * R
        rel = np.abs(q[:, wm] - g2[:, wm]) / np.maximum(g2[:, wm], 1e-9)
        res.append(dict(case=str(z["case_names"][c]), n_cols=int(wm.sum()),
                        rel_median=float(np.median(rel)), rel_p90=float(np.percentile(rel, 90)),
                        mean_ratio=float(q[:, wm].mean() / g2[:, wm].mean())))
        print(f"[gap-quotient] {res[-1]}", flush=True)
    arr = lambda k: np.array([r[k] for r in res])
    out = dict(definition="gap quotient |u_top - u_bot| / h against the one-sided second-order wafer-side "
                          "gradient from z-levels 26, 27, 28; 4,000 fully resolved columns, r/R <= 0.9 under "
                          "the wafer, t in [1, 2] s; relative deviation per column and time, summarised per case",
               npz=a.npz, n_cases=len(res), t_steps=[float(t / 100.0) for t in tsel],
               rel_median_over_cases=float(np.median(arr("rel_median"))),
               rel_p90_max_over_cases=float(arr("rel_p90").max()),
               mean_ratio_min=float(arr("mean_ratio").min()), mean_ratio_max=float(arr("mean_ratio").max()),
               per_case=res)
    p = REPO / a.out
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1))
    print(f"[gap-quotient] wrote {p}: median {out['rel_median_over_cases']:.4f}, worst P90 {out['rel_p90_max_over_cases']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
