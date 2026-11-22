# =============================================================================
# Purpose: apply the KVS onset detector "oscMed10" to the truth and to every
#   run directory of the paper registry, and report the onset latency
#   dt* = t*_pred - t*_truth per case, per run and per model.
# Pipeline role: this file produces the onset column of the KVS results table
#   (Table 5). It reads results/benchmark/runs/<rundir>/predictions.npz and the
#   truth results/benchmark/truth/kvs_<split>.npz and writes
#   reports/onset_oscmed10_per_model<sfx>[_extrap].npz,
#   whose mean_abs_dt scripts/build_paper_matrix.py (_onset_map) reads as the
#   metric onset_latency_oscmed10.
# Eval-consistency: every constant below is part of the reported number:
#   W = 13 frames (0.26 s) for the centred rolling median,
#   FRAC = 0.10 of p95(O_truth) as the threshold, applied to truth and
#   prediction alike, CONSENSUS = 2 of 3 stripes, PERSIST = 3 frames, search
#   window [SEARCH_LO, SEARCH_HI] = [0.4, 3.5] s, stripe geometry DX = 0.04 /
#   DY = 0.10 at x_c + {-0.02, +0.02, +0.06}. The stripes are taken on the
#   fixed eval nodes (benchmark/splits/kvs_eval_nodes.npz) around the mesh's
#   own cylinder_center, so truth and every model are measured at exactly the
#   same points. No crossing means the onset is undefined (NaN), never "the
#   earliest one": a damped prediction has no onset by definition, and that is
#   how the two empty extrap onset cells of the KVS matrix arise.
#   The threshold comes from the TRUTH trace, not from the prediction's own
#   pre-onset segment, so a noisy prediction cannot raise its own threshold.
#   Registry and output suffix come from benchmark/paper_models.py
#   (active_kvs_2a / kvs_rundirs / registry_suffix). _SFX is "_clean", so the
#   files this writes are the files the matrix builder reads.
#   Rebuilding these numbers needs the per-run predictions.npz files, which
#   are not included. The two small NPZ outputs are in reports/, so the
#   matrix rebuild runs without them.
# Note: the module has no __main__ guard. Importing it runs the full scoring
#   pass and writes the report files.
# =============================================================================
"""KVS onset detector oscMed10.

Definition:

  s_k(t)  = mean_top_k[v] + mean_bot_k[v]      SIGNED stripe signal, 3 stripes
            x in {0.18, 0.22, 0.26} (+-0.02), y halves +-0.10 around y_c, P2 eval nodes
  O_k(t)  = rolling MEDIAN (centred, w = 13 frames = 0.26 s) of |Delta s_k|
  thr_k   = 0.10 x p95(O_truth_k over the full trace), used for BOTH truth and prediction
  Onset   = first frame with >= 2 of 3 stripes above thr, persistent for 3 frames,
            search window [0.4, 3.5] s. No crossing -> onset undefined (NaN):
            damped predictions have no onset by definition.

The model list comes from the registry (benchmark/paper_models.py, including DMDc). Input:
results/benchmark/runs/<rundir>/predictions.npz and results/benchmark/truth/kvs_{split}.npz.

Output:
  reports/onset_oscmed10_per_model<sfx>.npz / <sfx>_extrap.npz   (truth_t, case_ids,
      model_names, pred_t_matrix, dt_matrix, mean_abs_dt, median_dt, std_dt, n_finite
      -> build_paper_matrix._onset_map)
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]   # repository root
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from benchmark.paper_models import active_kvs_2a, kvs_rundirs, registry_suffix
_SFX = registry_suffix()

RUNS = REPO / "results" / "benchmark" / "runs"
REPORTS = REPO / "reports"

# --- detector parameters (they define the reported onset numbers) ---
W = 13
FRAC = 0.10
PERSIST = 3
CONSENSUS = 2
SEARCH_LO, SEARCH_HI = 0.4, 3.5

# ---------- geometry and stripes ----------
mesh = np.load(REPO / "meshes" / "refined_4x" / "mesh.npz", allow_pickle=False)
cc = np.array(mesh["cylinder_center"], dtype=np.float64)
xc, yc = float(cc[0]), float(cc[1])
ev = np.load(REPO / "benchmark" / "splits" / "kvs_eval_nodes.npz", allow_pickle=False)
exy = np.array(mesh["coords_p2"], dtype=np.float64)[np.array(ev["eval_idx_p2"], dtype=np.int64)]
DX, DY = 0.04, 0.10
sxs = [xc - 0.02, xc + 0.02, xc + 0.06]
wtop = (exy[:, 1] > yc) & (exy[:, 1] < yc + DY)
wbot = (exy[:, 1] < yc) & (exy[:, 1] > yc - DY)
STRIPES = [((exy[:, 0] > s - DX / 2) & (exy[:, 0] < s + DX / 2) & wtop,
            (exy[:, 0] > s - DX / 2) & (exy[:, 0] < s + DX / 2) & wbot) for s in sxs]


def signed_sig_all(v_all):
    """(n_cases, T, n_eval) -> (n_cases, 3, T)."""
    return np.stack([v_all[:, :, mt].mean(2) + v_all[:, :, mb].mean(2)
                     for mt, mb in STRIPES], axis=1)


def osc_median(S):
    """S: (..., 3, T) signed -> rolling median (centred, w=W) of |diff|."""
    d = np.abs(np.diff(S, axis=-1, prepend=S[..., :1]))
    T = d.shape[-1]
    h = W // 2
    O = np.empty_like(d)
    for t in range(T):
        a, b = max(0, t - h), min(T, t + h + 1)
        O[..., t] = np.median(d[..., a:b], axis=-1)
    return O


def detect(O, thr, tg, search):
    """O: (n, 3, T), thr: (n, 3) -> (n,) onsets (NaN = undefined)."""
    act = (O >= thr[..., None]).sum(axis=1) >= CONSENSUS
    cand = act & search[None, :]
    roll = cand.copy()
    for k in range(1, PERSIST):
        roll[:, k:] &= cand[:, :-k]
    has = roll.any(axis=1)
    first = roll.argmax(axis=1) - (PERSIST - 1)
    return np.where(has, tg[np.clip(first, 0, None)], np.nan)


def load_signed(run, ids):
    """Stripe signals of one run from its predictions.npz."""
    z = np.load(RUNS / run / "predictions.npz", allow_pickle=False)
    v = np.asarray(z["u_eval"], dtype=np.float32)[..., 1]
    pids = [str(c) for c in z["case_ids"]]
    if pids != ids:
        pos = {c: i for i, c in enumerate(pids)}
        v = v[np.array([pos[c] for c in ids], dtype=np.int64)]
    return signed_sig_all(v).astype(np.float64)


for split in ("interp", "extrap"):
    tz = np.load(REPO / "results" / "benchmark" / "truth" / f"kvs_{split}.npz",
                 allow_pickle=False)
    tg = np.array(tz["t_grid"], dtype=np.float64)
    ids = [str(c) for c in tz["case_ids"]]
    truth_S = signed_sig_all(np.asarray(tz["u_eval"], dtype=np.float32)[..., 1])
    search = (tg >= SEARCH_LO) & (tg <= SEARCH_HI)

    O_t = osc_median(truth_S)
    thr = FRAC * np.percentile(O_t, 95, axis=-1)         # (nc, 3)
    truth_t = detect(O_t, thr, tg, search)
    print(f"[{split}] truth: finite={np.isfinite(truth_t).sum()}/{len(ids)} "
          f"median={np.nanmedian(truth_t):.2f}")

    model_names, pred_rows, dt_rows = [], [], []
    for m in active_kvs_2a():
        for run in kvs_rundirs(m, split):
            if not (RUNS / run / "predictions.npz").is_file():
                print(f"  [MISS] {run}"); continue
            S_p = load_signed(run, ids)
            pred_t = detect(osc_median(S_p), thr, tg, search)
            dt = pred_t - truth_t
            fin = np.isfinite(dt)
            model_names.append(run)
            pred_rows.append(pred_t)
            dt_rows.append(dt)
            ma = float(np.mean(np.abs(dt[fin]))) if fin.any() else None
            print(f"  {run:<52s} mean|dt|={'undef' if ma is None else f'{ma:.3f}'} "
                  f"n={int(fin.sum())}")

    # per-model NPZ (read by build_paper_matrix._onset_map)
    dt_mat = np.stack(dt_rows)
    abs_dt = np.abs(dt_mat)
    out_npz = REPORTS / (f"onset_oscmed10_per_model{_SFX}.npz" if split == "interp"
                         else f"onset_oscmed10_per_model{_SFX}_extrap.npz")
    np.savez(out_npz,
             truth_t=truth_t, case_ids=np.array(ids),
             model_names=np.array(model_names),
             pred_t_matrix=np.stack(pred_rows), dt_matrix=dt_mat,
             mean_abs_dt=np.array([np.nanmean(r) if np.isfinite(r).any() else np.nan
                                   for r in abs_dt]),
             median_dt=np.array([np.nanmedian(r) if np.isfinite(r).any() else np.nan
                                 for r in dt_mat]),
             std_dt=np.array([np.nanstd(r) if np.isfinite(r).any() else np.nan
                              for r in dt_mat]),
             n_finite=np.isfinite(dt_mat).sum(axis=1))
    print(f"[saved] {out_npz.name} ({len(model_names)} runs)")
