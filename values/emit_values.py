#!/usr/bin/env python3
# ============================================================================
# Purpose: the value layer between the result matrices and the manuscript. It
#   turns each matrix row into one LaTeX macro (\rowKvs<Split><Model>,
#   \rowCmp<Split><Model>), recomputes which cell is bold, derives the cost
#   macros, and emits the single numbers that the text quotes (\v... macros)
#   from a few result JSONs.
# Pipeline role: the last stage, after the matrices. Reads
#   reports/PAPER_MATRIX_CLEAN/kvs_matrix.csv and
#   reports/PAPER_MATRIX_FIX/cmp_matrix.csv as the table sources, and the
#   result JSONs named in the constants below (the encoder JSON already at
#   import time). Output: values/paper_values.tex, which the paper \inputs.
#   `make check` emits it again and compares it byte for byte with the
#   committed file.
# Eval-consistency: this file decides what a number looks like, so its
#   formatting rules matter as much as the metrics. Table cells round with
#   Decimal ROUND_HALF_UP. A cell has 1 decimal if |mean| >= 10
#   or std >= 10, and 3 decimals otherwise. A "-0.000" is printed as "0.000".
#   The +-std subscript is printed only for n > 1. The bold cell is recomputed
#   from the numbers, not pinned, so a matrix rebuild can move it. The onset
#   column prints "---" for the models in DASH_ONSET and for empty values.
#   The cost macros rest on named constants. CAE_FIT_H is read from the
#   encoder result JSON, not inlined.
# ============================================================================
"""Emit values/paper_values.tex, the LaTeX macros for the table cells and quoted numbers.

Table sources (relative to the repository root):
  reports/PAPER_MATRIX_CLEAN/kvs_matrix.csv    all 16 KVS cells (8 models x 2 splits)
  reports/PAPER_MATRIX_FIX/cmp_matrix.csv      all 16 CMP cells (8 models x 2 splits)
The single numbers quoted in the text come from the result JSONs named in the constants.

Conventions: cells are $mean_{\\pm std}$ with 3 decimals (1 decimal if |mean| or std >= 10).
Deterministic rows (n = 1) carry no subscript. The onset cell is '---' where undefined.
Bold via \\mathbf marks the largest shedding power ratio P (KVS) and the smallest cum-WSS
error (CMP), per split.

CLI:
  python values/emit_values.py [--out PATH]
"""
from __future__ import annotations

import argparse
import csv
import json
import numpy as np
import sys
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:            # runnable without PYTHONPATH
    sys.path.insert(0, str(ROOT))

DEFAULT_OUT = ROOT / "values" / "paper_values.tex"

KVS_CLEAN = ROOT / "reports/PAPER_MATRIX_CLEAN/kvs_matrix.csv"
CMP_FIX = ROOT / "reports/PAPER_MATRIX_FIX/cmp_matrix.csv"

# Models whose onset cell prints "---" in both splits: they shed on fewer than half of
# the test cases (criterion stated in the table caption). DMDc and the S4D one-shot.
DASH_ONSET = {"dmdc", "oneshot_s4d"}

# The GPU times of the cost table were measured on one NVIDIA RTX PRO 6000 GPU in dedicated
# runs with no other job on the GPU. The encoder share is read from the encoder result
# JSON (convae_83x442_interp_seed0, meta.train_wallclock_s = 3793.29 s = 1.05 h).
CAE_FIT_JSON = ROOT / "results/benchmark/kvs_interp_convae_83x442_clean_0.json"
CAE_FIT_H = json.loads(CAE_FIT_JSON.read_text())["meta"]["train_wallclock_s"] / 3600.0
# The offline cost and the training time in solver queries (Results, Sec. 6.3) use the exact
# per-case FEM time from reports/fem_scaling.json, not the rounded table constants
# (644 x 0.77 + 1.59 = 497.47 would round to 497; the exact value 497.89 rounds to 498).
FEM_SCALING = ROOT / "reports/fem_scaling.json"
T_FEM_KVS_S_EXACT = json.loads(FEM_SCALING.read_text())["KVS"]["t_fem_per_case_s"]
# Encoder floor per split (Methods "instrument resolution", Results "branches at the
# encoder floor"): encode->decode of the test fields through the frozen encoder, scored with
# the same harness as every latent row (scripts/encoder_roundtrip_floor.py). The interp
# floor equals the metrics block of CAE_FIT_JSON.
ENC_FLOOR = {s: ROOT / f"results/benchmark/kvs_{s}_convae_83x442_floor_0.json" for s in ("interp", "extrap")}
# Latent orbits of the GRU pair and the S4D pair (Discussion, Fig. 12).
# Stats from scripts/latent_orbit_stats.py, computed on the latents.npz that
# scripts/latent_orbits.py writes.
R23_STATS = ROOT / "results/latent_orbits_gru/interp/orbit_paper_stats.json"
R23_STATS_S4D = ROOT / "results/latent_orbits_s4d/interp/orbit_paper_stats.json"
R23_LATENTS = ROOT / "results/latent_orbits_gru/interp/latents.npz"
# Per-step error growth of the film rollouts (scripts/cmp_rollout_error_growth.py).
CMP_GROWTH = ROOT / "results/benchmark/cmp_rollout_error_growth_interp.json"
# Film encoder floor: POD-k8 round trip of the CMP reference fields (scripts/cmp_encoder_floor.py).
FILM_FLOOR = ROOT / "results/benchmark/cmp3d_encoder_floor_fix.json"
# Mirror branch (Discussion): mirror-aligned RMSE of the S4D row and Z2 phase calls per seed
# (scripts/mirror_aligned_rmse.py).
MIRROR_JSON = ROOT / "reports/kvs_mirror_aligned_rmse.json"
# Field DeepONet checkpoint choice (Sec. 4.3): P of best.pt vs last.pt at the probes
# (scripts/field_don_checkpoint_probe_check.py).
FIELD_DON_CKPT = ROOT / "results/benchmark/field_don_checkpoint_probe_check_interp.json"
# Strouhal gate sentence in Metrics (what the gate excludes). Source
# scripts/strouhal_gate_count.py, same estimator and same condition as the scorer.
FFT_GATE = ROOT / "results/benchmark/kvs_strouhal_gate_counts.json"
# A priori rule of the Discussion (shedding period against the ramp, critical U_max from
# the Reynolds number definition). Source scripts/kvs_shedding_period_range.py.
SHED_PERIOD = ROOT / "results/benchmark/kvs_truth_shedding_period.json"
# KVS POD spectrum on the encoder grid 83x442 (Methods, Fig. 6); scripts/kvs_pod_spectrum.py.
POD_SPECTRUM_KVS = ROOT / "reports/kvs_pod_spectrum.json"
# Dynamics fit times in hours. s4d/mamba: timed on the RTX PRO 6000 with the paper configs
# (1941 s / 2301 s). gru: timed on an NVIDIA A40 GPU; the paper does not quote it.
TFIT_H = {"s4d": 0.54, "mamba": 0.64, "gru": 0.50}
N_TRAIN_KVS = 644      # KVS training cases
T_FEM_KVS_H = 0.77     # FEM time per KVS case in hours (reports/fem_scaling.json, rounded)
# Warm per-case inference latency in seconds, batch 1 (reports/inference_timing.json, rounded).
T_INFER_S4D_S = 0.66
T_INFER_MAMBA_S = 0.60

# Model order of the tables (display name, CSV model id)
KVS_IDS = [("FieldDON", "field_deeponet"), ("SFourD", "s4d_1phase4k"),
           ("Mamba", "mamba_1phase4k"), ("GRU", "gru_1phase4k"),
           ("ARMLP", "ar_mlp"), ("DMDc", "dmdc"),
           ("OneshotGRU", "oneshot_gru"), ("OneshotSFourD", "oneshot_s4d")]
CMP_IDS = [("FieldDON", "sdeeponet"), ("SFourD", "s4d_k8"), ("Mamba", "mamba_k8"),
           ("GRU", "gru_k8"), ("ARMLP", "armlp_k8"), ("DMDc", "dmdc_k8"),
           ("OneshotGRU", "oneshot_gru"), ("OneshotSFourD", "oneshot_s4d")]


# --------------------------------------------------------------------------- IO
def read_matrix(path):
    with open(path) as f:
        return {(r["model"], r["split"]): r for r in csv.DictReader(f)}


def _r(v, dec):
    return str(Decimal(str(v)).quantize(Decimal("1." + "0" * dec), rounding=ROUND_HALF_UP))


def fmt(v, std=None, n=None, signed=False):
    v = float(v)
    dec = 1 if abs(v) >= 10 or (std and float(std) >= 10) else 3
    body = _r(v, dec)
    if body.startswith("-0.") and float(body) == 0.0:
        body = body[1:]
    if std is not None and n not in ("1", 1, None) and int(n) > 1:
        body += f"_{{\\pm {_r(float(std), dec)}}}"
    cell = f"${body}$"
    if signed and not body.startswith("-"):
        cell = "$\\phantom{-}" + cell[1:]
    return cell


def _bold(cell):
    """$x_{\\pm y}$ -> $\\mathbf{x}_{\\pm y}$: only the mean is set in bold."""
    inner = cell.strip("$")
    mean, _, sub = inner.partition("_")
    return f"$\\mathbf{{{mean}}}{'_' + sub if sub else ''}$"


# ------------------------------------------------------------------------ cells
def kvs_cells(row, bold_power=False):
    """7 cells in table order: rmse, rmse_w, H, alpha, P, onset, fft."""
    n = row.get("rollout_rmse_n", "3")
    cells = [fmt(row[k], row.get(k + "_std"), n)
             for k in ("rollout_rmse", "rmse_warped", "halluc_pixel_frac_pre", "alpha_rms")]
    p = fmt(row["power_ratio"], row.get("power_ratio_std"), n)
    cells.append(_bold(p) if bold_power else p)
    if row["onset_latency_oscmed10"] in ("", None) or row["model"] in DASH_ONSET:
        cells.append("---")
    else:
        cells.append(fmt(row["onset_latency_oscmed10"],
                         row.get("onset_latency_oscmed10_std"), n))
    cells.append(fmt(row["fft_freq_err"], row.get("fft_freq_err_std"), n))
    return cells


def cmp_cells(row, bold_cw=False, bulk_key="nrmse_inplane"):
    """5 cells: bulk field error, corr (signed), D, cwL2, dWIWNU (signed).

    The bulk column is the in-plane error (u_x, u_y over the full 3D volume). u_z is at
    noise level in the CMP data (nrmse_z ~ 1 for every model) and has a small normaliser,
    so it would dominate nrmse_3d. bulk_key selects the matrix column."""
    n = row.get(bulk_key + "_n") or row.get("cumwss_rel_l2_inner_mean_n") or "3"
    c_n = fmt(row[bulk_key], row.get(bulk_key + "_std"), n)
    c_corr = fmt(row["spatial_corr_inner90"], row.get("spatial_corr_inner90_std"), n, signed=True)
    c_d = fmt(row["steady_drift_excess"], row.get("steady_drift_excess_std"), n)
    c_cw = fmt(row["cumwss_rel_l2_inner_mean"], row.get("cumwss_rel_l2_inner_mean_std"), n)
    if bold_cw:
        c_cw = _bold(c_cw)
    c_w = fmt(row["wiwnu_excess"], row.get("wiwnu_excess_std"), n, signed=True)
    return [c_n, c_corr, c_d, c_cw, c_w]


def _v(mat, model, split, key):
    return float(mat[(model, split)][key])


def _growth(grow):
    """Per-step error curves of the film (seed mean of the case means): values at the first
    predicted step, at the ramp end (t = 1 s) and at the end (t = 2 s), the ratio end/ramp end,
    and the largest relative deviation of the curve-based self-check from the scorer."""
    t = np.array(grow["t_grid"])
    i_ramp = int(np.argmin(np.abs(t - 1.0)))
    out = {"at_first": {}, "at_ramp_end": {}, "at_end": {}, "ratio": {}, "ar_keys": []}
    dev = 0.0
    for k, m in grow["models"].items():
        c = np.array(m["seed_mean"]["curve_mean"])
        i0 = m["first_pred_index"]
        out["at_first"][k] = float(c[i0]); out["at_ramp_end"][k] = float(c[i_ramp])
        out["at_end"][k] = float(c[-1]); out["ratio"][k] = float(c[-1] / c[i_ramp])
        if m["cell"].endswith("AR"):
            out["ar_keys"].append(k)
        for sd in m["seeds"].values():
            pooled = sd["pooled_nrmse_inplane_from_curve"]
            dev = max(dev, abs(sd["pooled_minus_scorer"]) / pooled)
            if sd.get("rel_delta_vs_reinfer_nrmse_inplane") is not None:
                dev = max(dev, abs(sd["rel_delta_vs_reinfer_nrmse_inplane"]))
    out["max_rel_dev"] = dev
    out["i_ramp"] = i_ramp
    return out


def _ladder(orb):
    """U_max ladder of the training and test cases (from theta in the GRU latents.npz) and
    the Nyquist minimum number of ladder levels for the phase at t*: ceil(2 x turns) + 1 levels."""
    z = np.load(R23_LATENTS)
    th = np.concatenate([z["theta_train"], z["theta_test"]])
    levels = np.unique(np.round(th[:, 0], 6))
    step = float(np.median(np.diff(levels)))
    wraps = orb["winding"]["wraps_across_ladder_at_t_star_median"]
    # The phase difference spans the lowest to the highest level, i.e. n_levels-1 steps.
    # Nyquist: more than two steps per turn -> ceil(2*wraps) steps, i.e. ceil(2*wraps)+1 levels.
    n_steps = int(levels.size) - 1
    n_req_levels = int(np.ceil(2.0 * wraps)) + 1
    return {"n_levels": int(levels.size), "n_steps": n_steps, "umin": float(levels.min()),
            "umax": float(levels.max()), "step": step, "turns_per_step": wraps / n_steps,
            "n_required": n_req_levels, "nyquist_factor": n_req_levels / levels.size}


def build(out_path):
    kvs = read_matrix(KVS_CLEAN)
    cmp_ = read_matrix(CMP_FIX)

    lines = [f"% Generated by values/emit_values.py -> {out_path.name}; do not edit by hand.",
             "% Sources: reports/PAPER_MATRIX_CLEAN/kvs_matrix.csv (wake), "
             "reports/PAPER_MATRIX_FIX/cmp_matrix.csv (film) and the result files named in the emitter."]

    for split, cap in [("interp", "Interp"), ("extrap", "Extrap")]:
        bold = max((m for _, m in KVS_IDS), key=lambda m: _v(kvs, m, split, "power_ratio"))
        for name, mid in KVS_IDS:
            cells = kvs_cells(kvs[(mid, split)], bold_power=(mid == bold))
            lines.append(f"\\newcommand{{\\rowKvs{cap}{name}}}"
                         f"{{{' & '.join(cells)}}}")

    for split, cap in [("interp", "Interp"), ("extrap", "Extrap")]:
        bold = min((m for _, m in CMP_IDS),
                   key=lambda m: _v(cmp_, m, split, "cumwss_rel_l2_inner_mean"))
        for name, mid in CMP_IDS:
            cells = cmp_cells(cmp_[(mid, split)], bold_cw=(mid == bold))
            lines.append(f"\\newcommand{{\\rowCmp{cap}{name}}}"
                         f"{{{' & '.join(cells)}}}")

    # Cost table and derived quantities
    t_train = {k: CAE_FIT_H + v for k, v in TFIT_H.items()}
    qstar = {}
    for k, t_inf_s in [("s4d", T_INFER_S4D_S), ("mamba", T_INFER_MAMBA_S)]:
        qstar[k] = (N_TRAIN_KVS * T_FEM_KVS_H + t_train[k]) / (T_FEM_KVS_H - t_inf_s / 3600.0)
    # Results, Sec. 6.3: offline cost N_train*t_FEM + t_train (wake champion S4D) and t_train
    # in solver queries, both with the exact t_FEM from fem_scaling.json.
    offline_h = {k: (N_TRAIN_KVS * T_FEM_KVS_S_EXACT + t_train[k] * 3600.0) / 3600.0
                 for k in ("s4d", "mamba")}
    train_queries = {k: t_train[k] * 3600.0 / T_FEM_KVS_S_EXACT for k in ("s4d", "mamba")}

    cw = {m: _v(cmp_, m, "interp", "cumwss_rel_l2_inner_mean") for _, m in CMP_IDS}
    ip = {m: _v(cmp_, m, "interp", "nrmse_inplane") for _, m in CMP_IDS}
    lines += [
        f"\\newcommand{{\\vCaeFitH}}{{{CAE_FIT_H:.2f}}}",
        f"\\newcommand{{\\vSFourDFitH}}{{{TFIT_H['s4d']:.2f}}}",
        f"\\newcommand{{\\vMambaFitH}}{{{TFIT_H['mamba']:.2f}}}",
        f"\\newcommand{{\\vGruFitH}}{{{TFIT_H['gru']:.2f}}}",
        f"\\newcommand{{\\vSFourDTrainH}}{{{t_train['s4d']:.2f}}}",
        f"\\newcommand{{\\vMambaTrainH}}{{{t_train['mamba']:.2f}}}",
        f"\\newcommand{{\\vKvsQstar}}{{{round(qstar['s4d'])}}}",
        # Results, Sec. 6.3: offline cost and training time in solver queries
        f"\\newcommand{{\\vKvsOfflineH}}{{{round(offline_h['s4d'])}}}",
        f"\\newcommand{{\\vKvsTrainQueriesLo}}{{{_r(min(train_queries.values()), 1)}}}",
        f"\\newcommand{{\\vKvsTrainQueriesHi}}{{{_r(max(train_queries.values()), 1)}}}",
        # Film bulk-error factor GRU-DON vs S-DeepONet, nrmse in-plane (Discussion "factor of 1.4")
        f"\\newcommand{{\\vFilmIpFactor}}{{{ip['gru_k8'] / ip['sdeeponet']:.1f}}}",
        # Field DeepONet last checkpoint (Sec. 4.3), source FIELD_DON_CKPT: max P over seeds
        f"\\newcommand{{\\vKvsFieldDonLastP}}{{{_r(max(v['last.pt']['P'] for v in json.loads(FIELD_DON_CKPT.read_text())['seeds'].values()), 2)}}}",
        # Mirror branch (Discussion), source MIRROR_JSON, extrap runs only
        f"\\newcommand{{\\vKvsMirrorCloserCases}}{{{round(np.mean([v['n_mirror_better'] for k, v in json.loads(MIRROR_JSON.read_text())['per_run'].items() if '_extrap_seed' in k]))}}}",
        f"\\newcommand{{\\vKvsMirrorNCases}}{{{sorted({v['n_cases'] for k, v in json.loads(MIRROR_JSON.read_text())['per_run'].items() if '_extrap_seed' in k})[0]}}}",
        f"\\newcommand{{\\vKvsMirrorGainPct}}{{{round(100 * (1 - np.mean([v['rmse_best2'] for k, v in json.loads(MIRROR_JSON.read_text())['per_run'].items() if '_extrap_seed' in k]) / np.mean([v['rmse_raw_valid'] for k, v in json.loads(MIRROR_JSON.read_text())['per_run'].items() if '_extrap_seed' in k])))}}}",
        # Film encoder floor (Limitations), source FILM_FLOOR: POD-k8 round trip
        f"\\newcommand{{\\vFilmEncFloorInterpCw}}{{{_r(json.loads(FILM_FLOOR.read_text())['interp']['k8']['cumwss_rel_l2_inner_mean'], 3)}}}",
        f"\\newcommand{{\\vFilmEncFloorExtrapCw}}{{{_r(json.loads(FILM_FLOOR.read_text())['extrap']['k8']['cumwss_rel_l2_inner_mean'], 3)}}}",
        f"\\newcommand{{\\vFilmEncFloorInterpIp}}{{{_r(json.loads(FILM_FLOOR.read_text())['interp']['k8']['nrmse_inplane'], 2)}}}",
        f"\\newcommand{{\\vFilmEncFloorExtrapIp}}{{{_r(json.loads(FILM_FLOOR.read_text())['extrap']['k8']['nrmse_inplane'], 2)}}}",
        # KVS POD spectrum (Methods, Fig. 6), source POD_SPECTRUM_KVS: encoder grid 83x442
        f"\\newcommand{{\\vKvsPodModesNinetyNine}}{{{json.loads(POD_SPECTRUM_KVS.read_text())['modes_for_99pct']}}}",
        f"\\newcommand{{\\vKvsPodModeOnePct}}{{{_r(100 * json.loads(POD_SPECTRUM_KVS.read_text())['cumEVR']['1'], 0)}}}",
    ]

    # Encoder floor macros (Methods, Results, Limitations), source ENC_FLOOR
    floors = {s: json.loads(p.read_text())["metrics"] for s, p in ENC_FLOOR.items()}
    for s, cap in [("interp", "Interp"), ("extrap", "Extrap")]:
        lines += [
            f"\\newcommand{{\\vKvsEncFloor{cap}Rmse}}{{{floors[s]['rollout_rmse']:.2f}}}",
            f"\\newcommand{{\\vKvsEncFloor{cap}Rmsew}}{{{floors[s]['rmse_warped']:.2f}}}",
            f"\\newcommand{{\\vKvsEncFloor{cap}P}}{{{floors[s]['power_ratio']:.2f}}}",
        ]

    # Latent orbits of the GRU pair (Discussion), source R23_STATS
    orb = json.loads(R23_STATS.read_text())
    w = orb["winding"]
    lines += [
        f"\\newcommand{{\\vKvsOrbitPowerAr}}{{{_r(orb['latent_power_mean']['ar'], 2)}}}",
        f"\\newcommand{{\\vKvsOrbitPowerOs}}{{{_r(orb['latent_power_mean']['os'], 2)}}}",
        f"\\newcommand{{\\vKvsOrbitRadiusAr}}{{{_r(orb['radius_median']['ar'], 2)}}}",
        f"\\newcommand{{\\vKvsOrbitRadiusOs}}{{{_r(orb['radius_median']['os'], 2)}}}",
        f"\\newcommand{{\\vKvsOrbitRadiusOsLate}}{{{_r(orb['radius_late_median']['os'], 2)}}}",
        f"\\newcommand{{\\vKvsOrbitBandArPct}}{{{round(100 * orb['band_ratio_median']['ar'])}}}",
        f"\\newcommand{{\\vKvsOrbitBandOsPct}}{{{round(100 * orb['band_ratio_median']['os'])}}}",
        f"\\newcommand{{\\vKvsOrbitAliasedHiPct}}{{{round(100 * w['frac_neighbours_over_quarter_turn_umax_ge_split'])}}}",
        f"\\newcommand{{\\vKvsOrbitAliasedLoPct}}{{{round(100 * w['frac_neighbours_over_quarter_turn_umax_lt_split'])}}}",
        f"\\newcommand{{\\vKvsOrbitUmaxSplit}}{{{orb['definitions']['umax_split']:.1f}}}",
        f"\\newcommand{{\\vKvsOrbitTstar}}{{{orb['definitions']['t_star_s']:.0f}}}",
        f"\\newcommand{{\\vKvsOrbitTlate}}{{{orb['definitions']['late_window_s']:.1f}}}",
        f"\\newcommand{{\\vKvsOrbitCase}}{{{orb['exhibit']['case_id'].replace('_', '\\_')}}}",
        f"\\newcommand{{\\vKvsOrbitNall}}{{{orb['n_all_cases']}}}",
        # In-plane / out-of-plane split of the radius (late window), GRU pair
        f"\\newcommand{{\\vKvsOrbitInPlaneTruthLate}}{{{_r(orb['radius_split']['late']['truth_in'], 2)}}}",
        f"\\newcommand{{\\vKvsOrbitInPlaneArLate}}{{{_r(orb['radius_split']['late']['ar_in'], 2)}}}",
        f"\\newcommand{{\\vKvsOrbitInPlaneOsLate}}{{{_r(orb['radius_split']['late']['os_in'], 2)}}}",
        f"\\newcommand{{\\vKvsOrbitOutPlaneTruthLate}}{{{_r(orb['radius_split']['late']['truth_out'], 2)}}}",
        f"\\newcommand{{\\vKvsOrbitOutPlaneOsLate}}{{{_r(orb['radius_split']['late']['os_out'], 2)}}}",
    ]
    # Sampling bound: turns across the U_max ladder at t*, ladder levels, Nyquist requirement
    lad = _ladder(orb)
    lines += [
        f"\\newcommand{{\\vKvsOrbitWraps}}{{{round(w['wraps_across_ladder_at_t_star_median'])}}}",
        f"\\newcommand{{\\vKvsLadderLevels}}{{{lad['n_levels']}}}",
        f"\\newcommand{{\\vKvsLadderStep}}{{{_r(lad['step'], 1)}}}",
        f"\\newcommand{{\\vKvsLadderTurnsPerLevel}}{{{_r(lad['turns_per_step'], 2)}}}",
        f"\\newcommand{{\\vKvsLadderNyquistLevels}}{{{lad['n_required']}}}",
        f"\\newcommand{{\\vKvsLadderNyquistFactor}}{{{_r(lad['nyquist_factor'], 1)}}}",
    ]

    # Film error growth (Results, CMP), source CMP_GROWTH; prefix vCmpGrow
    grow = json.loads(CMP_GROWTH.read_text())
    g = _growth(grow)
    lines += [
        f"\\newcommand{{\\vCmpGrowDirectRampEnd}}{{{_r(g['at_ramp_end']['sdeeponet'], 2)}}}",
        f"\\newcommand{{\\vCmpGrowDirectEnd}}{{{_r(g['at_end']['sdeeponet'], 2)}}}",
        f"\\newcommand{{\\vCmpGrowDirectRatio}}{{{_r(g['ratio']['sdeeponet'], 2)}}}",
        f"\\newcommand{{\\vCmpGrowGruRampEnd}}{{{_r(g['at_ramp_end']['gru_k8'], 2)}}}",
        f"\\newcommand{{\\vCmpGrowGruEnd}}{{{_r(g['at_end']['gru_k8'], 2)}}}",
        f"\\newcommand{{\\vCmpGrowGruRatio}}{{{_r(g['ratio']['gru_k8'], 2)}}}",
        f"\\newcommand{{\\vCmpGrowSsdRampEnd}}{{{_r(g['at_ramp_end']['s4d_k8'], 2)}}}",
        f"\\newcommand{{\\vCmpGrowMambaRampEnd}}{{{_r(g['at_ramp_end']['mamba_k8'], 2)}}}",
        f"\\newcommand{{\\vCmpGrowMlpRatio}}{{{_r(g['ratio']['armlp_k8'], 1)}}}",
        f"\\newcommand{{\\vCmpGrowMlpEnd}}{{{_r(g['at_end']['armlp_k8'], 2)}}}",
        f"\\newcommand{{\\vCmpGrowDmdcRatio}}{{{_r(g['ratio']['dmdc_k8'], 1)}}}",
        f"\\newcommand{{\\vCmpGrowArFloor}}{{{_r(g['at_first']['gru_k8'], 3)}}}",
        f"\\newcommand{{\\vCmpGrowNsteps}}{{{grow['n_steps']}}}",
    ]

    # Latent orbits of the S4D pair (Discussion), source R23_STATS_S4D; prefix vKvsOrbitS
    orb2 = json.loads(R23_STATS_S4D.read_text())
    lines += [
        f"\\newcommand{{\\vKvsOrbitSPowerAr}}{{{_r(orb2['latent_power_mean']['ar'], 2)}}}",
        f"\\newcommand{{\\vKvsOrbitSPowerOs}}{{{_r(orb2['latent_power_mean']['os'], 2)}}}",
        f"\\newcommand{{\\vKvsOrbitSRadiusAr}}{{{_r(orb2['radius_median']['ar'], 2)}}}",
        f"\\newcommand{{\\vKvsOrbitSBandArPct}}{{{round(100 * orb2['band_ratio_median']['ar'])}}}",
        f"\\newcommand{{\\vKvsOrbitSBandOsPct}}{{{_r(100 * orb2['band_ratio_median']['os'], 1)}}}",
        f"\\newcommand{{\\vKvsOrbitSInPlaneArLate}}{{{_r(orb2['radius_split']['late']['ar_in'], 2)}}}",
        f"\\newcommand{{\\vKvsOrbitSInPlaneOsLate}}{{{_r(orb2['radius_split']['late']['os_in'], 2)}}}",
        f"\\newcommand{{\\vKvsOrbitSOutPlaneOsLate}}{{{_r(orb2['radius_split']['late']['os_out'], 2)}}}",
    ]

    # Strouhal gate macros (Metrics), source FFT_GATE
    gate = json.loads(FFT_GATE.read_text())["splits"]
    lines += [
        f"\\newcommand{{\\vKvsFftPairsInterp}}{{{gate['interp']['n_pairs']}}}",
        f"\\newcommand{{\\vKvsFftDroppedInterp}}{{{gate['interp']['n_dropped']}}}",
        f"\\newcommand{{\\vKvsFftDroppedExtrap}}{{{gate['extrap']['n_dropped']}}}",
    ]

    # Shedding period macros (Datasets, Discussion practical rule), source SHED_PERIOD
    shed = json.loads(SHED_PERIOD.read_text())
    lines += [
        f"\\newcommand{{\\vKvsShedPeriodMin}}{{{_r(shed['period_min_s'], 2)}}}",
        f"\\newcommand{{\\vKvsShedPeriodMax}}{{{_r(shed['period_max_s'], 2)}}}",
        f"\\newcommand{{\\vKvsUmaxCrit}}{{{_r(shed['umax_crit'], 1)}}}",
    ]

    out_path.write_text("\n".join(lines) + "\n")
    return lines


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT,
                    help=f"output path of the .tex file (default: {DEFAULT_OUT})")
    args = ap.parse_args(argv)

    lines = build(args.out)
    print(f"wrote {args.out} ({len(lines)} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
