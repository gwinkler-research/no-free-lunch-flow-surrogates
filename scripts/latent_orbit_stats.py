#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ============================================================================
# Purpose: turn the stored latent trajectories of a KVS AR/one-shot pair
#   (default: GRU pair, latent-AR against latent one-shot, both on the frozen
#   interp encoder) into the numbers the Discussion quotes: the latent
#   oscillation power, the orbit radius relative to the truth, the share of
#   the truth's shedding-band variance a model retains, and the phase winding
#   of the truth across the U_max ladder of every ramp schedule.
# Pipeline role: reads results/latent_orbits_gru/interp/latents.npz (written
#   by scripts/latent_orbits.py, GPU re-inference of the paper runs) and writes
#   orbit_paper_stats.json next to it. values/emit_values.py turns that JSON
#   into the \vKvsOrbit* macros. --src points the same computation at the
#   S4D pair (results/latent_orbits_s4d/interp).
# Eval-consistency: the steady window is the window of the shedding power P
#   (t >= 2 s). The latent power is the pooled temporal variance ratio, the
#   latent analogue of P, and must equal the P_latent block of stats.json from
#   the re-inference (self-check, tolerance 1e-6). Every ratio is taken against
#   the truth of the same case, centred on the truth's own steady-window mean,
#   so AR and one-shot are measured against one reference. The phase is the
#   analytic-signal (Hilbert) phase of the first global principal component of
#   the truth latents.
# ============================================================================
"""Paper statistics from the stored latent orbits of a KVS AR/one-shot pair.

  python3 -u scripts/latent_orbit_stats.py
  python3 -u scripts/latent_orbit_stats.py --check   # only compare against the existing JSON
  python3 -u scripts/latent_orbit_stats.py --src results/latent_orbits_s4d/interp   # S4D pair
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import hilbert

_REPO = Path(__file__).resolve().parents[1]
SRC_DEFAULT = "results/latent_orbits_gru/interp"      # GRU pair
OUT_NAME = "orbit_paper_stats.json"

T_STEADY = 2.0          # steady window t >= 2 s (as for the shedding power P)
T_LATE = 2.5            # window with the initial transient excluded
T_STAR = 3.0            # instant at which the truth phase is read across the ladder
UMAX_SPLIT = 2.5        # half of the design range U_max in [0.5, 5]
QUARTER_TURN = np.pi / 2
BAND_HALF_BINS = 1      # shedding band: dominant truth bin +- 1 bin (df = 0.495 Hz)


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _centred(zz, zt, mask):
    return zz[:, mask] - zt[:, mask].mean(axis=1, keepdims=True)


def _power(zz_s, zt, mask):
    return float(zz_s[:, mask].var(axis=1).sum() / zt[:, mask].var(axis=1).sum())


def _radius(zz_s, zt, mask):
    num = np.sqrt((_centred(zz_s, zt, mask) ** 2).sum(-1).mean(1))
    den = np.sqrt((_centred(zt, zt, mask) ** 2).sum(-1).mean(1))
    return num / den


def _radius_split(zz_s, zt, mask_win, mask_ref):
    """The orbit radius split into the part INSIDE the case's own orbit plane and the
    residual OUTSIDE it. Plane and centring come from the reference window mask_ref: the
    plane is the leading 2D SVD plane of the centred truth of that case over mask_ref (the
    plane of Fig. 12, t >= 2 s), the centring is the truth's mean over mask_ref (as in
    _centred). RMS over time and the normalisation by the truth's
    own RMS excursion (as in _radius) are taken over mask_win. For the steady window both
    masks are t >= 2 s and r_in**2 + r_out**2 == _radius**2 per case; the late window
    (t >= 2.5 s) keeps the steady plane and centring, so it differs from radius_late_median,
    which re-centres on the late-window mean."""
    zt_c = _centred(zt, zt, mask_ref)                       # (N, T_ref, d): plane source
    mu = zt[:, mask_ref].mean(axis=1, keepdims=True)
    zt_w = zt[:, mask_win] - mu
    zz_w = zz_s[:, mask_win] - mu
    den = np.sqrt((zt_w ** 2).sum(-1).mean(1))
    r_in, r_out = np.empty(len(zt_c)), np.empty(len(zt_c))
    for i in range(len(zt_c)):
        _, _, Vc = np.linalg.svd(zt_c[i], full_matrices=False)
        V2 = Vc[:2]
        p = zz_w[i] @ V2.T                                   # (T_win, 2) in-plane coordinates
        res = zz_w[i] - p @ V2                               # (T_win, d) out-of-plane residual
        r_in[i] = np.sqrt((p ** 2).sum(-1).mean())
        r_out[i] = np.sqrt((res ** 2).sum(-1).mean())
    return r_in / den, r_out / den


def _band_power(zz_c, k):
    """Latent variance in the shedding band of each case (all latent dims pooled)."""
    F = np.fft.rfft(zz_c - zz_c.mean(axis=1, keepdims=True), axis=1)
    Pw = (np.abs(F) ** 2).sum(-1)
    Pw[:, 0] = 0.0
    band = np.array([Pw[i, max(k[i] - BAND_HALF_BINS, 1):k[i] + BAND_HALF_BINS + 1].sum()
                     for i in range(len(Pw))])
    return band, Pw


def _parse_tau(ids):
    return np.array([int(str(c).split("_")[3]) for c in ids])


def compute(z, src: Path) -> dict:
    zt, zar, zos, t = z["z_true_test"], z["z_ar"], z["z_os"], z["t_grid"]
    S = zar.shape[0]
    m2 = t >= T_STEADY - 1e-9
    mlate = t >= T_LATE - 1e-9
    ts = t[m2]
    dt = float(ts[1] - ts[0])

    power = {k: [_power(zz[s], zt, m2) for s in range(S)] for k, zz in (("ar", zar), ("os", zos))}
    rad = {k: np.concatenate([_radius(zz[s], zt, m2) for s in range(S)]) for k, zz in (("ar", zar), ("os", zos))}
    rad_late = {k: np.concatenate([_radius(zz[s], zt, mlate) for s in range(S)]) for k, zz in (("ar", zar), ("os", zos))}

    zt_c = _centred(zt, zt, m2)
    Pw_t = (np.abs(np.fft.rfft(zt_c - zt_c.mean(axis=1, keepdims=True), axis=1)) ** 2).sum(-1)
    Pw_t[:, 0] = 0.0
    k_t = np.argmax(Pw_t, axis=1)                 # dominant truth bin per case
    b_t, _ = _band_power(zt_c, k_t)
    freqs = np.fft.rfftfreq(int(m2.sum()), dt)
    band_ratio = {}
    for k, zz in (("ar", zar), ("os", zos)):
        r = []
        for s in range(S):
            b, _ = _band_power(_centred(zz[s], zt, m2), k_t)
            r.append(b / b_t)
        band_ratio[k] = np.concatenate(r)

    # phase winding of the TRUTH across the U_max ladder (train + test, 782 cases)
    z_all = np.concatenate([z["z_true_train"], zt])
    th_all = np.concatenate([z["theta_train"], z["theta_test"]])
    ids_all = np.concatenate([z["ids_train"], z["ids_test"]])
    tau_all = _parse_tau(ids_all)
    umax_all = th_all[:, 0]
    zc_all = z_all[:, m2] - z_all[:, m2].mean(axis=1, keepdims=True)
    pc1 = (zc_all @ z["pca_Vt2"].T)[..., 0]
    phi = np.angle(hilbert(pc1 - pc1.mean(1, keepdims=True), axis=1))
    k_star = int(np.argmin(np.abs(ts - T_STAR)))
    omega = np.array([np.polyfit(ts, np.unwrap(phi[i]), 1)[0] for i in range(len(phi))])

    pairs_hi, pairs_lo, wraps = [], [], {}
    for tv in np.unique(tau_all):
        sel = np.where(tau_all == tv)[0]
        sel = sel[np.argsort(umax_all[sel])]
        dphi = np.angle(np.exp(1j * np.diff(phi[sel, k_star])))
        mid_u = 0.5 * (umax_all[sel][1:] + umax_all[sel][:-1])
        pairs_hi.append(np.abs(dphi[mid_u >= UMAX_SPLIT]) > QUARTER_TURN)
        pairs_lo.append(np.abs(dphi[mid_u < UMAX_SPLIT]) > QUARTER_TURN)
        wraps[int(tv)] = float(T_STAR * (omega[sel].max() - omega[sel].min()) / (2 * np.pi))
    pairs_hi = np.concatenate(pairs_hi)
    pairs_lo = np.concatenate(pairs_lo)

    # orbit radius split in / out of the case's own orbit plane (truth over cases,
    # AR and one-shot over cases x seeds, plus the per-seed medians)
    radius_split = {}
    for wname, wmask in (("steady", m2), ("late", mlate)):
        t_in, t_out = _radius_split(zt, zt, wmask, m2)
        blk = {"truth_in": float(np.median(t_in)), "truth_out": float(np.median(t_out)), "per_seed": {}}
        for k, zz in (("ar", zar), ("os", zos)):
            parts = [_radius_split(zz[s], zt, wmask, m2) for s in range(S)]
            blk[f"{k}_in"] = float(np.median(np.concatenate([p[0] for p in parts])))
            blk[f"{k}_out"] = float(np.median(np.concatenate([p[1] for p in parts])))
            blk["per_seed"][f"{k}_in"] = [float(np.median(p[0])) for p in parts]
            blk["per_seed"][f"{k}_out"] = [float(np.median(p[1])) for p in parts]
        radius_split[wname] = blk
    radius_split["definition"] = ("RMS over the window (steady t >= 2 s, late t >= 2.5 s) of the latent's projection "
                                  "onto (in) and residual from (out) the case's own leading 2D SVD plane of the centred "
                                  "truth over t >= 2 s, centred on the truth's t >= 2 s mean, over the truth's own RMS "
                                  "excursion in the window; medians over cases (truth) or cases x seeds (AR, one-shot)")

    # exhibit case for the figure: median truth amplitude among the test cases
    amp = np.sqrt((zt_c ** 2).sum(-1).mean(1))
    ci = int(np.argsort(amp)[len(amp) // 2])
    case_id = str(z["ids_test"][ci])

    return {
        "source": {"latents_npz": str(src.relative_to(_REPO) / "latents.npz"), "sha256": _sha(src / "latents.npz")},
        "definitions": {
            "steady_window_s": T_STEADY, "late_window_s": T_LATE, "t_star_s": T_STAR,
            "umax_split": UMAX_SPLIT, "neighbour_gap_threshold": "quarter turn (pi/2)",
            "shedding_band": f"dominant truth bin +- {BAND_HALF_BINS} bin, df = {freqs[1]:.3f} Hz",
            "phase": "analytic-signal phase of the first global PC of the truth latents",
            "radius": "RMS distance to the truth's steady-window mean, over the truth's own",
        },
        "n_test_cases": int(zt.shape[0]), "n_all_cases": int(z_all.shape[0]), "seeds": [int(s) for s in z["seeds"]],
        "latent_power": {k: [float(x) for x in v] for k, v in power.items()},
        "latent_power_mean": {k: float(np.mean(v)) for k, v in power.items()},
        "radius_median": {k: float(np.median(v)) for k, v in rad.items()},
        "radius_late_median": {k: float(np.median(v)) for k, v in rad_late.items()},
        "band_ratio_median": {k: float(np.median(v)) for k, v in band_ratio.items()},
        "truth_shedding_freq_median_hz": float(np.median(freqs[k_t])),
        "winding": {
            "frac_neighbours_over_quarter_turn_umax_ge_split": float(pairs_hi.mean()),
            "frac_neighbours_over_quarter_turn_umax_lt_split": float(pairs_lo.mean()),
            "n_pairs_ge_split": int(len(pairs_hi)), "n_pairs_lt_split": int(len(pairs_lo)),
            "wraps_across_ladder_at_t_star_median": float(np.median(list(wraps.values()))),
            "wraps_across_ladder_at_t_star_min": float(min(wraps.values())),
            "wraps_across_ladder_at_t_star_max": float(max(wraps.values())),
            "wraps_per_tau": wraps,
        },
        "exhibit": {"case_id": case_id, "test_index": ci, "tau_index": int(case_id.split("_")[3]),
                    "seed_index": 0, "rule": "median truth amplitude among the test cases"},
        "radius_split": radius_split,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="recompute and compare against the stored JSON")
    ap.add_argument("--src", default=SRC_DEFAULT,
                    help="directory with latents.npz + stats.json of one pair (repo-relative); "
                         f"default {SRC_DEFAULT} (GRU pair), S4D pair: results/latent_orbits_s4d/interp")
    args = ap.parse_args(argv)
    SRC = (_REPO / args.src).resolve()
    OUT = SRC / OUT_NAME
    z = np.load(SRC / "latents.npz")
    stats = compute(z, SRC)

    ref = json.loads((SRC / "stats.json").read_text())["P_latent"]
    dev = max(abs(a - b) for k in ("ar", "os") for a, b in zip(stats["latent_power"][k], ref[k]))
    print(f"[orbit-stats] latent power AR {stats['latent_power']['ar']} OS {stats['latent_power']['os']} "
          f"(max |delta| to stats.json {dev:.1e})")
    if dev > 1e-6:
        print("[orbit-stats] FAIL: latent power does not reproduce stats.json of the re-inference")
        return 1
    print(f"[orbit-stats] radius median AR {stats['radius_median']['ar']:.3f} OS {stats['radius_median']['os']:.3f}; "
          f"late OS {stats['radius_late_median']['os']:.3f}; band ratio AR {stats['band_ratio_median']['ar']:.3f} "
          f"OS {stats['band_ratio_median']['os']:.3f}")
    w = stats["winding"]
    print(f"[orbit-stats] neighbours > quarter turn: U>={UMAX_SPLIT} {w['frac_neighbours_over_quarter_turn_umax_ge_split']:.3f}, "
          f"U<{UMAX_SPLIT} {w['frac_neighbours_over_quarter_turn_umax_lt_split']:.3f}; wraps median "
          f"{w['wraps_across_ladder_at_t_star_median']:.1f}; exhibit {stats['exhibit']['case_id']}")
    if args.check:
        old = json.loads(OUT.read_text())
        # round-trip through JSON so the int keys of wraps_per_tau compare as the str keys they are on disk
        same = old == json.loads(json.dumps(stats))
        print("[orbit-stats]", "PASS - identical to stored JSON" if same else "FAIL - differs from stored JSON")
        return 0 if same else 1
    OUT.write_text(json.dumps(stats, indent=1) + "\n")
    print(f"[orbit-stats] wrote {OUT.relative_to(_REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
