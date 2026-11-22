# ============================================================================
# Purpose: the model registry of the paper. It maps each reported model row to
#   its run directories and core-JSON key (KVS) or its re-inference key (CMP),
#   and lists the metric columns (KVS_2A_CLEAN, CMP_2B, KVS_METRICS,
#   CMP_METRICS).
# Pipeline role: read by scripts/build_paper_matrix.py (matrix assembly) and by
#   the KVS scorers scripts/kvs_paper_diagnose.py and
#   scripts/onset_oscmed_apply.py, so that every stage uses the same model list.
# Eval-consistency: a wrong core_key, reinfer_key or seed list here redirects
#   the matrix to another run's numbers without an error. The seed-count check
#   in build_paper_matrix.py catches only count mismatches, not wrong run
#   directories with the right count.
# ============================================================================
"""Model list of the paper (allow-list in code).

Only these models enter the paper. Every entry has key, label, role, cell and n, plus per
split the run directories (KVS) or the re-inference JSON stem (CMP). Run directories live
under results/benchmark/runs/. Seed sets differ between models, so they are listed
explicitly. The empty direct x AR cell of the design has no entry (N/A in the matrix).
"""
from __future__ import annotations

# ----------------------------------------------------------------------------- KVS
# Run directories per split: results/benchmark/runs/<rundir>/predictions.npz
# core_key = stem of the run_benchmark core JSON
#            (results/benchmark/kvs_<split>_<core_key>_<seed>.json)
# ----------------------------------------------------------------------------- CMP
# CMP metrics come from run_cmp3d_gate2 (re-inference) or cmp3d_dmdc_gate2 (DMDc).
# re-inference JSON: reports/_reinfer_fix/cmp3d_6metric_<split>_<reinfer_key>_fix_seed<seed>.json
# DMDc JSON:         reports/cmp3d_gate2_cmp3d_dmdc_k8[_extrap]_fix_seed1.json
CMP_2B = [
    # The S-DeepONet row is trained without the auxiliary WSS loss (wss_lambda = 0); its
    # re-inference JSONs carry the stem sdeeponet_nowss.
    dict(key="sdeeponet", label="S-DeepONet (field)", role="direct operator", cell="direct×one-shot", n=3,
         reinfer_key="sdeeponet_nowss", seeds=(1, 2, 3)),
    dict(key="mamba_k8", label="Mamba-DON k8", role="recurrent branch DeepONet", cell="latent×AR", n=3,
         reinfer_key="latent_ar_mamba_k8", seeds=(1, 2, 3)),      # non-tuned; tuned diverges
    dict(key="s4d_k8", label="S4D-DON k8", role="recurrent branch DeepONet", cell="latent×AR", n=3,
         reinfer_key="latent_ar_s4d_k8", seeds=(1, 2, 3)),
    dict(key="gru_k8", label="GRU-DON k8", role="recurrent branch DeepONet", cell="latent×AR", n=3,
         reinfer_key="latent_ar_gru_k8", seeds=(1, 2, 3)),
    dict(key="armlp_k8", label="AR-MLP k8", role="MLP baseline", cell="latent×AR", n=3,
         reinfer_key="latent_ar_armlp_k8", seeds=(1, 2, 3)),
    dict(key="dmdc_k8", label="DMDc k8", role="linear baseline", cell="latent×AR", n=1,
         reinfer_key=None, dmdc=True, seeds=(1,)),                # no network training, own scorer
    dict(key="oneshot_gru", label="GRU-branch one-shot", role="time-axis control", cell="latent×one-shot", n=3,
         reinfer_key="latent_oneshot_gru_tuned", seeds=(1, 2, 3)),
    dict(key="oneshot_s4d", label="S4D-branch one-shot", role="time-axis control", cell="latent×one-shot", n=3,
         reinfer_key="latent_oneshot_s4d_tuned", seeds=(1, 2, 3)),
]

# The 7 KVS metric columns of the matrix. Core metrics come from the run_benchmark JSON, the
# others from the diagnostic scorers. The onset column uses the oscMed10 detector. Strip RMSE
# is not a column: it correlates at 0.998 with rollout_rmse.
KVS_METRICS = ["rollout_rmse", "power_ratio", "rmse_warped",
               "onset_latency_oscmed10", "halluc_pixel_frac_pre", "alpha_rms", "fft_freq_err"]
KVS_CORE = {"rollout_rmse", "power_ratio", "rmse_warped", "fft_freq_err"}   # in kvs_<split>_*.json
# CMP metric columns. nrmse_* = error of the bulk velocity field; cumwss_rel_l2_inner_mean =
# the polishing-relevant target (cumulative wall shear stress, inner 90 % of the wafer
# radius). nrmse_inplane (u_x, u_y) and nrmse_z split nrmse_3d by component: u_z is at noise
# level with a small normaliser and dominates nrmse_3d. The paper table reports
# nrmse_inplane; nrmse_3d and nrmse_z are descriptive columns.
CMP_METRICS = ["nrmse_3d", "nrmse_inplane", "nrmse_z", "cumwss_rel_l2_inner_mean",
               "spatial_corr_inner90", "onset_latency_excess",
               "radial_profile_l2", "steady_drift_excess", "wiwnu_excess"]


def kvs_rundirs(model: dict, split: str):
    return model["interp"] if split == "interp" else model["extrap"]


# --- KVS registry of the paper ('clean', selected by KVS_PAPER_REGISTRY=clean, the default).
# Interp rows: runs on the interp ConvAE encoder (hash 47132e7b07), names ending in _clean.
# Extrap rows: the runs named *_dsspan_x for the three branch DeepONets and the plain names
# for the other models. All 8 rows, so that matrix and scorers build the full table from one
# registry.
def _clean(key, label, role, cell, n, interp_stem, extrap_stem, seeds_i, seeds_e, core_e=None):
    return dict(key=key, label=label, role=role, cell=cell, n=n,
                core_key=f"{interp_stem}", core_key_extrap=core_e or extrap_stem,
                interp=[f"{interp_stem}_interp_seed{s}" for s in seeds_i],
                extrap=[f"{extrap_stem}_extrap_seed{s}" for s in seeds_e])


KVS_2A_CLEAN = [
    _clean("s4d_1phase4k", "S4D-branch DON", "recurrent branch DeepONet", "latent×AR", 3,
           "s4d_1phase4k_clean", "s4d_1phase4k_dsspan_x", (1, 2, 3), (1, 2, 3)),
    _clean("mamba_1phase4k", "Mamba-branch DON", "recurrent branch DeepONet", "latent×AR", 3,
           "mamba_1phase4k_clean", "mamba_1phase4k_dsspan_x", (1, 2, 3), (1, 2, 3)),
    _clean("gru_1phase4k", "GRU-branch DON", "recurrent branch DeepONet", "latent×AR", 3,
           "gru_1phase4k_clean", "gru_1phase4k_dsspan_x", (1, 2, 3), (1, 2, 3)),
    _clean("ar_mlp", "AR-MLP", "MLP baseline", "latent×AR", 3,
           "ar_mlp_clean", "ar_mlp", (0, 1, 2), (0, 1, 2)),
    _clean("dmdc", "DMDc", "linear baseline", "latent×AR", 1,
           "kvs_dmdc_clean", "kvs_dmdc", (0,), (0,)),
    dict(key="field_deeponet", label="field DeepONet", role="direct operator", cell="direct×one-shot", n=3,
         core_key="kvs_field_deeponet_norm",
         interp=[f"kvs_field_deeponet_norm_interp_seed{s}" for s in (0, 1, 2)],
         extrap=[f"kvs_field_deeponet_norm_extrap_seed{s}" for s in (0, 1, 2)]),
    _clean("oneshot_gru", "GRU-branch one-shot", "time-axis control", "latent×one-shot", 3,
           "latent_oneshot_gru_tuned_clean", "latent_oneshot_gru_tuned", (1, 2, 3), (1, 2, 3)),
    _clean("oneshot_s4d", "S4D-branch one-shot", "time-axis control", "latent×one-shot", 3,
           "latent_oneshot_s4d_tuned_clean", "latent_oneshot_s4d_tuned", (1, 2, 3), (1, 2, 3)),
]


def registry_suffix() -> str:
    """File name suffix of the active KVS registry.

    'clean' is the only KVS registry this repository builds, so '_clean' is also the
    default. KVS_PAPER_REGISTRY=clean is accepted as an explicit switch; any other value
    raises ValueError.
    """
    import os
    r = os.environ.get("KVS_PAPER_REGISTRY", "clean")
    if r not in ("", "clean"):
        raise ValueError(f"KVS_PAPER_REGISTRY={r!r}: only 'clean' exists")
    return "_clean"


def active_kvs_2a():
    """KVS registry of the paper (KVS_2A_CLEAN)."""
    registry_suffix()          # validates KVS_PAPER_REGISTRY if it is set
    return KVS_2A_CLEAN


def kvs_core_key(model: dict, split: str) -> str:
    """core_key per split. Rows with core_key_extrap use it for extrap (e.g. the *_dsspan_x
    runs); the field DeepONet row has one key for both splits."""
    if split == "extrap" and "core_key_extrap" in model:
        return model["core_key_extrap"]
    return model["core_key"]


# --- Column and source tables -------------------------------------------------------------
# KVS: registry suffix -> (onset NPZ stem, onset column name). The NPZ is written by
# scripts/onset_oscmed_apply.py (oscMed10 detector).
KVS_ONSET_SOURCE = {
    "_clean":  dict(stem="onset_oscmed10_per_model_clean",  column="onset_latency_oscmed10"),
}


def kvs_metrics(suffix: str | None = None) -> list:
    """KVS column list of the active registry."""
    return list(KVS_METRICS)


def cmp_metrics(chain: str) -> list:
    """CMP column list (9 columns, including the in-plane split)."""
    return list(CMP_METRICS)


def all_core_keys() -> set:
    """All core_keys of the registry (both splits).

    Used by scripts/build_paper_matrix.py: all core JSONs sit in one directory, and a longer
    core_key can start with a shorter one (ar_mlp and ar_mlp_clean). Without this set the
    matrix builder could not drop the files that belong to the longer key.
    """
    keys = set()
    for reg in (KVS_2A_CLEAN,):
        for m in reg:
            keys.add(m["core_key"])
            if "core_key_extrap" in m:
                keys.add(m["core_key_extrap"])
    return keys
