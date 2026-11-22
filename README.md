# No Free Lunch in Flow Surrogates under Time-Varying Boundary Conditions: A Two-Regime Study

<img src="assets/funding_logos.png" alt="Co-funded by the European Union. Free State of Saxony." height="80">

We sincerely thank the European Social Fund (ESF) and the Free State of Saxony of the Federal
Republic of Germany for funding this work under grant number 100693458 (project WiProFlex).

Code and result files for the paper by Georg Winkler and Martin Stoll, *Journal of Computational
Science* (2026), DOI: [10.1016/j.jocs.2026.103075](https://doi.org/10.1016/j.jocs.2026.103075).

The paper compares data-driven surrogate models in two flow regimes that are both driven by a
time-varying boundary condition: a Karman vortex street behind a cylinder with a ramped inflow
("wake", KVS) and a thin Stokes film between a rotating pad and wafer in chemical-mechanical
polishing ("film", CMP). This repository contains the model and training code, the configuration of
every reported model row, the per-seed result files, and the scripts that turn them into the numbers
and tables of the paper.

## Contents

| Path | Content |
|---|---|
| `benchmark/`, `tools/` | models, training and inference adapters, metrics, data loading |
| `configs/` | one YAML file per reported model row: wake interpolation in `configs/clean/`, wake extrapolation in `configs/paper_kvs_extrap_*.yaml`, film in `configs/paper_cmp3d_*_fix.yaml` |
| `results/benchmark/`, `reports/_reinfer_fix/` | per-seed metric files of every reported row (wake and film) |
| `reports/` | further result files behind individual numbers (onset detector, diagnostics, POD spectra, timings) |
| `reports/PAPER_MATRIX_CLEAN/kvs_matrix.csv`, `reports/PAPER_MATRIX_FIX/cmp_matrix.csv` | the two result matrices (mean and standard deviation over seeds) |
| `values/emit_values.py`, `values/paper_values.tex` | turns the matrices and result files into the LaTeX macros printed in the paper |
| `scripts/` | training entry points and the analyses behind individual numbers |
| `benchmark/splits/`, `meshes/` | the fixed train/validation/test splits, evaluation nodes and the wake mesh |

## Checking the printed numbers

Every number in the result tables and the text of the paper is a macro in `values/paper_values.tex`.
The check rebuilds both result matrices from the per-seed result files and emits the macros again;
all three must equal the committed files byte for byte. It runs on a CPU in a few seconds:

```bash
conda env create -f environment.yml
conda activate kvs_repro
make check
```

## Where the tables come from

| Paper item | Produced by | Inputs |
|---|---|---|
| Table 4 (film results), Table 5 (wake results) | `scripts/build_paper_matrix.py`, then `values/emit_values.py` | per-seed files in `results/benchmark/` and `reports/_reinfer_fix/`, onset and diagnostic files in `reports/` (all included) |
| Table 6 (cost) | `values/emit_values.py` | `reports/fem_scaling.json`, `reports/inference_timing.json`, the encoder training time in `results/benchmark/kvs_interp_convae_83x442_clean_0.json` (included) |
| Numbers in the text | `values/emit_values.py` | result files written by the scripts listed below (included) |

The figure scripts are not part of this package. The data behind the quantitative figures are in
this repository: the champion figures (Figs. 8 and 11) plot the two result matrices,
Fig. 6 plots `reports/cmp3d_pod_spectrum_fix.json` and `reports/kvs_pod_spectrum.json`, Fig. 10
plots `results/benchmark/cmp_rollout_error_growth_interp.json`, and Fig. 12 plots the latent
trajectories in `results/latent_orbits_gru/interp/`.

Scripts behind individual numbers in the text, each writing the result file that
`values/emit_values.py` reads:

| Number | Script | Result file |
|---|---|---|
| encoder resolution floor (wake) | `scripts/encoder_roundtrip_floor.py` | `results/benchmark/kvs_{interp,extrap}_convae_83x442_floor_0.json` |
| POD floor (film) | `scripts/cmp_encoder_floor.py` | `results/benchmark/cmp3d_encoder_floor_fix.json` |
| rollout error growth (film) | `scripts/cmp_rollout_error_growth.py` | `results/benchmark/cmp_rollout_error_growth_interp.json` |
| latent orbits, phase winding, sampling bound | `scripts/latent_orbits.py`, `scripts/latent_orbit_stats.py` | `results/latent_orbits_{gru,s4d}/interp/orbit_paper_stats.json` |
| mirror branch | `scripts/mirror_aligned_rmse.py` | `reports/kvs_mirror_aligned_rmse.json` |
| last vs best checkpoint of the field DeepONet | `scripts/field_don_checkpoint_probe_check.py` | `results/benchmark/field_don_checkpoint_probe_check_interp.json` |
| Strouhal gate | `scripts/strouhal_gate_count.py` | `results/benchmark/kvs_strouhal_gate_counts.json` |
| shedding period and critical inflow | `scripts/kvs_shedding_period_range.py` | `results/benchmark/kvs_truth_shedding_period.json` |
| POD spectra | `scripts/kvs_pod_spectrum.py`, `scripts/cmp3d_pod_spectrum.py` | `reports/kvs_pod_spectrum.json`, `reports/cmp3d_pod_spectrum_fix.json` |

These scripts read predictions, checkpoints or the simulation data, which are not included.

## Training and inference

The simulation data and the trained checkpoints are not part of this repository; they are available
from the authors on reasonable request (see the Data availability statement of the paper). The code
expects them at these paths:

| Input | Path |
|---|---|
| wake fields (one HDF5 file per case) | `results/refined_4x_2026-05-07/chunk_*/<case>/fields.h5` |
| wake fields on the 83 x 442 grid | `results/benchmark/cache/kvs_grid_83x442.npz` |
| wake truth on the evaluation nodes | `results/benchmark/truth/kvs_{interp,extrap}.npz` |
| wake node pool of the field DeepONet | `data/kvs/field_cache_pool24000_8000/` (without it, remove `kvs_field_cache` from the config and the adapter draws the pool from the HDF5 files) |
| film training data | `data/cmp/cmp3d_full100fix_fem_training.npz` |
| frozen wake encoders | `results/benchmark/encoders_clean/kvs_interp_convae_83x442_bc32_47132e7b07.pt`, `results/benchmark/encoders/kvs_extrap_convae_83x442_bc32.pt` |
| film POD encoders | `results/benchmark/encoders/cmp3d_{interp,extrap}_pod139_fix.npz` (built by `scripts/build_cmp3d_pod.py`) |
| run directories (checkpoints, predictions) | `results/benchmark/runs/` |

One wake row and one film row (seed 1), and the DMDc row of the film:

```bash
python scripts/run_benchmark.py --config configs/clean/paper_kvs_interp_s4d_1phase4k_clean.yaml --seed 1
python scripts/run_cmp3d_gate2.py --config configs/paper_cmp3d_interp_latent_ar_s4d_k8_fix.yaml --seed 1
python scripts/cmp3d_dmdc_gate2.py --split interp
```

`run_benchmark.py` trains, predicts on the test cases and scores the prediction; the scores land in
`results/benchmark/kvs_<split>_<model>_<seed>.json`. `run_cmp3d_gate2.py` trains and scores a film
row; `--load-best` re-scores an existing run directory from its `best.pt` without training. Pass
`--out reports/_reinfer_fix/cmp3d_6metric_<split>_<key>_fix_seed<n>.json` to write the score
under the name `scripts/build_paper_matrix.py` reads (the keys are listed in
`benchmark/paper_models.py`). Every stochastic row is reported over three seeds; the seed is the last
number in a result file name. Training the wake encoder
(`configs/clean/paper_kvs_interp_convae_83x442_clean.yaml`) writes
`results/benchmark/encoders/kvs_interp_convae_83x442_bc32.pt`; the latent rows refer to the encoder
used in the paper by its file name, so point `encoder_ref` to a retrained encoder to use it.

The models were trained on an NVIDIA RTX PRO 6000 GPU. The Mamba branch needs the `mamba-ssm`
package and a CUDA toolchain.

### Run identifiers

File and run names carry suffixes that identify the runs reported in the paper: `_clean` for the
wake interpolation rows, `_dsspan_x` for the wake extrapolation rows of the three branch DeepONets,
and `_fix` for the film rows and the film data set.

## Citation

```bibtex
@article{winkler2026nofreelunch,
  title   = {No Free Lunch in Flow Surrogates under Time-Varying Boundary Conditions: A Two-Regime Study},
  author  = {Winkler, Georg and Stoll, Martin},
  journal = {Journal of Computational Science},
  year    = {2026},
  doi     = {10.1016/j.jocs.2026.103075}
}
```

## License

The code is released under the MIT License, see `LICENSE`. Copyright (c) 2026 Technische
Universität Chemnitz. Authors: Georg Winkler and Martin Stoll.
