# ADAPT RCT simulation testbed

Simulation environment and RL algorithms for the ADAPT walking-suggestion MRT.
The fitted parameter folders are the public testbed. Participant-level tables
stay on disk and are not committed.

## Pipeline

Run from the repository root. `ADAPR_COMBINED_DIR` is the extract folder.
`ADAPR_PARAMS_DIR` selects a parameter folder (default `env_para_vanilla`).

| # | Script | Where | Inputs | Outputs | Purpose |
| --- | --- | --- | --- | --- | --- |
| 00 | `00_combine_raw.py` | LOCAL | `ADAPR_RAW_DIR` JSON exports | combined raw tables | Merge the raw daily exports |
| 01 | `01_extract_data.py` | LOCAL | combined raw tables | `participant_dropout.csv`, `participant_dropout_daily.csv`, `df_weekly_filled.csv` | Build the decision panel and the activity tables |
| 02 | `02_combine.py` | LOCAL | script-01 tables | `df_merged.csv` | One row per decision slot, with study day and dropout day |
| 03 | `03_standardize.py` | LOCAL | `df_merged.csv` | `df_fit.csv`, `env_para_vanilla/std_params.json`, `week1_cae_inputs.csv` | Log and z-score the fit panel; write week-1 CAE inputs |
| 04 | `04_fit_activity_process.py` | LOCAL | the two dropout tables | `activity_process` in `generator_calibration.json`, report under `docs/` | Fit the gap and dropout process |
| 05 | `05_check_refit.py` | both | combined dir, or the parameter folder | PASS / WARN / FAIL lines | `--stage data` after 04; `--stage fit` after 08; `--stage priors` after 10 |
| 06 | `06_fit_pu.py` | CLUSTER | `df_fit.csv`, daily activity, `df_weekly_filled.csv` | `params_env_*.json` (penalized block), `pu_hier_diagnostics.json` | Bayesian engagement-state fit |
| 07 | `07_fit_testbed.py` | CLUSTER | `df_fit.csv`, `week1_cae_inputs.csv`, `std_params.json` | generator coefficients, `user_ids.txt`, `df_fit_11week.csv`, `testbed_diagnostics.json` | Hierarchical mediator and CAE fits, including the week-1 baseline |
| 08 | `08_est_ew_weights.py` | CLUSTER | `df_fit.csv`, `user_ids.txt` | `Ew_pooled_linear_coefs.json` | Pooled linear filter for the engagement state |
| 09 | `09_est_prior.py` | CLUSTER | `df_fit_11week.csv` | `rl_priors.json`, `loo_priors/` | RLSVI priors. The module remains `est_prior.py` |
| 10 | `10_stability.py` | CLUSTER | `params_env_*.json` | `stability_skeleton.json` | Engagement and outcome gains. The module remains `stability.py` |

After the refit, on the cluster: `tune_ste.py` (via `run_tune_ste.sh`) builds the
STE parameter folders; `run_array.sh` and `run_ste.sh` launch experiments;
`run_aggregate.sh` calls `aggregate.py`.

The simulator does not read `fourSC_support_aware_lag`,
`activity_carryover_calibration`, or `interaction_pageview_calibration`.
Do not run the superseded generator-patch script in `archive/superseded/`.

## Library modules

`activity_process.py` draws the exogenous activity path and holds the shared
hazard, study-week scale, and dropout seed. `vani_env.py` is the generative
model. CAE, page views, Fitbit wear, the daily survey, weekly presence, and
perceived utility have no study-week slope. `experiment.py` runs an episode. `dropout_utils.py` defines an active day and
the dropout table. `algorithm_helpers.py`, `derived_covariates.py`,
`ewm_utils.py`, and `state_names.py` are shared by the fit and the simulator.
`est_prior.py` and `stability.py` stay importable under those names.
`ste_vanilla.py` is the STE evaluation used by `tune_ste.py`.

## Launchers

`run_array.sh` submits the experiment array. `run_aggregate.sh` aggregates a
results folder. `run_ste.sh` runs one STE environment. `run_tune_ste.sh` tunes
the action knob to the STE targets. Each one is a SLURM script; `bash -n`
checks the syntax, and the cluster runs them with `sbatch`.

## analysis/ and archive/

`analysis/` holds checks that are not pipeline steps: `check_trend.py`,
`plot_coefficients.py`, `simulated_dropout_check.py`, and the figure scripts
in `analysis/figures/`. Figure output goes to `analysis/figures/out/`, which
is gitignored.

`archive/superseded/` holds the old engagement fit and the generator-patch script.
`archive/analyses/`, `archive/diagnostics/`, `archive/notebooks/`, and
`archive/notes/` are earlier checks and write-ups. `archive/results/` and
`archive/backups/` are local and gitignored.

## Data that never goes into git

`df_fit.csv`, `df_fit_11week.csv`, `df_merged.csv`, `week1_cae_inputs.csv`,
`participant_dropout.csv`, `modelled_cohort.csv`, `participant_dropout_daily.csv`,
`df_weekly_filled.csv`, anything matching `env_para_*/df_*.csv`, `docs/**/*.csv`,
`*.pkl`, and the `results*/` folders. `params_env_*.json`, `pred_*.json`,
`user_ids.txt`, `loo_priors/`, and `stability_skeleton.json` stay tracked as
the published testbed and are overwritten in place by a refit; the new
participant tables are transferred with rsync, not committed. The checklist
is `REFIT_CHECKLIST_47users.md`.

## Setup

Python 3.10+ recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export ADAPR_COMBINED_DIR=/path/to/extracted/tables
# script 00 only:
export ADAPR_RAW_DIR=/path/to/ADAPT_MRT
```

```bash
python experiment.py --seed 0
bash run_array.sh
python aggregate.py
```

`ADAPR_PARAMS_DIR=env_para_ste0.8` selects an STE copy.
