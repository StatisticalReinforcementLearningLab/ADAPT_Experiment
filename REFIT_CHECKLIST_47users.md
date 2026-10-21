# Refit checklist: 47-user cohort (2026-10-19, revised 2026-10-22)

Scope: one full refit after the October cohort/dropout/baseline-CAE/activity-process
changes. Scripts 1–3 and the activity-process fit run LOCALLY (raw data stay on the
laptop); their processed outputs are copied to the cluster by rsync (never via git);
4b onwards run on the CLUSTER; results are rsynced back. Between running script 3 locally
and pulling the refit back, the local `env_para_vanilla/` mixes new data files with the old
42-user parameters: do not run the simulator locally in that window.

## 0. Before starting (local)

- [ ] All code committed on `ste-refit-1009`. Code only: `00`–`10`, `est_prior.py`,
      `stability.py`, `vani_env.py`, `experiment.py`, `activity_process.py`,
      `ste_vanilla.py`, `aggregate.py`, `dropout_utils.py`, `tests/`.
      Not committed: `df_fit*`, `df_merged*`, `week1_cae_inputs.csv`,
      `env_para_*/*.json` that a refit rewrites, `pu_hier_posterior.nc`,
      `.pipeline_*.log`, `pf_cmp2*`.
      The simulator does not read `fourSC_support_aware_lag`,
      `activity_carryover_calibration`, or `interaction_pageview_calibration`.
- [ ] Local tests pass: `python -m pytest tests` (21 dropout + CAE-smoother tests)
      and `python -m pytest RL_agents`.
- [ ] Expected numbers to check against later (from `docs/dropout_check_20261017/`
      and `docs/cohort_checks_20261017/`):
      55 analysis starters before the high-CAE ceiling (54 span-eligible + force-included 379);
      47 modelled after dropping the 8 high-CAE users; dropouts 379/138/82/259 at
      days 9/58/62/63; `h_day = 4/3809 = 0.001050`.
      Baseline-CAE sources in the current 42-user table: 27 observed, 15 backcast, 0 model_mean
      (no 112/251/257/388/379 yet). After the refit expect the 4 no-CAE users as `model_mean`
      and 379 as `model_mean` or a short backcast; record the actual 47-user counts.

## 1. LOCAL: data pipeline (minutes)

`export ADAPR_COMBINED_DIR=<local folder with the raw extract>`. Set only
`ADAPR_PARAMS_DIR` if you need a non-default folder. Run in order; each step has
a check before the next.

- [ ] `python 01_extract_data.py`
      - [ ] prints 54 span-eligible + force-included 379 = 55 analysis starters; `EXCLUDE_IF_ALL_CAE_MISSING = False`
      - [ ] no-CAE users 112, 251, 257, 388 retained
      - [ ] `participant_dropout.csv` written: 64 starters, 55 complete, 9 still enrolled,
            4 dropouts with the days above
- [ ] `python 02_combine.py`
      - [ ] `study_day` = calendar offset (check a non-Monday starter: day 1 = first EOD task date)
      - [ ] `dropout_day` present and NaN for non-dropouts
- [ ] `python 03_standardize.py`
      - [ ] `study_day`, `dropout_day` carried into `df_fit.csv`; `std_params.json` rewritten
      - [ ] `week1_cae_inputs.csv` written next to `df_fit.csv`
(The engagement fit in `archive/superseded/` is superseded by 06; do not run it.)

- [ ] Activity process (gaps + dropout), LOCAL, right after script 03:
      ```
      python 04_fit_activity_process.py
      ```
      then copy the activity_process block into env_para_ste0.2/0.5/0.8 (on the cluster, after their re-tuning, see section 6).
      - [ ] counts: 58 gaps (49 ended, 9 open), 3537 risk-set days, 58 events (or the new extract's values; record them)
      - [ ] model 2 converges inside bounds; zero check (PV/FW/PJ/J on excluded rows) = 0
      - [ ] only one dropout mechanism: activity_process present => legacy dropout_* ignored
            (do NOT run the old `5b --dropout-only` loop)

## 2. LOCAL -> CLUSTER transfer

- [ ] `python 05_check_refit.py --stage data` (all PASS; WARN values recorded)
- [ ] On the cluster, FIRST pull the code and back up the old fit (section 3, first two items),
      so the backup is not overwritten by the transfer.
- [ ] rsync to the cluster (scp/rsync only; these are participant data, never git).
      Paths below are relative to the repo, and the extract copies are relative to
      `ADAPR_COMBINED_DIR` (call it `$C`):

      | File | Repo path | Extract path | Read by |
      | --- | --- | --- | --- |
      | fit panel | not in the repo | `$C/df_fit.csv` | 06, 07, 08 |
      | 11-week panel | `env_para_vanilla/df_fit_11week.csv` (gitignored; 07 writes it on the cluster) | — | 09 |
      | week-1 CAE inputs | not in the repo | `$C/week1_cae_inputs.csv` | 07 |
      | standardization | `env_para_vanilla/std_params.json` (tracked seed; transfer the script-03 rewrite) | — | 07, simulator |
      | dropout table | not in the repo | `$C/participant_dropout.csv` | 04, 05 |
      | daily activity | not in the repo | `$C/participant_dropout_daily.csv` | 04, 06 |
      | weekly check-ins | not in the repo | `$C/df_weekly_filled.csv` | 06 |

      `df_merged.csv` is not a cluster input. 07 no longer reads it.
      `env_para_vanilla/generator_calibration.json` is committed, including `activity_process`; the cluster reads it from git.
- [ ] On the cluster, set `ADAPR_COMBINED_DIR` to the folder holding the transferred files.

## 3. CLUSTER setup (`/n/home09/xliu08/ADAPT_clean`)

- [ ] `git fetch lab && git checkout ste-refit-1009 && git pull lab ste-refit-1009`
- [ ] Back up the current fit: `cp -r env_para_vanilla env_para_vanilla_42users_20261014`
      (also `env_para_ste0.2/0.5/0.8` → `*_42users_20261014`).
- [ ] `mkdir -p logs`; `module load python; mamba activate budgeted`.
- [ ] Set only `ADAPR_PARAMS_DIR` if you need a non-default folder.


## 4. CLUSTER: NUTS fits (`sbatch`, hours)

- [ ] `python 06_fit_pu.py` (defaults: 4 chains, 2000 tune, 3000 draws, target_accept 0.95)
      - [ ] `mask_dropout_fit_outcomes` applied (log line); no post-dropout zeros in the fit panel
      - [ ] `participant_dropout_daily.csv` present in `ADAPR_COMBINED_DIR` (written by script 1,
            covering every df_fit day incl. day-85 Sundays); 4b errors if it is missing
      - [ ] PV/FW/PJ emission likelihood evaluated on active days only, J_w only for weeks whose
            survey day was active (log line with the numbers excluded); weekly PV/FW/PJ means for
            the E_w transition unchanged
      - [ ] 0 divergences, max R-hat ≤ 1.01, min bulk ESS ≥ 400 for every model
      - [ ] `pu_hier_posterior.nc` written (not committed)
- [ ] `python 07_fit_testbed.py`
      - [ ] `user_ids.txt` has 47 ids; 379 present; 82/138/259 present
      - [ ] high-CAE ceiling still removes 8 users; no-CAE users are *not* removed
      - [ ] seven hierarchical models pass the same diagnostics as above
      - [ ] baseline CAE: `docs/cohort_checks_*/cae_baseline_imputation.csv` regenerated;
            `cae_baseline_source` counts: 42-user table was observed 27 / backcast 15 / model_mean 0;
            expect roughly observed 27 / backcast 15–16 / model_mean 4–5 for 47 users — record the actual counts
      - [ ] every `params_env_<uid>.json` has `cae_baseline_norm` and `cae_baseline_source`
      - [ ] CAE / reward / Q fits used only CAE users (`_cae_users` log line)
- [ ] `python 08_est_ew_weights.py`

## 5. CLUSTER: priors and stability (minutes)

- [ ] after 08: `python 05_check_refit.py --stage fit`

- [ ] `python 09_est_prior.py` → `env_para_vanilla/rl_priors.json`
      - [ ] PF priors fitted on all 47; reward/Q priors on CAE users only
      - [ ] fourSC dimension unchanged (17) unless the covariate set changed
- [ ] `python 09_est_prior.py --loo` → `env_para_vanilla/loo_priors/held_out_<uid>.json`, 47 files
- [ ] `python 10_stability.py` → max |g^E|, |g^Y| < 1 for all 47 × {never, always}
- [ ] `python 05_check_refit.py --stage priors`
      (2000 activity paths, stability, loo files, and pytest, including RL_agents when present)
- [ ] Copy `rl_priors.json` to `RL_agents/rl/rl_priors.json` and record the diff of `nu_0`.

## 6. CLUSTER: STE environments (sbatch, sequential; days)

The ste0.2/0.5/0.8 folders are derived from vanilla, so they must be re-tuned.

- [ ] `sbatch run_tune_ste.sh` (action knob → 0.2 / 0.5 / 0.8; auto-submits confirmation CQL jobs)
- [ ] `TUNE_PHASE=validate bash run_tune_ste.sh`, then `TUNE_PHASE=ratio`
- [ ] Re-tune any folder off target with `TUNE_PROXY_TO_TRUE=from:env_para_ste<t>`, validate again
- [ ] Each newly tuned `env_para_ste*/generator_calibration.json` carries the same `activity_process`
      block as vanilla; if tuning rewrote the file without it, copy it back
- [ ] STE reported **with dropout on** only; the `h = 0` mode is for tests.

## 7. CLUSTER: RL experiments

- [ ] `sbatch --export=ALL,PRIOR_MODE=loo,ENV_VARIANT=env_para_vanilla,RESULTS_ROOT=results_vanilla_loo_47 run_array.sh`
      (and the ste variants as before)
- [ ] `sbatch run_aggregate.sh`; aggregate now includes `dropout_day` per episode —
      report fraction dropped by week 12 / 36 and mean learned weeks alongside cumulative true CAE
- [ ] Rerun the (B) subgroup comparison on the 47 refit models: 43 non-dropout models vs 82/138/259/379
      (replaces the provisional 39-model result in `docs/dropout_check_20261017/report.md`)

## 8. Back to LOCAL

- [ ] Pull the new `env_para_*` folders back to the laptop (rsync; not via git)
- [ ] Update `docs/dropout_check_20261017/report.md` (B) and the slides
      (`slides/cohort_dropout_frames.tex`: cohort counts, hazard, baseline-source counts)
- [ ] Open items, not part of this refit:
      - `RL_agents`: apply the 14-day inactivity rule in deployment and stop pooling a user's
        weeks after it fires (pending Susan's answer on what the system does after disengagement)
      - `RL_agents`: PF initialisation for real participants without week-1 CAE
        (same backcast / model mean as `vani_env.make_initial_state`)
      - `est_prior.py` known issues: FQI not converged at 25 iterations; σ²_Y inflated by the
        placeholder 1.0; prior covariance confounds noise with heterogeneity
      - next raw extract: the 9 still-enrolled users finish → re-estimate `h_day`
