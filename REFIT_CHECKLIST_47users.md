# Refit checklist: 47-user cohort (2026-10-19, revised 2026-10-22)

Scope: one full refit after the October cohort/dropout/baseline-CAE/activity-process
changes. Scripts 1–3 and the activity-process fit run LOCALLY (raw data stay on the
laptop); their processed outputs are copied to the cluster by rsync (never via git);
4b onwards run on the CLUSTER; results are rsynced back. Between running script 3 locally
and pulling the refit back, the local `env_para_vanilla/` mixes new data files with the old
42-user parameters: do not run the simulator locally in that window.

## 0. Before starting (local)

- [ ] All code committed on `ste-refit-1009` and pushed to `lab` and `origin`. Code only:
      scripts 1–5, `5b_patch_generator.py`, `6_est_Ew_weights.py`, `est_prior.py`,
      `vani_env.py`, `experiment.py`, `ste_vanilla.py`, `aggregate.py`,
      `dropout_utils.py`, `tests/`, `scripts/dropout_*.py`, `docs/`.
      Not committed: `df_fit*`, `df_merged*`, `env_para_*/*.json`,
      `pu_hier_posterior.nc`, `.pipeline_*.log`, `pf_cmp2*`.
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

`export ADAPR_COMBINED_DIR=<local folder with the raw extract>`. Run in order; each step has
a check before the next.

- [ ] `python 1_data_extraction.py`
      - [ ] prints 54 span-eligible + force-included 379 = 55 analysis starters; `EXCLUDE_IF_ALL_CAE_MISSING = False`
      - [ ] no-CAE users 112, 251, 257, 388 retained
      - [ ] `participant_dropout.csv` written: 64 starters, 55 complete, 9 still enrolled,
            4 dropouts with the days above
- [ ] `python 2_combine_data_frame.py`
      - [ ] `study_day` = calendar offset (check a non-Monday starter: day 1 = first EOD task date)
      - [ ] `dropout_day` present and NaN for non-dropouts
- [ ] `python 3_standardization.py`
      - [ ] `study_day`, `dropout_day` carried into `df_fit.csv`; `std_params.json` rewritten
(`4_perceived_utility.py` is superseded by 4b, which writes the same outputs; do not run it.)

- [ ] Activity process (gaps + dropout), LOCAL, right after script 3. Reads
      `participant_dropout_daily.csv` and `participant_dropout.csv` from
      `ADAPR_COMBINED_DIR` (script 1; survey-inclusive active definition,
      protocol weeks 1-12, table through day 87). `ADAPR_ACTIVITY_DAILY`
      overrides the daily file; the participant table is the sibling in that
      directory. The script errors if either file is missing. Figures and
      `report.md` go to `--out` (default `docs/activity_process_<YYYYMMDD>/`,
      created if absent):
      ```
      python scripts/fit_activity_process.py            # writes env_para_vanilla/generator_calibration.json["activity_process"]
      ```
      then copy the activity_process block into env_para_ste0.2/0.5/0.8 (on the cluster, after their re-tuning, see section 6).
      - [ ] counts: 58 gaps (49 ended, 9 open), 3537 risk-set days, 58 events (or the new extract's values; record them)
      - [ ] model 2 converges inside bounds; zero check (PV/FW/PJ/J on excluded rows) = 0
      - [ ] only one dropout mechanism: activity_process present => legacy dropout_* ignored
            (do NOT run the old `5b --dropout-only` loop)

## 2. LOCAL -> CLUSTER transfer

- [ ] `python scripts/check_refit.py --stage data` (all PASS; WARN values recorded)
- [ ] On the cluster, FIRST pull the code and back up the old fit (section 3, first two items),
      so the backup is not overwritten by the transfer.
- [ ] rsync to the cluster (scp/rsync only; these are participant data, never git):
      the processed files that 4b, 5, 6, est_prior and the simulator read — at least
      `df_fit.csv` / `df_fit_11week.csv`, `std_params.json`, `participant_dropout.csv`,
      `participant_dropout_daily.csv`, and `env_para_vanilla/generator_calibration.json`
      (with the new `activity_process` block). The exact list comes from Cursor's
      "files read by the cluster steps" inventory; transfer exactly that list.
- [ ] On the cluster, set `ADAPR_COMBINED_DIR` to the folder holding the transferred files.

## 3. CLUSTER setup (`/n/home09/xliu08/ADAPT_clean`)

- [ ] `git fetch lab && git checkout ste-refit-1009 && git pull lab ste-refit-1009`
- [ ] Back up the current fit: `cp -r env_para_vanilla env_para_vanilla_42users_20261014`
      (also `env_para_ste0.2/0.5/0.8` → `*_42users_20261014`).
- [ ] `mkdir -p logs`; `module load python; mamba activate budgeted`.


## 4. CLUSTER: NUTS fits (`sbatch`, hours)

- [ ] `python 4b_perceived_utility_bayes.py` (defaults: 4 chains, 2000 tune, 3000 draws, target_accept 0.95)
      - [ ] `mask_dropout_fit_outcomes` applied (log line); no post-dropout zeros in the fit panel
      - [ ] `participant_dropout_daily.csv` present in `ADAPR_COMBINED_DIR` (written by script 1,
            covering every df_fit day incl. day-85 Sundays); 4b errors if it is missing
      - [ ] PV/FW/PJ emission likelihood evaluated on active days only, J_w only for weeks whose
            survey day was active (log line with the numbers excluded); weekly PV/FW/PJ means for
            the E_w transition unchanged
      - [ ] 0 divergences, max R-hat ≤ 1.01, min bulk ESS ≥ 400 for every model
      - [ ] `pu_hier_posterior.nc` written (not committed)
- [ ] `python 5_fit_vanilla_testbed.py`
      - [ ] `user_ids.txt` has 47 ids; 379 present; 82/138/259 present
      - [ ] high-CAE ceiling still removes 8 users; no-CAE users are *not* removed
      - [ ] seven hierarchical models pass the same diagnostics as above
      - [ ] baseline CAE: `docs/cohort_checks_*/cae_baseline_imputation.csv` regenerated;
            `cae_baseline_source` counts: 42-user table was observed 27 / backcast 15 / model_mean 0;
            expect roughly observed 27 / backcast 15–16 / model_mean 4–5 for 47 users — record the actual counts
      - [ ] every `params_env_<uid>.json` has `cae_baseline_norm` and `cae_baseline_source`
      - [ ] CAE / reward / Q fits used only CAE users (`_cae_users` log line)
- [ ] `python 6_est_Ew_weights.py`

## 5. CLUSTER: priors and stability (minutes)

- [ ] after 4b/5/6: `python scripts/check_refit.py --stage fit`

- [ ] `python est_prior.py` → `env_para_vanilla/rl_priors.json`
      - [ ] PF priors fitted on all 47; reward/Q priors on CAE users only
      - [ ] fourSC dimension unchanged (17) unless the covariate set changed
- [ ] `python est_prior.py --loo` → `env_para_vanilla/loo_priors/held_out_<uid>.json`, 47 files
- [ ] `python stability.py` → max |g^E|, |g^Y| < 1 for all 47 × {never, always}
- [ ] `python scripts/dropout_simulator_smoke.py` on 3 users
      (reads `participant_dropout.csv` from `ADAPR_COMBINED_DIR`; errors if missing)
      - [ ] activity paths: terminal gap >= 14 days by day 84 near 9% (data 8.5%); inactive-day share near 9.5%
      - [ ] lambda = 0 reproduces the pre-activity simulator bit-for-bit (regression test)
      - [ ] activity path seeded by (episode seed, uid): different users differ; same (seed, uid) identical across algorithms
- [ ] `python -m pytest tests RL_agents`
- [ ] `python scripts/check_refit.py --stage priors`
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
