from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dropout_utils import calibrate_constant_daily_hazard
from vani_env import EnvConfig, PARAMS_DIR
import experiment


ROOT = Path(__file__).resolve().parents[1]
PARAMS = ROOT / "env_para_vanilla"
DROPOUT_TABLE = ROOT / "docs" / "dropout_check_20261017" / "participant_dropout.csv"
GOLDEN = ROOT / "tests" / "fixtures" / "dropout_hazard_zero_random_send.npz"


class _FreezeAgent:
    """Records agent calls; skips PF/RLSVI internals."""

    needs_belief = False

    def __init__(self):
        self.begin_weeks: list[int] = []
        self.rlsvi_weeks: list[int] = []
        self.acts: list[tuple[int, int, int]] = []
        self.rng = np.random.default_rng(0)

    def reset(self, dataset, week0_actions=None):
        self.dataset = dataset
        if week0_actions is not None:
            dataset.bootstrap_week0(week0_actions, self.rng)

    def begin_week(self, k, packet):
        self.begin_weeks.append(int(k))
        return 1

    def update_rlsvi(self, k):
        self.rlsvi_weeks.append(int(k))

    def act(self, k, d, t, state):
        self.acts.append((int(k), int(d), int(t)))
        return 1, 1.0

    def results(self, dataset):
        return {
            "I": dataset.I_hist,
            "A": dataset.A_hist,
            "b_hat": dataset.b_hat_hist,
            "b_tilde": dataset.b_tilde_hist,
            "pi_A": dataset.pi_A_hist,
            "y_hat": None,
            "v_hat": None,
            "pf": {},
            "betas": np.empty((dataset.W, 0, 0), dtype=float),
        }


def test_calibration_formula_matches_hand_count():
    table = pd.read_csv(DROPOUT_TABLE)
    complete = table.loc[table["window_complete"].astype(bool)]
    dropout_day = pd.to_numeric(complete["dropout_day"], errors="coerce")
    event = dropout_day.notna() & dropout_day.le(71)
    exposure = np.where(
        event,
        dropout_day.to_numpy(dtype=float) - 1.0,
        71.0,
    )
    assert int(event.sum()) == 4
    assert int(np.sum(exposure)) == 3809
    got = calibrate_constant_daily_hazard(table)
    assert got["dropout_events"] == 4
    assert got["dropout_person_days"] == 3809
    assert got["dropout_risk_window_days"] == 71
    np.testing.assert_allclose(got["dropout_hazard_day"], 4 / 3809)


def test_envconfig_defaults_to_zero_when_calibration_missing(tmp_path):
    std = PARAMS / "std_params.json"
    params = PARAMS / "params_env_13.json"
    dest = tmp_path / "params"
    dest.mkdir()
    dest.joinpath("std_params.json").write_bytes(std.read_bytes())
    dest.joinpath("params_env_13.json").write_bytes(params.read_bytes())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cfg = EnvConfig(13, params_dir=dest, nweek=2)
    assert cfg.dropout_hazard_day == 0.0
    assert cfg.dropout_mechanism == "none"
    assert any("neither activity_process nor dropout_hazard_day" in str(w.message) for w in caught)


def test_activity_process_and_constant_hazard_do_not_stack(tmp_path):
    dest = tmp_path / "params"
    dest.mkdir()
    for name in (
        "std_params.json",
        "params_env_13.json",
        "Ew_pooled_linear_coefs.json",
    ):
        (dest / name).write_bytes((PARAMS / name).read_bytes())
    df_fit = PARAMS / "df_fit_11week.csv"
    if df_fit.is_file():
        (dest / df_fit.name).write_bytes(df_fit.read_bytes())
    calibration = json.loads((PARAMS / "generator_calibration.json").read_text())
    calibration["dropout_hazard_day"] = 1.0
    (dest / "generator_calibration.json").write_text(json.dumps(calibration))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cfg, env, oenv = experiment._make_online_env(13, seed=1, params_dir=dest)
    assert cfg.dropout_mechanism == "activity"
    assert cfg.activity_process is not None
    assert cfg.dropout_hazard_day == 0.0
    assert not any(
        "neither activity_process nor dropout_hazard_day" in str(w.message)
        for w in caught
    )

    def _forbid_constant_hazard(state):
        raise AssertionError("constant hazard was drawn on top of the activity path")

    env.dropout_hazard = _forbid_constant_hazard
    for d_global in range(oenv.D):
        oenv._maybe_dropout(d_global)
    assert oenv._activity_status is not None
    assert oenv.dropout_day == oenv._dropout_day_planned


def test_hazard_zero_matches_golden_random_send():
    gold = np.load(GOLDEN)
    seed = int(gold["seed"])
    for uid in gold["uids"]:
        uid = int(uid)
        _res, oenv = experiment.run_random_send(
            uid, seed=seed, params_dir=PARAMS, dropout_hazard_day=0.0
        )
        prefix = f"uid{uid}_"
        assert int(oenv.dropout_day) == int(gold[prefix + "dropout_day"])
        assert np.all(np.asarray(oenv._activity_status)[: oenv.D] == 0)
        assert oenv.n_gaps == 0
        for name in (
            "CAE_all",
            "pu_all",
            "action_all",
            "stepCountNext4HourAll",
            "pageViewNext4HourAll",
            "morningFitbitWearAll",
            "dailySurveyCompleteAll",
        ):
            np.testing.assert_array_equal(
                np.asarray(getattr(oenv, name), dtype=float),
                gold[prefix + name],
            )


def test_lambda_one_pi_one_drops_on_day_2_and_zeros_presence():
    seed = 11
    uid = 13
    _base, base_env = experiment.run_random_send(
        uid, seed=seed, params_dir=PARAMS, dropout_hazard_day=0.0
    )
    _drop, drop_env = experiment.run_random_send(
        uid, seed=seed, params_dir=PARAMS, dropout_hazard_day=1.0
    )
    assert drop_env.dropout_day == 2
    assert drop_env.n_gaps == 1
    assert int(np.asarray(drop_env._activity_status)[0]) == 0
    assert np.all(np.asarray(drop_env._activity_status)[1:drop_env.D] == 2)
    # Day 1 is active, so its page views and steps match the all-active episode.
    # Its stored wear is tomorrow morning and is forced to 0.
    np.testing.assert_array_equal(
        drop_env.pageViewNext4HourTruthAll[:2],
        base_env.pageViewNext4HourTruthAll[:2],
    )
    np.testing.assert_array_equal(
        drop_env.stepCountNext4HourTruthAll[:2],
        base_env.stepCountNext4HourTruthAll[:2],
    )
    assert drop_env.morningFitbitWearTruthAll[0] == 0.0
    assert drop_env.dailySurveyCompleteTruthAll[0] == base_env.dailySurveyCompleteTruthAll[0]
    assert np.isnan(drop_env.stepCountNext4HourAll[2:]).all()
    assert np.isnan(drop_env.pageViewNext4HourAll[2:]).all()
    assert np.all(drop_env.pageViewNext4HourTruthAll[2:] == drop_env._pv_absent)
    assert np.all(drop_env.dailySurveyCompleteTruthAll[1:] == 0.0)
    assert np.isnan(drop_env.dailySurveyCompleteAll[1:]).all()
    assert np.isnan(drop_env.morningFitbitWearAll[1:]).all()
    assert np.all(drop_env.morningFitbitWearTruthAll[1:] == 0.0)
    assert np.isnan(drop_env.dailyAnticipatedAffectObsAll[1:]).all()
    assert np.all(np.isfinite(drop_env.dailyAnticipatedAffectAll))
    assert np.isnan(drop_env.U1_all[1:]).all()
    assert np.isnan(drop_env.U2_all[1:]).all()
    assert np.all(np.isfinite(drop_env.U1_truth_all[1:]))
    assert np.isnan(drop_env.CAE_short_all[1:]).all()
    assert np.all(drop_env.wp_all[1:] == 0.0)
    assert np.all(np.isfinite(drop_env.CAE_all[1:]))
    assert np.all(np.isfinite(drop_env.stepCountNext4HourTruthAll))


def test_activity_path_matches_the_dropout_rng_stream():
    from activity_process import coefficients_from_record, simulate_activity_paths

    cfg, _env, oenv = experiment._make_online_env(13, seed=7, params_dir=PARAMS)
    coef, fit = coefficients_from_record(cfg.activity_process)
    rng = np.random.default_rng(
        experiment.OnlineEnv._dropout_seed_sequence(7, 13)
    )
    sim = simulate_activity_paths(1, oenv.D, coef, fit, oenv.nweek, rng)
    assert oenv._dropout_day_planned == int(sim["dropout_day"][0])
    assert oenv.n_gaps == int(sim["n_gaps"][0])
    assert oenv.inactive_days == int(sim["inactive_days"][0])
    np.testing.assert_array_equal(oenv._activity_status, sim["status"][0])


def test_inactive_day_zeros_presence_and_checkin_window():
    _cfg, _env, oenv = experiment._make_online_env(
        13, seed=3, params_dir=PARAMS, dropout_hazard_day=0.0
    )
    # Saturday inactive; Sunday–Tuesday inactive so that check-in is forced off.
    oenv._activity_status[:] = 0
    oenv._activity_status[5:9] = 1
    oenv._dropout_day_planned = -1
    oenv._reset_episode_state()
    oenv.start_week(0, 1)
    for d in range(experiment.N_RL_DAYS):
        for t in range(oenv.K):
            oenv.step_action(0, d, t, 1, 1)
    oenv._finalize_week(0)

    absent = oenv._pv_absent
    sat = slice(5 * oenv.K, 6 * oenv.K)
    assert np.all(oenv.pageViewNext4HourTruthAll[sat] == absent)
    assert np.all(oenv.pageViewNext4HourAll[sat] == absent)
    assert np.all(oenv.ws_interaction_all[sat] == 0.0)
    assert np.all(np.isnan(oenv.stepCountNext4HourAll[sat]))
    assert np.all(np.isfinite(oenv.stepCountNext4HourTruthAll[sat]))
    assert oenv.dailySurveyCompleteTruthAll[5] == 0.0
    assert oenv.dailySurveyCompleteAll[5] == 0.0
    assert np.isnan(oenv.dailyAnticipatedAffectObsAll[5])
    assert np.isfinite(oenv.dailyAnticipatedAffectAll[5])
    # Friday's row holds Saturday morning's wear.
    assert oenv.morningFitbitWearTruthAll[4] == 0.0
    assert oenv.morningFitbitWearAll[4] == 0.0
    assert oenv.wp_all[1] == 0.0
    assert oenv.wp_truth_all[1] == 0.0
    assert np.isnan(oenv.U1_all[1])
    assert np.isnan(oenv.U2_all[1])
    assert np.isfinite(oenv.U1_truth_all[1])
    assert np.isfinite(oenv.U2_truth_all[1])


def test_agent_freezes_after_dropout_week_and_omits_td():
    cfg, env, oenv = experiment._make_online_env(
        13, seed=5, params_dir=PARAMS, dropout_hazard_day=1.0
    )
    agent = _FreezeAgent()
    dataset = experiment.EpisodeDataset(cfg.nweek)
    week0, i_hist = experiment.shared_episode_exogenous(5, cfg.nweek)
    result = oenv.run_episode(agent, dataset, week0_actions=week0, I_hist=i_hist)
    assert oenv.dropout_day == 2
    assert agent.begin_weeks == [0]
    assert agent.rlsvi_weeks == []
    assert agent.acts == [(0, 0, 0), (0, 0, 1)]
    assert np.all(result["A"][0, 0] == 1)
    assert np.all(result["A"][0, 1:] == 0)
    assert np.all(result["A"][1:] == 0)
    assert np.all(result["I"][1:] == 0)
    assert 0 not in dataset.med_full_hist
    packet = oenv.get_week_packet(1)
    assert packet is not None
    assert packet.J_w == 0
    assert np.isfinite(packet.Y_prev)
