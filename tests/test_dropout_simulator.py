from __future__ import annotations

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


def _truncated_geometric_mean(hazard: float, horizon: int) -> float:
    q = 1.0 - hazard
    probs = hazard * q ** np.arange(horizon)
    mass = float(probs.sum())
    return float(np.dot(np.arange(1, horizon + 1), probs) / mass)


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
    assert any("dropout_hazard_day" in str(w.message) for w in caught)


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


def test_hazard_one_matches_never_send_latent_and_masks_observations():
    seed = 11
    uid = 13
    _never, never_env = experiment.run_never_send(
        uid, seed=seed, params_dir=PARAMS, dropout_hazard_day=0.0
    )
    _drop, drop_env = experiment.run_random_send(
        uid, seed=seed, params_dir=PARAMS, dropout_hazard_day=1.0
    )
    assert drop_env.dropout_day == 1
    np.testing.assert_allclose(drop_env.CAE_all, never_env.CAE_all)
    np.testing.assert_allclose(
        experiment.cumulative_true_cae(drop_env),
        experiment.cumulative_true_cae(never_env),
    )
    assert np.isnan(drop_env.stepCountNext4HourAll).all()
    assert np.isnan(drop_env.pageViewNext4HourAll).all()
    assert np.isnan(drop_env.morningFitbitWearAll).all()
    assert np.isnan(drop_env.dailySurveyCompleteAll).all()
    assert np.isnan(drop_env.dailyAnticipatedAffectObsAll).all()
    assert np.isnan(drop_env.U1_all[1:]).all()
    assert np.isnan(drop_env.U2_all[1:]).all()
    assert np.isnan(drop_env.CAE_short_all[1:]).all()
    assert np.all(drop_env.wp_all[1:] == 0.0)
    assert np.all(drop_env.action_all == 0.0)
    assert np.all(np.isfinite(drop_env.CAE_all[1:]))
    assert np.all(np.isfinite(drop_env.stepCountNext4HourTruthAll))
    assert np.all(np.isfinite(drop_env.morningFitbitWearTruthAll))


def test_calibrated_hazard_dropout_days_are_geometric():
    hazard = float(calibrate_constant_daily_hazard(pd.read_csv(DROPOUT_TABLE))[
        "dropout_hazard_day"
    ])
    n_episodes = 2000
    days84 = np.empty(n_episodes, dtype=int)
    days252 = np.empty(n_episodes, dtype=int)
    for i in range(n_episodes):
        rng = np.random.default_rng(
            np.random.SeedSequence([i, experiment.OnlineEnv._DROPOUT_RNG_SALT])
        )
        d84 = d252 = -1
        for d in range(252):
            if rng.random() < hazard:
                if d < 84:
                    d84 = d + 1
                d252 = d + 1
                break
        days84[i] = d84
        days252[i] = d252
    empirical84 = float(np.mean(days84 > 0))
    empirical252 = float(np.mean(days252 > 0))
    expected84 = 1.0 - (1.0 - hazard) ** 84
    expected252 = 1.0 - (1.0 - hazard) ** 252
    se84 = np.sqrt(expected84 * (1.0 - expected84) / n_episodes)
    se252 = np.sqrt(expected252 * (1.0 - expected252) / n_episodes)
    assert abs(empirical84 - expected84) < 3.0 * se84
    assert abs(empirical252 - expected252) < 3.0 * se252
    dropped = days84 > 0
    mean_day = float(np.mean(days84[dropped]))
    expected_mean = _truncated_geometric_mean(hazard, 84)
    assert abs(mean_day - expected_mean) / expected_mean < 0.05
    # Recorded in report.md section (C): n=2000, 8.60% by day 84, 24.35% by day 252.
    assert empirical84 == pytest.approx(0.0860, abs=1e-4)
    assert empirical252 == pytest.approx(0.2435, abs=1e-4)

    # One live OnlineEnv draw sequence uses the same RNG contract.
    cfg, _env, oenv = experiment._make_online_env(
        13, seed=7, params_dir=PARAMS, dropout_hazard_day=hazard
    )
    assert cfg.nweek >= 12
    for d_global in range(84):
        oenv._maybe_dropout(d_global)
        if oenv.dropped_out:
            break
    rng = np.random.default_rng(
        np.random.SeedSequence([7, experiment.OnlineEnv._DROPOUT_RNG_SALT])
    )
    expected_day = -1
    for d in range(84):
        if rng.random() < hazard:
            expected_day = d + 1
            break
    assert oenv.dropout_day == expected_day


def test_agent_freezes_after_dropout_week_and_omits_td():
    cfg, env, oenv = experiment._make_online_env(
        13, seed=5, params_dir=PARAMS, dropout_hazard_day=1.0
    )
    agent = _FreezeAgent()
    dataset = experiment.EpisodeDataset(cfg.nweek)
    week0, i_hist = experiment.shared_episode_exogenous(5, cfg.nweek)
    result = oenv.run_episode(agent, dataset, week0_actions=week0, I_hist=i_hist)
    assert oenv.dropout_day == 1
    assert agent.begin_weeks == [0]
    assert agent.rlsvi_weeks == []
    assert agent.acts == []
    assert np.all(result["A"] == 0)
    assert np.all(result["I"][1:] == 0)
    assert 0 not in dataset.med_full_hist
    packet = oenv.get_week_packet(1)
    assert packet is not None
    assert packet.J_w == 0
    assert np.isfinite(packet.Y_prev)
