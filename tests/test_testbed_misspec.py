from __future__ import annotations

import io
import json
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pytest

import experiment
from misspec import DECAY_C, MISSPEC_C_BLURB, decay_c, decay_m, draw_confounder
from testbed import (
    MISSPEC_BLURBS,
    catalog_24,
    log_ste_activity_off,
    resolve_testbed_spec,
    results_suffix,
)
from vani_env import EnvConfig


ROOT = Path(__file__).resolve().parents[1]
PARAMS = ROOT / "env_para_vanilla"
TEST_STRENGTHS_A = {"lam_SC": 0.3, "lam_AA": 0.3}


def _traj(oenv):
    return {
        name: np.asarray(getattr(oenv, name), dtype=float)
        for name in (
            "CAE_all",
            "pu_all",
            "action_all",
            "stepCountNext4HourAll",
            "pageViewNext4HourAll",
            "morningFitbitWearAll",
            "dailySurveyCompleteAll",
        )
    }


def test_catalog_has_24_distinct_folder_names():
    rows = catalog_24(ROOT)
    assert len(rows) == 24
    suffixes = [r.suffix for r in rows]
    assert len(set(suffixes)) == 24
    roots = [r.results_root for r in rows]
    assert len(set(roots)) == 24
    layer1 = [r for r in rows if r.layer == 1]
    layer2 = [r for r in rows if r.layer == 2]
    layer3 = [r for r in rows if r.layer == 3]
    assert len(layer1) == 3 and all(r.activity == "off" and r.misspec == "none" for r in layer1)
    assert len(layer2) == 9 and all(r.activity == "on" and r.misspec == "none" for r in layer2)
    assert len(layer3) == 12 and all(r.activity == "on" and r.stayer_fraction == 0.0 for r in layer3)


def test_results_suffix_changes_with_each_axis():
    a = results_suffix("env_para_ste0.5", "on", 0.25, "A")
    b = results_suffix("env_para_ste0.5", "off", 0.25, "A")
    c = results_suffix("env_para_ste0.5", "on", 0.0, "A")
    d = results_suffix("env_para_ste0.8", "on", 0.25, "A")
    e = results_suffix("env_para_ste0.5", "on", 0.25, "none")
    assert a == "ste0.5_gapon_psi0.25_misspecA"
    assert len({a, b, c, d, e}) == 5


def test_activity_off_matches_lambda_zero_hook():
    seed, uid = 11, 13
    _a, off = experiment.run_random_send(
        uid, seed=seed, params_dir=PARAMS, activity="off", misspec="none"
    )
    _b, hook = experiment.run_random_send(
        uid, seed=seed, params_dir=PARAMS, dropout_hazard_day=0.0, misspec="none"
    )
    assert np.all(np.asarray(off._activity_status)[: off.D] == 0)
    assert off.n_gaps == 0
    for name, arr in _traj(off).items():
        np.testing.assert_array_equal(arr, _traj(hook)[name])


def test_misspec_none_matches_current_simulator():
    seed, uid = 5, 13
    _a, baseline = experiment.run_random_send(uid, seed=seed, params_dir=PARAMS)
    _b, explicit = experiment.run_random_send(
        uid, seed=seed, params_dir=PARAMS, misspec="none"
    )
    for name, arr in _traj(baseline).items():
        np.testing.assert_array_equal(arr, _traj(explicit)[name])
    np.testing.assert_array_equal(baseline._activity_status, explicit._activity_status)


def test_confounder_and_activity_identical_across_policies():
    """C_w and the activity path match across policies for the same (seed, uid).

    Except under misspecification D, where the gap-start hazard depends on
    recent suggestions; the random draws are still shared, but the path
    changes with the policy. This test uses misspec A.
    """
    seed, uid = 9, 13
    _n, never = experiment.run_never_send(
        uid, seed=seed, params_dir=PARAMS, activity="on", misspec="A",
        misspec_strengths=TEST_STRENGTHS_A,
    )
    _a, always = experiment.run_always_send(
        uid, seed=seed, params_dir=PARAMS, activity="on", misspec="A",
        misspec_strengths=TEST_STRENGTHS_A,
    )
    assert not np.array_equal(never.action_all, always.action_all)
    np.testing.assert_array_equal(never.env._confounder_C, always.env._confounder_C)
    expected = draw_confounder(seed, uid, never.nweek)
    np.testing.assert_allclose(never.env._confounder_C, expected)
    np.testing.assert_array_equal(never._activity_status, always._activity_status)


def test_decay_m_mean_and_week36_and_policy_independence():
    c = decay_c()
    assert c == pytest.approx(DECAY_C)
    weeks = np.arange(1, 13)
    mean12 = float(np.mean([decay_m(w) for w in weeks]))
    assert mean12 == pytest.approx(1.0, abs=1e-12)
    assert decay_m(1) == pytest.approx(1.35, abs=0.01)
    assert decay_m(12) == pytest.approx(0.71, abs=0.01)
    assert decay_m(36) == pytest.approx(c * 2 ** (-35 / 12), rel=1e-12)
    assert decay_m(36) == pytest.approx(0.18, abs=0.02)
    assert decay_m(5) == decay_m(5)
    assert "decay from week 1" in MISSPEC_C_BLURB
    assert MISSPEC_BLURBS["C"] == MISSPEC_C_BLURB
    assert "decay after the 12-week" not in MISSPEC_C_BLURB


def test_misspec_A_B_require_calibration_or_explicit_strengths():
    with pytest.raises(FileNotFoundError, match="misspec_calibration"):
        EnvConfig(13, params_dir=PARAMS, misspec="A")
    with pytest.raises(FileNotFoundError, match="misspec_calibration"):
        EnvConfig(13, params_dir=PARAMS, misspec="B")
    cfg_a = EnvConfig(
        13, params_dir=PARAMS, misspec="A", misspec_strengths=TEST_STRENGTHS_A,
    )
    assert cfg_a.misspec_lam_SC == 0.3
    assert cfg_a.misspec_lam_AA == 0.3
    cfg_b = EnvConfig(
        13, params_dir=PARAMS, misspec="B", misspec_strengths={"beta_D": 0.12},
    )
    assert cfg_b.misspec_beta_D == pytest.approx(0.12)


def test_tuned_folder_copies_activity_process(tmp_path, monkeypatch):
    monkeypatch.setenv("ADAPR_STE_CENTER", "0")
    from tune_ste import write_scaled_params

    dst = tmp_path / "env_para_ste_test"
    write_scaled_params(PARAMS, dst, [13], multipliers={"A_to_MY": 1.0})
    src_block = json.loads(
        (PARAMS / "generator_calibration.json").read_text(encoding="utf-8")
    )["activity_process"]
    dst_block = json.loads(
        (dst / "generator_calibration.json").read_text(encoding="utf-8")
    )["activity_process"]
    assert dst_block == src_block
    assert "stayer_variants" in dst_block


def test_misspec_d_requires_activity_on():
    with pytest.raises(ValueError, match="misspec D requires activity on"):
        EnvConfig(13, params_dir=PARAMS, activity="off", misspec="D")
    with pytest.raises(ValueError, match="misspec D requires activity on"):
        resolve_testbed_spec(PARAMS, activity="off", misspec="D")


def test_calibrate_misspec_A_defaults_to_12_weeks():
    """A OLS episodes match the 11-week fit panel; B stays at 36."""
    from calibrate_misspec import _run_A_episode, build_parser

    args = build_parser().parse_args([])
    assert args.nweek == 12
    assert args.nweek_b == 36
    X, y = _run_A_episode(PARAMS, 13, 1, args.nweek, 0.3, 0.3)
    assert y.size == 12
    assert X.shape == (12, 4)


def test_load_confirmation_ratio_from_ste_tuning(tmp_path):
    from proxy_ste import _format_proxy_line, load_confirmation_ratio

    missing, why = load_confirmation_ratio(tmp_path)
    assert missing is None and "missing" in why
    (tmp_path / "ste_tuning.json").write_text("{}", encoding="utf-8")
    none, why = load_confirmation_ratio(tmp_path)
    assert none is None and "TUNE_PHASE=ratio" in why
    (tmp_path / "ste_tuning.json").write_text(
        json.dumps({"confirmation": {"proxy_to_true_estimate": 1.25}}),
        encoding="utf-8",
    )
    ratio, why = load_confirmation_ratio(tmp_path)
    assert ratio == 1.25 and why == ""
    line = _format_proxy_line(
        {
            "proxy_ste": 0.4,
            "scaled_proxy_ste": 0.5,
            "confirmation_ratio": 1.25,
            "effective_proxy_ste": None,
        }
    )
    assert "scaled proxy STE = 0.5000" in line


def test_ste_logs_activity_off():
    spec = resolve_testbed_spec(PARAMS, activity="on", stayer_fraction=0.25, misspec="none")
    buf = io.StringIO()
    with redirect_stdout(buf):
        msg = log_ste_activity_off(spec)
    assert "activity off" in msg
    assert "activity off" in buf.getvalue()


def test_d_stepper_at_mean_burden_matches_exogenous_path():
    """When B_z = 0 the D addend is 0, so the stepper matches the pooled path."""
    import json

    from activity_process import (
        ActivityStepper,
        activity_draw,
        dropout_seed_sequence,
        simulate_activity_paths,
    )

    record = json.loads((PARAMS / "generator_calibration.json").read_text())["activity_process"]
    uid, seed, n_days, nweek = 13, 7, 84, 12
    rng_a = np.random.default_rng(dropout_seed_sequence(seed, uid))
    rng_b = np.random.default_rng(dropout_seed_sequence(seed, uid))
    coef, fit, u, gamma, _ = activity_draw(record, "pooled", uid, rng_a)
    sim = simulate_activity_paths(
        1, n_days, coef, fit, nweek, rng_a,
        u=None if u is None else np.asarray([u]),
        gamma=gamma,
    )
    coef2, fit2, u2, gamma2, _ = activity_draw(record, "pooled", uid, rng_b)
    stepper = ActivityStepper(
        n_days, nweek, coef2, fit2, rng_b,
        u=None if u2 is None else float(u2),
        gamma=gamma2,
    )
    for d in range(n_days + 2):
        stepper.ensure_day(d, 0.0)
    np.testing.assert_array_equal(stepper.status_hist, sim["status"][0])
    assert stepper.n_gaps == int(sim["n_gaps"][0])
    assert stepper.inactive_days == int(sim["inactive_days"][0])
    assert stepper.dropout_day == int(sim["dropout_day"][0])


def test_d_random_send_matches_pooled_84day_table():
    """Misspec D under random-0.5 stays within MC of the pooled 84-day table.

    Recent burden is MRT-z-scored, so under random-0.5 the Model-1 addend has
    mean near 0 and the gap table should match the exogenous pooled fit on
    the same (seed, uid) design.
    """
    import json

    from activity_process import simulate_cohort_paths, summarize_paths

    record = json.loads((PARAMS / "generator_calibration.json").read_text())["activity_process"]
    expected_full = record["validation_2000"]["pooled"]
    uids = [
        int(u)
        for u in record["validation_2000"]["uids"]
        if (PARAMS / f"params_env_{int(u)}.json").is_file()
    ]
    assert uids
    n = min(80, 2 * len(uids))
    base_seed = int(expected_full["base_seed_84"])
    n_days = 84
    expected = summarize_paths(
        simulate_cohort_paths(
            record, "pooled", n, n_days, 12, uids, base_seed,
        )
    )
    parts = []
    for i in range(n):
        uid = uids[i % len(uids)]
        seed = base_seed + i // len(uids)
        _res, oenv = experiment.run_random_send(
            uid,
            seed=seed,
            params_dir=PARAMS,
            activity="on",
            misspec="D",
            nweek=12,
        )
        stepper = oenv._activity_stepper
        if stepper is not None:
            parts.append(
                {
                    "n_gaps": np.asarray([stepper.n_gaps]),
                    "inactive_days": np.asarray([stepper.inactive_days]),
                    "dropout_day": np.asarray([stepper.dropout_day]),
                    "ended": np.asarray(stepper.ended, dtype=int),
                    "censored": np.asarray(stepper.censored, dtype=int),
                    "terminal14": np.asarray([stepper.terminal14]),
                }
            )
        else:
            parts.append(
                {
                    "n_gaps": np.asarray([oenv.n_gaps]),
                    "inactive_days": np.asarray([oenv.inactive_days]),
                    "dropout_day": np.asarray([oenv.dropout_day]),
                    "ended": np.asarray([], dtype=int),
                    "censored": np.asarray([], dtype=int),
                    "terminal14": np.asarray([False]),
                }
            )

    def cat(key):
        return np.concatenate([np.atleast_1d(p[key]) for p in parts])

    got = summarize_paths(
        {
            "n_gaps": cat("n_gaps"),
            "inactive_days": cat("inactive_days"),
            "dropout_day": cat("dropout_day"),
            "ended": cat("ended"),
            "censored": cat("censored"),
            "terminal14": cat("terminal14"),
            "n_days": n_days,
            "n_paths": n,
        }
    )
    n_got = int(got["n_paths"])
    for key in (
        "mean_gaps", "p0", "p1", "p2", "p3plus",
        "inactive_fraction", "terminal14", "permanent",
    ):
        diff = abs(float(got[key]) - float(expected[key]))
        level = float(expected[key])
        if key == "mean_gaps":
            se = (max(level, 0.0) / n_got) ** 0.5
            tol = max(3.0 * se, 0.08)
        else:
            se = (max(level, 0.0) * (1.0 - min(level, 1.0)) / n_got) ** 0.5
            tol = max(3.0 * se, 0.02)
        assert diff <= tol, f"{key}: {got[key]} vs {expected[key]} (tol {tol})"
