import numpy as np

import os
import warnings
from pathlib import Path

import pandas as pd
import pytest

from vani_env import (
    c1_from_week1_ewma,
    cae_mediator_ewma_rows,
    impute_baseline_cae,
    load_week1_cae_inputs,
    mask_dropout_fit_outcomes,
    week1_cae_drift,
    week1_cae_input_record,
    week1_cae_input_table,
    week1_mediator_slots,
)


def _inputs(a1=0.6, mu=0.4):
    theta_cae = np.array([0.0, a1, 0.0, 0.0, 0.0])
    c_panel = np.full(12, (1.0 - a1) * mu)
    y = np.full(12, np.nan)
    return theta_cae, c_panel, y


def test_no_observations_returns_week1_model_mean():
    theta, c_panel, y = _inputs(a1=0.7, mu=-0.25)
    got, source = impute_baseline_cae(theta, c_panel, y)
    assert source == "model_mean"
    np.testing.assert_allclose(
        got, c_panel[0] / (1.0 - 0.7), rtol=0.0, atol=1e-12
    )


def test_constant_covariates_reproduce_old_formula():
    a1, mu, k, observed = 0.55, 0.35, 7, -0.8
    theta, c_panel, y = _inputs(a1=a1, mu=mu)
    y[k - 1] = observed
    got, source = impute_baseline_cae(theta, c_panel, y)
    expected = mu + a1 ** (k - 1) * (observed - mu)
    assert source == "backcast_6w"
    np.testing.assert_allclose(got, expected, rtol=0.0, atol=1e-12)


def test_week2_backcast_uses_week1_mean_and_week2_intercept():
    a1 = 0.4
    theta, c_panel, y = _inputs(a1=a1)
    c_panel[0] = 0.2
    c_panel[1] = 0.3
    y[1] = 1.2
    m1 = c_panel[0] / (1.0 - a1)

    got, source = impute_baseline_cae(theta, c_panel, y)
    expected = m1 + a1 * (y[1] - a1 * m1 - c_panel[1])

    assert source == "backcast_1w"
    np.testing.assert_allclose(got, expected, rtol=0.0, atol=1e-12)


def test_missing_c1_falls_back_to_c2_and_records_source():
    a1 = 0.5
    theta, c_panel, y = _inputs(a1=a1, mu=0.2)
    c_panel[0] = np.nan
    c_panel[1] = 0.4
    y[1] = 1.0
    m1 = c_panel[1] / (1.0 - a1)

    got, source = impute_baseline_cae(theta, c_panel, y)
    expected = m1 + a1 * (y[1] - a1 * m1 - c_panel[1])

    assert source == "backcast_1w_c2"
    np.testing.assert_allclose(got, expected, rtol=0.0, atol=1e-12)


def test_observed_week1_is_returned_and_later_values_are_ignored():
    theta, c_panel, y = _inputs()
    y[0] = 1.23456789
    y[3] = -3.0
    y[8] = 4.0

    got, source = impute_baseline_cae(theta, c_panel, y)

    assert source == "observed"
    assert got == y[0]


def test_only_first_future_observation_is_used():
    theta, c_panel, y = _inputs(a1=0.65, mu=0.2)
    y[3] = -0.6
    first_only, source = impute_baseline_cae(theta, c_panel, y)
    y[8] = 100.0
    with_later, source_with_later = impute_baseline_cae(theta, c_panel, y)

    assert source == "backcast_3w"
    assert source_with_later == source
    np.testing.assert_allclose(
        with_later, first_only, rtol=0.0, atol=0.0
    )


def test_week1_drift_uses_mediator_ewma():
    a1, intercept, beta_sc, beta_aa = 0.5, 0.1, 0.3, 0.4
    theta = np.array([intercept, a1, beta_sc, beta_aa])
    foursc = np.arange(14, dtype=float)
    antic = np.linspace(-1.0, 1.0, 14)
    foursc_e, antic_e = cae_mediator_ewma_rows(foursc, antic)
    got = week1_cae_drift(theta, foursc, antic)
    expected = (
        intercept
        + beta_sc * float(foursc_e[0])
        + beta_aa * float(antic_e[0])
    )
    np.testing.assert_allclose(got, expected, rtol=0.0, atol=1e-12)


def test_week1_drift_fills_an_all_missing_mediator_like_later_weeks():
    theta = np.array([0.1, 0.5, 0.3, 0.4])
    foursc = np.arange(14, dtype=float)
    antic_missing = np.full(14, np.nan)
    foursc_e, antic_e = cae_mediator_ewma_rows(foursc, antic_missing)
    np.testing.assert_allclose(antic_e[0], 0.0, rtol=0.0, atol=0.0)
    got = week1_cae_drift(theta, foursc, antic_missing)
    expected = (
        theta[0]
        + theta[2] * float(foursc_e[0])
        + theta[3] * float(antic_e[0])
    )
    assert np.isfinite(got)
    np.testing.assert_allclose(got, expected, rtol=0.0, atol=1e-12)
    both_missing = week1_cae_input_record(np.full(14, np.nan), np.full(14, np.nan))
    assert both_missing["irregular"] == 0
    assert both_missing["reason"] == ""
    assert np.isfinite(both_missing["foursc_ewma"])
    assert np.isfinite(both_missing["antic_ewma"])


def test_no_week1_rows_is_the_c2_fallback():
    rec = week1_cae_input_record(np.array([]), np.array([]))
    assert rec["reason"] == "no_rows"
    assert rec["irregular"] == 0
    assert np.isnan(rec["foursc_ewma"]) and np.isnan(rec["antic_ewma"])
    theta = np.array([0.1, 0.5, 0.3, 0.4])
    assert np.isnan(c1_from_week1_ewma(theta, rec["foursc_ewma"], rec["antic_ewma"]))


def test_missing_week1_inputs_file_is_an_error(tmp_path):
    missing = tmp_path / "week1_cae_inputs.csv"
    with pytest.raises(FileNotFoundError, match="week1_cae_inputs.csv"):
        load_week1_cae_inputs(missing)


def test_irregular_week_falls_back_to_c2_with_label():
    foursc = np.arange(3, dtype=float)
    antic = np.linspace(-1.0, 1.0, 3)
    rec = week1_cae_input_record(foursc, antic)
    assert rec["irregular"] == 1
    assert np.isnan(rec["foursc_ewma"])
    theta = np.array([0.1, 0.5, 0.3, 0.4])
    assert np.isnan(c1_from_week1_ewma(theta, rec["foursc_ewma"], rec["antic_ewma"]))
    c_panel = np.full(12, 0.2)
    c_panel[0] = np.nan
    y = np.full(12, np.nan)
    y[1] = 1.0
    _value, source = impute_baseline_cae(theta, c_panel, y)
    assert source == "backcast_1w_c2"
    y[:] = np.nan
    _value, source = impute_baseline_cae(theta, c_panel, y)
    assert source == "model_mean_c2"


def test_regular_week_c1_matches_ewma_formula():
    theta = np.array([0.1, 0.5, 0.3, 0.4])
    foursc = np.arange(14, dtype=float)
    antic = np.linspace(-1.0, 1.0, 14)
    rec = week1_cae_input_record(foursc, antic)
    assert rec["irregular"] == 0
    got = c1_from_week1_ewma(theta, rec["foursc_ewma"], rec["antic_ewma"])
    assert got == week1_cae_drift(theta, foursc, antic)
    foursc_e, antic_e = cae_mediator_ewma_rows(foursc, antic)
    expected = (
        theta[0]
        + theta[2] * float(foursc_e[0])
        + theta[3] * float(antic_e[0])
    )
    np.testing.assert_allclose(got, expected, rtol=0.0, atol=1e-12)


def test_zero_week_slot_is_dropped_and_a_nonzero_slot_raises():
    from vani_env import (
        _DROPPED_WEEK_WARNED,
        _split_ml_fw,
        _split_ml_pv_hurdle,
        _trim_theta_j,
    )

    plain = np.arange(23, dtype=float)
    plain[-1] = 0.2

    def _insert(base):
        return np.concatenate([base[:5], [0.0], base[5:]])

    padded = np.concatenate([
        _insert(plain[:8]),
        plain[8:11],
        _insert(plain[11:19]),
        plain[19:22],
        plain[22:],
    ])
    assert padded.size == 25
    _DROPPED_WEEK_WARNED.discard("theta_ml_PV")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        dropped = _split_ml_pv_hurdle(padded)
        written = _split_ml_pv_hurdle(plain)
    for left, right in zip(dropped, written):
        if left is None:
            assert right is None
        else:
            np.testing.assert_allclose(left, right)
    assert any("zero study-week coefficient" in str(w.message) for w in caught)
    bad = padded.copy()
    bad[5] = 0.25
    with pytest.raises(ValueError, match="study-week coefficient"):
        _split_ml_pv_hurdle(bad)

    fw = np.arange(12, dtype=float)
    fw_week = np.concatenate([fw[:4], [0.0], fw[4:]])
    assert fw_week.size == 13
    _DROPPED_WEEK_WARNED.discard("theta_ml_FW")
    base, query = _split_ml_fw(fw_week)
    np.testing.assert_allclose(base, fw[:9])
    np.testing.assert_allclose(query, fw[9:])
    fw_week[4] = 1.0
    with pytest.raises(ValueError, match="study-week coefficient"):
        _split_ml_fw(fw_week)

    _DROPPED_WEEK_WARNED.discard("theta_ml_J")
    np.testing.assert_allclose(_trim_theta_j(np.array([1.0, 2.0, 0.0])), [1.0, 2.0])
    with pytest.raises(ValueError, match="study-week coefficient"):
        _trim_theta_j(np.array([1.0, 2.0, 0.4]))


def test_legacy_five_coefficient_cae_drops_week_once():
    from vani_env import _DROPPED_WEEK_WARNED, trim_theta_cae

    _DROPPED_WEEK_WARNED.discard("theta_CAE")
    legacy = np.array([0.1, 0.5, 9.0, 0.3, 0.4])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        trimmed = trim_theta_cae(legacy)
        again = trim_theta_cae(legacy)
    np.testing.assert_allclose(trimmed, [0.1, 0.5, 0.3, 0.4])
    np.testing.assert_allclose(again, trimmed)
    messages = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
    assert messages.count(
        "theta_CAE has 5 coefficients including week; dropping the week coefficient"
    ) == 1


# The 42-user table from the October baseline check. The current extract also
# has 112, 251, 257, 379, and 388; those are checked the same way.
_LOCAL_42 = (
    13, 18, 22, 31, 33, 72, 75, 82, 86, 106, 117, 118, 128, 129, 138, 143,
    170, 184, 195, 204, 219, 239, 248, 252, 259, 269, 277, 281, 291, 299, 307,
    327, 333, 339, 345, 346, 350, 356, 361, 369, 381, 393,
)


def test_local_42_users_match_slot_ewma_c1():
    """c_1 from the week-1 EWMA table equals week1_cae_drift on the local extract."""
    combined = os.environ.get("ADAPR_COMBINED_DIR", "").strip()
    merged_path = Path(combined) / "df_merged.csv" if combined else Path()
    std_path = Path(__file__).resolve().parents[1] / "env_para_vanilla" / "std_params.json"
    if not merged_path.is_file() or not std_path.is_file():
        pytest.skip("local df_merged.csv or std_params.json is not available")
    import json
    std_params = json.loads(std_path.read_text(encoding="utf-8"))
    week1 = pd.read_csv(merged_path)
    week1 = week1.loc[pd.to_numeric(week1["week"], errors="coerce").eq(1)].copy()
    week1 = mask_dropout_fit_outcomes(week1)
    table = week1_cae_input_table(week1, std_params)
    present = set(int(u) for u in table["ParticipantIdentifier"])
    missing = sorted(set(_LOCAL_42) - present)
    assert not missing, missing
    theta = np.array([0.1, 0.4, 0.3, -0.1])
    for uid in sorted(present):
        dat = week1.loc[pd.to_numeric(week1["ParticipantIdentifier"], errors="coerce").eq(uid)]
        foursc, antic = week1_mediator_slots(dat, std_params)
        direct = week1_cae_drift(theta, foursc, antic)
        row = table.loc[table["ParticipantIdentifier"].eq(uid)].iloc[0]
        via_file = c1_from_week1_ewma(theta, row["foursc_ewma"], row["antic_ewma"])
        if np.isnan(direct):
            assert np.isnan(via_file)
        else:
            np.testing.assert_allclose(via_file, direct, rtol=0.0, atol=1e-12)
    assert len(_LOCAL_42) == 42
