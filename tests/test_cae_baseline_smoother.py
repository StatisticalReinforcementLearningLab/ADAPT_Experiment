import numpy as np

from vani_env import impute_baseline_cae


def _inputs(a1=0.6, mu=0.4):
    theta_cae = np.array([0.0, a1, 0.0, 0.0, 0.0])
    c_panel = np.full(11, (1.0 - a1) * mu)
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


def test_week2_backcast_uses_week2_intercept():
    a1 = 0.4
    theta, c_panel, y = _inputs(a1=a1)
    c_panel[0] = 0.3
    y[1] = 1.2
    m1 = c_panel[0] / (1.0 - a1)

    got, source = impute_baseline_cae(theta, c_panel, y)
    expected = m1 + a1 * (y[1] - a1 * m1 - c_panel[0])

    assert source == "backcast_1w"
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
