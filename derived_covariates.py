"""Derived covariates shared by the data pipeline and the simulator.

Every history summary used by the testbed models is defined here once:
``02_combine.py`` calls :func:`add_derived_covariates` on the MRT
panel, and the simulator computes the same quantities from its generated
series with the same primitives (:func:`prior_ewma`, :func:`prior_ewma_masked`).
A fitted coefficient therefore always multiplies a covariate that the
simulator regenerates in the same way.

Conventions
-----------
* Counts (steps, page views) are summarized on the ``log(x + 1)`` scale.
* "Recent" summaries are exponentially weighted averages of the *prior*
  window (the current day or slot is excluded), with the project-wide weights
  of :func:`ewm_utils.ewma_gamma`: the most recent value has weight 1, the one
  before ``lambda``, and so on, ``lambda = 1 - 1/n`` for an ``n``-point window
  (6/7 for 7 days, 13/14 for 14 slots). Windows are calendar windows.
* Missing base values (testbed only; the simulator generates complete series)
  are imputed before summarizing:
    - engagement (page views, Fitbit wear, suggestions sent, interaction with
      a delivered suggestion): 0, i.e. no engagement, on the raw scale;
    - everything else (step counts, active-day status): the participant's mean.
  A slot without a delivered suggestion is not missing: it is skipped in the
  interaction summary (there was nothing to interact with).

Columns produced by :func:`add_derived_covariates` (slot-level panel, two rows
per day, sorted by participant, date and decision time):

=============================== ==================================================
``FourSC_lag1_log``             next-4h log steps of the previous decision slot
``prior2hour_step_lag1_log``    prior-2h log steps of the previous decision slot
``HourlyPageviewCount_lag1_log`` log page views of the previous decision slot
``YesterdayStepCount``          mean of yesterday's two next-4h log step counts
``EMA_Prior2HourStepCount``     same slot, prior 7 days, prior-2h log steps
``Past7DaysHourlyPageviewEMA``  same slot, prior 7 days, log page views
``Past7DaysPageviewEMA``        prior 7 days, daily mean of the two slots' log page views
``recent_burden``               prior 7 days, suggestions sent per day (0--2)
``past7days_morning_wearing``   prior 7 days, morning Fitbit wear
``active_status_fraction_7days`` prior 7 days, self-reported active day
``Interacted_7d_walk``          prior 14 slots, interaction among delivered suggestions
=============================== ==================================================

The two step and page-view "yesterday"/"7-day" summaries are built from the
two decision windows only (not from full-day totals), because the simulator
generates steps and page views only in those windows.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ewm_utils import ewma_gamma

WINDOW_DAYS = 7
WINDOW_SLOTS = 14
SLOTS_PER_DAY = 2


# --------------------------------------------------------------------------
# Primitives (used by the pipeline and by the simulator)
# --------------------------------------------------------------------------

def prior_ewma(values, window: int) -> np.ndarray:
    """``out[t]`` = EWMA of ``values[t-window:t]`` (current value excluded).

    Calendar positions are kept; non-finite values are skipped in the
    average. ``out[0]`` and windows with no finite value are NaN.
    """
    x = np.asarray(values, dtype=float).ravel()
    out = np.full(x.size, np.nan)
    for t in range(1, x.size):
        out[t] = ewma_gamma(x[max(0, t - window):t], empty=np.nan)
    return out


def prior_ewma_masked(values, include, window: int) -> np.ndarray:
    """:func:`prior_ewma` using only positions where ``include`` is true."""
    x = np.asarray(values, dtype=float).ravel()
    keep = np.asarray(include, dtype=bool).ravel()
    return prior_ewma(np.where(keep, x, np.nan), window)


def last_ewma(history, window: int) -> float:
    """EWMA of the last ``window`` entries of ``history`` (simulator form of :func:`prior_ewma`)."""
    h = np.asarray(history, dtype=float).ravel()
    return float(ewma_gamma(h[-window:], empty=np.nan)) if h.size else float("nan")


def seeded_last_ewma(history, seed: float, window: int) -> float:
    """Simulator form of :func:`prior_ewma` at the start of a simulation.

    The window is left-padded with ``seed`` (the participant's value of the
    summary at study entry) until ``window`` simulated values exist, so the
    summary starts at the MRT value and is displaced by simulated values.
    """
    h = [float(v) for v in list(history)[-int(window):]]
    seed = float(seed) if np.isfinite(seed) else 0.0
    vals = np.asarray([seed] * (int(window) - len(h)) + h, dtype=float)
    return float(ewma_gamma(vals, empty=seed))


def seeded_last_ewma_masked(values, include, seed: float, window: int) -> float:
    """:func:`seeded_last_ewma` over positions where ``include`` is true (NaN if none)."""
    v = [float(x) for x in list(values)[-int(window):]]
    m = [bool(x) for x in list(include)[-int(window):]]
    pad = int(window) - len(v)
    seed = float(seed) if np.isfinite(seed) else np.nan
    vals = np.asarray([seed] * pad + v, dtype=float)
    keep = np.asarray([np.isfinite(seed)] * pad + m, dtype=bool)
    return float(ewma_gamma(np.where(keep, vals, np.nan), empty=np.nan))


def log1p_count(x) -> np.ndarray:
    v = np.asarray(x, dtype=float)
    return np.log1p(np.where(np.isfinite(v), np.maximum(v, 0.0), np.nan))


# --------------------------------------------------------------------------
# Panel
# --------------------------------------------------------------------------

def _impute_mean(x: np.ndarray) -> np.ndarray:
    m = np.nanmean(x) if np.isfinite(x).any() else 0.0
    return np.where(np.isfinite(x), x, m)


def _impute_zero(x: np.ndarray) -> np.ndarray:
    return np.where(np.isfinite(x), x, 0.0)


def _participant_covariates(g: pd.DataFrame) -> pd.DataFrame:
    """Derived covariates for one participant's complete slot grid."""
    g = g.sort_values(["Date", "DecisionTime"], kind="mergesort")
    n = len(g)
    slot = g["DecisionTime"].to_numpy(dtype=int)
    dates = pd.to_datetime(g["Date"]).dt.normalize().to_numpy()
    if (n % SLOTS_PER_DAY or not np.array_equal(slot, np.tile(np.arange(SLOTS_PER_DAY), n // SLOTS_PER_DAY))
            or not np.all(dates[0::2] == dates[1::2])
            or not np.all(np.diff(dates[0::2]).astype("timedelta64[D]").astype(int) == 1)):
        raise ValueError(
            f"participant {g['ParticipantIdentifier'].iloc[0]}: expected a complete calendar "
            "grid with two decision slots per day"
        )

    # Base series with the imputation rules of the module docstring.
    steps4 = _impute_mean(log1p_count(g["4hour_step"]))
    steps2 = _impute_mean(log1p_count(g["prior2hour_step"]))
    pv = _impute_zero(log1p_count(g["HourlyPageviewCount"]))
    sent = _impute_zero(g["WalkingSuggestion"].to_numpy(dtype=float))
    delivered = sent == 1.0
    inter = _impute_zero(g["Interacted_walk"].to_numpy(dtype=float))

    out = pd.DataFrame(index=g.index)
    out["FourSC_lag1_log"] = np.r_[np.nan, steps4[:-1]]
    out["prior2hour_step_lag1_log"] = np.r_[np.nan, steps2[:-1]]
    out["HourlyPageviewCount_lag1_log"] = np.r_[np.nan, pv[:-1]]

    # Same-slot 7-day summaries.
    for name, series in (("EMA_Prior2HourStepCount", steps2),
                         ("Past7DaysHourlyPageviewEMA", pv)):
        col = np.full(n, np.nan)
        for s in range(SLOTS_PER_DAY):
            idx = np.flatnonzero(slot == s)
            col[idx] = prior_ewma(series[idx], WINDOW_DAYS)
        out[name] = col

    out["Interacted_7d_walk"] = prior_ewma_masked(inter, delivered, WINDOW_SLOTS)

    # Day-level series (one value per date; both slots carry it).
    day_steps = steps4.reshape(-1, SLOTS_PER_DAY).mean(axis=1)
    day_pv = pv.reshape(-1, SLOTS_PER_DAY).mean(axis=1)
    day_sent = sent.reshape(-1, SLOTS_PER_DAY).sum(axis=1)
    first = np.arange(0, n, SLOTS_PER_DAY)
    day_wear = _impute_zero(g["morning_wearing"].to_numpy(dtype=float)[first])
    day_active = _impute_mean(g["active_status"].to_numpy(dtype=float)[first])

    daily = {
        "YesterdayStepCount": np.r_[np.nan, day_steps[:-1]],
        "Past7DaysPageviewEMA": prior_ewma(day_pv, WINDOW_DAYS),
        "recent_burden": prior_ewma(day_sent, WINDOW_DAYS),
        "past7days_morning_wearing": prior_ewma(day_wear, WINDOW_DAYS),
        "active_status_fraction_7days": prior_ewma(day_active, WINDOW_DAYS),
    }
    for name, v in daily.items():
        out[name] = np.repeat(v, SLOTS_PER_DAY)
    return out


DERIVED_COLUMNS = (
    "FourSC_lag1_log", "prior2hour_step_lag1_log", "HourlyPageviewCount_lag1_log",
    "YesterdayStepCount",
    "EMA_Prior2HourStepCount", "Past7DaysHourlyPageviewEMA",
    "Past7DaysPageviewEMA", "recent_burden", "past7days_morning_wearing",
    "active_status_fraction_7days", "Interacted_7d_walk",
)

REQUIRED_COLUMNS = (
    "ParticipantIdentifier", "Date", "DecisionTime", "4hour_step", "prior2hour_step",
    "HourlyPageviewCount", "WalkingSuggestion", "Interacted_walk",
    "morning_wearing", "active_status",
)


def add_derived_covariates(df: pd.DataFrame) -> pd.DataFrame:
    """Add :data:`DERIVED_COLUMNS` to a slot-level panel (raw-scale base columns).

    Call on the full extraction grid, before burn-in days are dropped, so the
    first analysis day has a full 7-day history.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"derived covariates need columns {missing}")
    out = df.copy()
    out = out.drop(columns=[c for c in DERIVED_COLUMNS if c in out.columns])
    out["Date"] = pd.to_datetime(out["Date"]).dt.normalize()
    base = list(REQUIRED_COLUMNS)
    parts = []
    for pid, g in out[base].groupby("ParticipantIdentifier", sort=False):
        # Complete calendar grid (days with no extracted row get NaN base
        # values, which the imputation rules then fill), so windows follow
        # calendar time.
        days = pd.date_range(g["Date"].min(), g["Date"].max(), freq="D")
        grid = pd.MultiIndex.from_product([days, range(SLOTS_PER_DAY)], names=["Date", "DecisionTime"])
        full = (g.drop_duplicates(["Date", "DecisionTime"]).set_index(["Date", "DecisionTime"])
                .reindex(grid).reset_index())
        full["ParticipantIdentifier"] = pid
        cov = _participant_covariates(full)
        cov[["ParticipantIdentifier", "Date", "DecisionTime"]] = full[["ParticipantIdentifier", "Date", "DecisionTime"]]
        parts.append(cov)
    derived = pd.concat(parts, ignore_index=True)
    out["DecisionTime"] = out["DecisionTime"].astype(int)
    derived["DecisionTime"] = derived["DecisionTime"].astype(int)
    return out.merge(derived, on=["ParticipantIdentifier", "Date", "DecisionTime"], how="left")
