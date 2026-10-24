"""Fit the two-state activity process on the modelled participants.

Reads the daily activity table, fits the active-to-inactive logistic hazard and
the promotion-time cure model for gap length, and writes the report, figures,
and ``activity_process`` block of ``generator_calibration.json``. Does not
refit scripts 01–03 or 06–07.

Run from the repository root, with ``ADAPR_COMBINED_DIR`` set to the folder
that holds the tables written by ``01_extract_data.py``::

    python 04_fit_activity_process.py
    python 04_fit_activity_process.py --out docs/activity_process_20261021

``ADAPR_ACTIVITY_DAILY`` overrides the daily csv. ``participant_dropout.csv``
is read from the same directory. Both must exist. Figures and ``report.md``
go to ``--out`` (default ``docs/activity_process_<YYYYMMDD>/``).
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_adapr")
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from activity_process import (
    dropout_seed_sequence,
    expit,
    hazard_uncured,
    month_c_from_week,
    month_c_of,
    simulate_cohort_paths,
    summarize_paths,
    survival_uncured,
    weibull_F,
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import approx_fprime, minimize

from vani_env import resolve_params_dir


DAILY_PATH: Path | None = None
META_PATH: Path | None = None
OUT_DIR: Path | None = None
CALIBRATION_PATH = resolve_params_dir() / "generator_calibration.json"

N_DAYS = 84
N_WEEKS_FIT = 12
N_WEEKS_LONG = 36
MODEL1_NAMES = ("intercept", "month", "act7", "prior_gap")
SOURCE_FILE = ""
SOURCE_DATE = ""
HAZARD_CURRENT = 0.0010501443948542925
DAILY_NAME = "participant_dropout_daily.csv"
META_NAME = "participant_dropout.csv"


def resolve_activity_tables() -> tuple[Path, Path]:
    """Daily and participant tables from the combined extract.

    ``ADAPR_ACTIVITY_DAILY`` overrides the daily csv. The participant table
    is the sibling ``participant_dropout.csv``. Otherwise both files are
    read from ``ADAPR_COMBINED_DIR``. Either missing file is an error.
    """
    override = os.environ.get("ADAPR_ACTIVITY_DAILY", "").strip()
    combined = os.environ.get("ADAPR_COMBINED_DIR", "").strip()
    if override:
        daily = Path(override).expanduser().resolve()
        if not daily.is_file():
            raise SystemExit(f"ADAPR_ACTIVITY_DAILY does not exist: {daily}")
        meta = daily.parent / META_NAME
    else:
        if not combined:
            raise SystemExit(
                "Set ADAPR_COMBINED_DIR to the folder with "
                f"{DAILY_NAME} and {META_NAME}, or set ADAPR_ACTIVITY_DAILY."
            )
        root = Path(combined).expanduser().resolve()
        daily = root / DAILY_NAME
        meta = root / META_NAME
    missing = [path for path in (daily, meta) if not path.is_file()]
    if missing:
        raise SystemExit(
            "Missing activity tables: "
            + ", ".join(str(path) for path in missing)
            + f". Script 1 writes {DAILY_NAME} and {META_NAME} in "
            "ADAPR_COMBINED_DIR. ADAPR_ACTIVITY_DAILY overrides the daily "
            f"file; {META_NAME} is read from the same directory."
        )
    return daily, meta


def resolve_output_dir(out: str | None) -> Path:
    """Figures and report directory. Created if it does not exist."""
    if out:
        path = Path(out).expanduser().resolve()
    else:
        stamp = datetime.date.today().strftime("%Y%m%d")
        path = ROOT / "docs" / f"activity_process_{stamp}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def configure_paths(out: str | None) -> None:
    """Set the module paths used by the fit, the report, and the calibration write."""
    global DAILY_PATH, META_PATH, OUT_DIR, SOURCE_FILE, SOURCE_DATE
    daily, meta = resolve_activity_tables()
    DAILY_PATH = daily
    META_PATH = meta
    SOURCE_FILE = str(daily)
    SOURCE_DATE = datetime.date.today().isoformat()
    OUT_DIR = resolve_output_dir(out)


def protocol_week(study_day: int, start_dow: int) -> int:
    """Mon–Sun study week, matching ``02_combine.py``.

    ``start_dow`` is the weekday of study day 1 (Monday 0, Sunday 6). A Sunday
    start is week 0; week 1 begins the next Monday.
    """
    wk = (int(study_day) - 1 + int(start_dow)) // 7
    if int(start_dow) == 0:
        wk += 1
    return int(wk)


def window_bounds(start_dow: int) -> tuple[int, int]:
    """Inclusive study days of protocol weeks 1–12."""
    if int(start_dow) == 0:
        return 1, 84
    if int(start_dow) == 6:
        return 2, 85
    raise SystemExit(
        f"Study day 1 falls on weekday {start_dow}; only Monday and Sunday starts are in this cohort."
    )


def log1mexp(a: np.ndarray) -> np.ndarray:
    """log(1 - exp(-a)) for a > 0."""
    a = np.asarray(a, dtype=float)
    out = np.empty(a.shape, dtype=float)
    large = a > np.log(2.0)
    out[large] = np.log1p(-np.exp(-a[large]))
    out[~large] = np.log(-np.expm1(-a[~large]))
    return out


def gap_survival(d: np.ndarray, theta: np.ndarray, mu: float, alpha: float) -> np.ndarray:
    """S(d) = exp(-theta F(d)). ``d`` broadcasts against ``theta``."""
    return np.exp(-theta * weibull_F(d, mu, alpha))


def load_panel() -> tuple[pd.DataFrame, pd.DataFrame]:
    meta = pd.read_csv(META_PATH)
    meta["ParticipantIdentifier"] = meta["ParticipantIdentifier"].astype(str)
    if "in_modelled" not in meta.columns:
        raise SystemExit(
            "participant_dropout.csv has no in_modelled column. "
            "Script 01 writes it from modelled_cohort.csv."
        )
    ids = set(meta.loc[meta["in_modelled"].astype(bool), "ParticipantIdentifier"])
    daily = pd.read_csv(DAILY_PATH)
    daily["ParticipantIdentifier"] = daily["ParticipantIdentifier"].astype(str)
    panel = daily[
        daily["ParticipantIdentifier"].isin(ids) & daily["study_day"].between(1, N_DAYS + 1)
    ].copy()
    panel["study_day"] = panel["study_day"].astype(int)
    panel["Date"] = pd.to_datetime(panel["Date"])
    panel["observed"] = pd.to_numeric(panel["observed"], errors="coerce").fillna(0).astype(int)
    observed = panel["observed"].eq(1)
    if "survey_any" not in panel.columns:
        raise SystemExit("participant_dropout_daily.csv has no survey_any. Rebuild it.")
    panel["active"] = np.where(
        observed,
        (
            (panel["pageview_any"] == 1)
            | (panel["wear_any"] == 1)
            | (panel["survey_any"] == 1)
        ).astype(int),
        0,
    )
    comparable = observed.to_numpy()
    if not np.array_equal(
        panel.loc[comparable, "active"].to_numpy(),
        panel.loc[comparable, "active_any"].fillna(0).astype(int).to_numpy(),
    ):
        raise SystemExit(
            "active_d = pageview OR wear OR survey does not match active_any. Stop."
        )
    day1 = panel.loc[panel["study_day"].eq(1), ["ParticipantIdentifier", "Date"]].drop_duplicates(
        "ParticipantIdentifier"
    )
    day1["start_dow"] = day1["Date"].dt.dayofweek.astype(int)
    bounds = day1["start_dow"].map(lambda dow: window_bounds(int(dow)))
    day1["window_start"] = [b[0] for b in bounds]
    day1["window_end"] = [b[1] for b in bounds]
    panel = panel.merge(
        day1[["ParticipantIdentifier", "start_dow", "window_start", "window_end"]],
        on="ParticipantIdentifier",
        how="left",
        validate="m:1",
    )
    panel["in_window"] = panel["study_day"].between(panel["window_start"], panel["window_end"])
    missing_end = panel.loc[panel["in_window"]].groupby("ParticipantIdentifier")["study_day"].max()
    expected_end = panel.groupby("ParticipantIdentifier")["window_end"].first()
    short = expected_end[missing_end.reindex(expected_end.index).lt(expected_end)]
    if len(short):
        raise SystemExit(
            "Activity table does not cover the week-12 Sunday for "
            + ", ".join(f"{uid} (need day {int(expected_end[uid])})" for uid in short.index[:8])
        )
    return panel.sort_values(["ParticipantIdentifier", "study_day"]), meta


def build_gaps(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for uid, group in panel.groupby("ParticipantIdentifier", sort=False):
        obs = dict(zip(group["study_day"], group["observed"].astype(int)))
        act = dict(zip(group["study_day"], group["active"].astype(int)))
        start_day = int(group["window_start"].iloc[0])
        end_day = int(group["window_end"].iloc[0])
        dow = int(group["start_dow"].iloc[0])
        day = start_day
        while day <= end_day:
            if obs.get(day, 0) != 1:
                day += 1
                continue
            if act.get(day, 0) == 1:
                day += 1
                continue
            start = day
            while day <= end_day and obs.get(day, 0) == 1 and act.get(day, 0) == 0:
                day += 1
            length = day - start
            returned = day <= end_day and obs.get(day, 0) == 1 and act.get(day, 0) == 1
            wk = protocol_week(start, dow)
            rows.append(
                {
                    "uid": uid,
                    "start": start,
                    "length": length,
                    "delta": int(returned),
                    "return_day": day if returned else np.nan,
                    "week_number": wk,
                    "month": float(month_c_from_week(wk)),
                    "window_start": start_day,
                }
            )
    gaps = pd.DataFrame(rows)
    prior = []
    for uid, group in gaps.groupby("uid", sort=False):
        returns = []
        for row in group.itertuples(index=False):
            prior.append(int(any(r <= row.start for r in returns)))
            if row.delta == 1:
                returns.append(int(row.return_day))
    gaps["prior_gap"] = prior
    gaps["early"] = (gaps["week_number"] <= 6).astype(int)
    return gaps


def assert_cohort(panel: pd.DataFrame, gaps: pd.DataFrame) -> dict:
    n_users = panel["ParticipantIdentifier"].nunique()
    open_lengths = tuple(sorted(int(x) for x in gaps.loc[gaps["delta"].eq(0), "length"]))
    problems = []
    window = panel.loc[panel["in_window"]]
    if int((window["observed"] != 1).sum()) != 0:
        problems.append("a week 1–12 day is unobserved")
    sizes = window.groupby("ParticipantIdentifier").size()
    if not sizes.eq(N_DAYS).all():
        problems.append("protocol weeks 1–12 are not 84 days for every user")
    day1 = panel.loc[panel["study_day"].eq(1)]
    if int(day1["active"].sum()) != n_users or int(day1["observed"].sum()) != n_users:
        problems.append("study day 1 is not an observed active day for every modelled user")
    n_monday = int(panel.groupby("ParticipantIdentifier")["start_dow"].first().eq(0).sum())
    n_sunday = int(panel.groupby("ParticipantIdentifier")["start_dow"].first().eq(6).sum())
    if n_monday + n_sunday != n_users:
        problems.append(f"start weekdays are not only Monday/Sunday ({n_monday} Monday, {n_sunday} Sunday)")
    if problems:
        raise SystemExit("Cohort checks failed:\n- " + "\n- ".join(problems))
    n_with = int(gaps["uid"].nunique())
    return {
        "n_users": n_users,
        "n_with_gap": n_with,
        "n_without_gap": n_users - n_with,
        "n_gaps": int(len(gaps)),
        "n_ended": int(gaps["delta"].sum()) if len(gaps) else 0,
        "n_open": int((gaps["delta"] == 0).sum()) if len(gaps) else 0,
        "open_lengths": open_lengths,
        "n_monday": n_monday,
        "n_sunday": n_sunday,
    }


def activity_maps(panel: pd.DataFrame) -> dict[str, dict[int, int]]:
    maps = {}
    for uid, group in panel.groupby("ParticipantIdentifier", sort=False):
        maps[uid] = dict(zip(group["study_day"].astype(int), group["active"].astype(int)))
    return maps


def act7_of(day: int, active: dict[int, int]) -> float:
    values = []
    for t in range(day - 7, day):
        if t < 1:
            values.append(1.0)
        elif t in active:
            values.append(float(active[t]))
    if not values:
        raise SystemExit(f"act7 has no observed days before day {day}.")
    return float(np.mean(values))


def prior_gap_of(day: int, return_days: list[int]) -> int:
    """1 on and after the day activity resumes (spec §3: set prior_gap on return)."""
    return int(any(r <= day for r in return_days))


def model1_frame(panel: pd.DataFrame, gaps: pd.DataFrame) -> pd.DataFrame:
    maps = activity_maps(panel)
    returns = {
        uid: [int(x) for x in group["return_day"].dropna().astype(int)]
        for uid, group in gaps.groupby("uid")
    }
    bounds = (
        panel.groupby("ParticipantIdentifier")[["window_start", "window_end", "start_dow"]]
        .first()
    )
    rows = []
    for uid, active in maps.items():
        ret = returns.get(uid, [])
        start_day = int(bounds.loc[uid, "window_start"])
        end_day = int(bounds.loc[uid, "window_end"])
        dow = int(bounds.loc[uid, "start_dow"])
        for day in range(start_day, end_day):
            if active.get(day, 0) != 1 or active.get(day + 1) is None:
                continue
            if int(panel.loc[
                (panel["ParticipantIdentifier"] == uid) & (panel["study_day"] == day + 1),
                "observed",
            ].iloc[0]) != 1:
                continue
            wk = protocol_week(day, dow)
            rows.append(
                {
                    "uid": uid,
                    "day": day,
                    "y": int(active[day + 1] == 0),
                    "week_number": wk,
                    "month": float(month_c_from_week(wk)),
                    "act7": act7_of(day, active),
                    "prior_gap": prior_gap_of(day, ret),
                }
            )
    frame = pd.DataFrame(rows)
    boundary = gaps["start"].eq(gaps["window_start"])
    if int(frame["y"].sum()) != int((~boundary).sum()):
        raise SystemExit(
            f"Model 1 events {int(frame['y'].sum())} != interior gaps {int((~boundary).sum())}. "
            "A gap inside the window is not preceded by an observed active day."
        )
    # Ring-buffer replay includes the week-0 Sunday as observed history.
    for uid, active in maps.items():
        hist = [1.0] * 7
        group = frame.loc[frame["uid"].eq(uid)]
        lookup = dict(zip(group["day"].astype(int), group["act7"]))
        last = int(bounds.loc[uid, "window_end"])
        for day in range(1, last + 1):
            got = float(np.mean(hist))
            if day in lookup and abs(got - lookup[day]) > 1e-12:
                raise SystemExit(f"act7 replay mismatch for {uid} day {day}.")
            hist = hist[1:] + [float(active[day]) if day in active else 1.0]
    return frame


def fit_logit(frame: pd.DataFrame, columns: list[str]) -> sm.discrete.discrete_model.BinaryResultsWrapper:
    x = sm.add_constant(frame[columns], has_constant="add") if columns else np.ones((len(frame), 1))
    if columns:
        x = np.column_stack([np.ones(len(frame)), frame[columns].to_numpy(dtype=float)])
    else:
        x = np.ones((len(frame), 1))
    groups = pd.Categorical(frame["uid"]).codes
    try:
        result = sm.Logit(frame["y"].to_numpy(dtype=float), x).fit(
            disp=False,
            cov_type="cluster",
            cov_kwds={"groups": groups},
        )
    except Exception as exc:
        raise SystemExit(f"Model 1 did not converge: {exc}") from exc
    if not result.mle_retvals.get("converged", True):
        raise SystemExit("Model 1 logistic MLE did not converge.")
    return result


def calibration_tables(frame: pd.DataFrame, fitted: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = frame.copy()
    work["fitted"] = fitted
    by_week = (
        work.groupby("week_number", sort=True)
        .agg(n=("y", "size"), events=("y", "sum"), observed=("y", "mean"), fitted=("fitted", "mean"))
        .reset_index()
    )
    work["decile"] = pd.qcut(work["fitted"], 10, labels=False, duplicates="drop") + 1
    by_decile = (
        work.groupby("decile", sort=True)
        .agg(
            n=("y", "size"),
            observed=("y", "mean"),
            fitted=("fitted", "mean"),
            fitted_min=("fitted", "min"),
            fitted_max=("fitted", "max"),
        )
        .reset_index()
    )
    return by_week, by_decile


def model2_nll(eta: np.ndarray, prior: np.ndarray, length: np.ndarray, delta: np.ndarray) -> float:
    c0, c1, log_mu, log_alpha = (float(v) for v in eta)
    mu = math.exp(log_mu)
    alpha = math.exp(log_alpha)
    theta = np.exp(c0 + c1 * prior)
    f_now = weibull_F(length, mu, alpha)
    cens = delta == 0
    ll = np.empty(len(length), dtype=float)
    ll[cens] = -theta[cens] * f_now[cens]
    if np.any(~cens):
        f_prev = weibull_F(length[~cens] - 1.0, mu, alpha)
        step = theta[~cens] * (f_now[~cens] - f_prev)
        if np.any(step <= 0):
            return 1e12
        ll[~cens] = -theta[~cens] * f_prev + log1mexp(step)
    if not np.all(np.isfinite(ll)):
        return 1e12
    return float(-ll.sum())


def fit_model2(gaps: pd.DataFrame) -> dict:
    prior = gaps["prior_gap"].to_numpy(dtype=float)
    length = gaps["length"].to_numpy(dtype=float)
    delta = gaps["delta"].to_numpy(dtype=float)
    bounds = [
        (-8.0, 8.0),
        (-6.0, 6.0),
        (math.log(0.25), math.log(100.0)),
        (math.log(0.05), math.log(8.0)),
    ]
    starts = [
        np.array([0.0, 0.0, math.log(4.0), math.log(0.6)]),
        np.array([math.log(1.5), 0.0, math.log(3.0), math.log(0.8)]),
        np.array([math.log(3.0), -0.5, math.log(5.0), math.log(0.5)]),
        np.array([0.5, 0.5, math.log(2.0), math.log(1.0)]),
        np.array([-0.5, 0.2, math.log(8.0), math.log(0.4)]),
        np.array([1.0, -1.0, math.log(6.0), math.log(1.2)]),
        np.array([math.log(2.0), 0.3, math.log(3.5), math.log(0.7)]),
        np.array([0.2, 0.0, math.log(10.0), math.log(0.3)]),
        np.array([math.log(0.8), 0.8, math.log(2.5), math.log(0.45)]),
        np.array([-1.0, -0.4, math.log(12.0), math.log(0.9)]),
    ]
    runs = []
    for start in starts:
        result = minimize(
            model2_nll,
            start,
            args=(prior, length, delta),
            method="L-BFGS-B",
            bounds=bounds,
            options={"ftol": 1e-14, "gtol": 1e-8, "maxiter": 500},
        )
        runs.append(result)
    finite = [r for r in runs if np.isfinite(r.fun)]
    if not finite:
        raise SystemExit("Model 2 likelihood did not return a finite value from any start.")
    best = min(finite, key=lambda r: r.fun)
    grad = approx_fprime(best.x, model2_nll, 1e-6, prior, length, delta)
    if (not best.success) or float(np.max(np.abs(grad))) > 1e-3:
        raise SystemExit(
            "Model 2 likelihood did not converge: "
            f"success={best.success}, max|grad|={float(np.max(np.abs(grad))):.3g}, "
            f"message={best.message}"
        )
    for lo, hi, value in zip(*zip(*bounds), best.x):
        if value <= lo + 1e-3 or value >= hi - 1e-3:
            raise SystemExit(
                f"Model 2 hit a bound at {best.x.tolist()} (bounds {bounds}). Stop."
            )
    close = [r.x for r in finite if r.fun <= best.fun + 0.01]
    for other in close:
        if float(np.max(np.abs(other - best.x))) > 0.05:
            raise SystemExit(
                "Model 2 has two nearby modes with different parameters: "
                f"{best.x.tolist()} vs {other.tolist()}."
            )
    hess = numerical_hessian(lambda z: model2_nll(z, prior, length, delta), best.x)
    eigval, eigvec = np.linalg.eigh(hess)
    if float(np.min(eigval)) <= 1e-8:
        raise SystemExit(
            "Model 2 Hessian is not positive definite "
            f"(min eigenvalue {float(np.min(eigval)):.3g}). Stop."
        )
    cov = eigvec @ np.diag(1.0 / eigval) @ eigvec.T
    se = np.sqrt(np.diag(cov))
    c0, c1, log_mu, log_alpha = (float(v) for v in best.x)
    mu = math.exp(log_mu)
    alpha = math.exp(log_alpha)
    return {
        "c0": c0,
        "c1": c1,
        "mu": mu,
        "alpha": alpha,
        "se": {
            "c0": float(se[0]),
            "c1": float(se[1]),
            "log_mu": float(se[2]),
            "log_alpha": float(se[3]),
            "mu": float(mu * se[2]),
            "alpha": float(alpha * se[3]),
        },
        "nll": float(best.fun),
        "loglik": float(-best.fun),
        "n_starts": len(starts),
        "n_starts_near_best": len(close),
        "max_grad": float(np.max(np.abs(grad))),
    }


def numerical_hessian(fun, x: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    n = len(x)
    hess = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            xpp, xpm, xmp, xmm = (x.copy() for _ in range(4))
            xpp[i] += eps
            xpp[j] += eps
            xpm[i] += eps
            xpm[j] -= eps
            xmp[i] -= eps
            xmp[j] += eps
            xmm[i] -= eps
            xmm[j] -= eps
            hess[i, j] = (fun(xpp) - fun(xpm) - fun(xmp) + fun(xmm)) / (4.0 * eps * eps)
            hess[j, i] = hess[i, j]
    return hess


def theta_of(prior, fit: dict) -> np.ndarray:
    prior = np.asarray(prior, dtype=float)
    return np.exp(fit["c0"] + fit["c1"] * prior)


def kaplan_meier(time: np.ndarray, event: np.ndarray) -> pd.DataFrame:
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    order = np.argsort(time, kind="mergesort")
    time, event = time[order], event[order]
    rows = []
    surv = 1.0
    var_sum = 0.0
    n_risk = len(time)
    idx = 0
    for t in np.unique(time):
        deaths = 0
        cens = 0
        while idx < len(time) and time[idx] == t:
            deaths += int(event[idx] == 1)
            cens += int(event[idx] == 0)
            idx += 1
        if deaths and n_risk > 0:
            if n_risk > deaths:
                var_sum += deaths / (n_risk * (n_risk - deaths))
            else:
                var_sum = np.inf
            surv *= 1.0 - deaths / n_risk
        n_risk -= deaths + cens
        se = surv * math.sqrt(var_sum) if np.isfinite(var_sum) else np.nan
        rows.append({"time": float(t), "surv": float(surv), "se": float(se), "n_risk_after": int(n_risk)})
    return pd.DataFrame(rows)


def maller_zhou(time: np.ndarray, event: np.ndarray) -> dict:
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    km = kaplan_meier(time, event)
    t_n = float(time.max())
    t_star = float(time[event == 1].max())
    lower = 2.0 * t_star - t_n
    n_late = int(np.sum((event == 1) & (time > lower) & (time <= t_star)))
    s_tn = float(km.loc[km["time"] <= t_n, "surv"].iloc[-1])
    se = float(km.loc[km["time"] <= t_n, "se"].iloc[-1])
    return {
        "n": int(len(time)),
        "t_n": t_n,
        "t_star": t_star,
        "lower": lower,
        "N_n": n_late,
        "q_n": n_late / len(time),
        "S_tn": s_tn,
        "se_tn": se,
        "ci_low": s_tn - 1.96 * se if np.isfinite(se) else np.nan,
        "ci_high": s_tn + 1.96 * se if np.isfinite(se) else np.nan,
    }


def median_returner(theta: float, mu: float, alpha: float, horizon: int = 200) -> int | None:
    for day in range(1, horizon + 1):
        su = float(survival_uncured(np.array([day]), np.array([theta]), mu, alpha)[0])
        if su <= 0.5:
            return day
    return None


def derived_quantities(gaps: pd.DataFrame, fit: dict, lambda_bar: float) -> dict:
    mu, alpha = fit["mu"], fit["alpha"]
    cells = []
    for prior, month_num, label in (
        (0, 2, "prior_gap=0, month 2"),
        (1, 2, "prior_gap=1, month 2"),
        (0, 9, "prior_gap=0, month 9"),
        (1, 9, "prior_gap=1, month 9"),
    ):
        month = float(month_num - 2)
        theta = float(theta_of(prior, fit))
        cells.append(
            {
                "label": label,
                "prior_gap": prior,
                "month_number": month_num,
                "month": month,
                "theta": theta,
                "pi": float(math.exp(-theta)),
                "median_return": median_returner(theta, mu, alpha),
            }
        )
    theta_i = theta_of(gaps["prior_gap"], fit)
    pi_i = np.exp(-theta_i)
    open_rows = []
    for row in gaps.loc[gaps["delta"].eq(0)].sort_values("length").itertuples(index=False):
        theta = float(theta_of(row.prior_gap, fit))
        pi = math.exp(-theta)
        s_c = float(gap_survival(np.array([row.length]), np.array([theta]), mu, alpha)[0])
        open_rows.append(
            {
                "uid": row.uid,
                "c": int(row.length),
                "start": int(row.start),
                "week_number": int(row.week_number),
                "prior_gap": int(row.prior_gap),
                "pi": pi,
                "S_c": s_c,
                "p_permanent": pi / s_c if s_c > 0 else np.nan,
                "rule_14": int(row.length) >= 14,
            }
        )
    hazard_rows = []
    for day in range(1, 31):
        entry = {"day": day}
        for cell in cells:
            theta = cell["theta"]
            s_prev = 1.0 if day == 1 else float(
                gap_survival(np.array([day - 1]), np.array([theta]), mu, alpha)[0]
            )
            s_cur = float(gap_survival(np.array([day]), np.array([theta]), mu, alpha)[0])
            entry[cell["label"]] = 1.0 - s_cur / s_prev
        hazard_rows.append(entry)
    return {
        "cells": cells,
        "mean_pi": float(pi_i.mean()),
        "open": open_rows,
        "expected_dropouts": float(sum(r["p_permanent"] for r in open_rows)),
        "hazard": hazard_rows,
        "lambda_bar_times_mean_pi": float(lambda_bar * pi_i.mean()),
    }


def simulate_checked(
    n_paths: int,
    n_days: int,
    coef: np.ndarray,
    fit: dict,
    n_weeks: int,
    seed: int,
    u: np.ndarray | None = None,
    gamma: float = 0.0,
) -> dict:
    """Simulate, keeping the end-of-window inactive indicator from before the advance."""
    rng = np.random.default_rng(seed)
    month_of = np.array([month_c_of(day) for day in range(n_days + 2)])
    if month_c_of(7 * int(n_weeks)) != float(math.ceil(int(n_weeks) / 4.0) - 2):
        raise SystemExit("Month covariate does not match ceil(week/4) - 2.")
    frailty = None if u is None else np.asarray(u, dtype=float).reshape(n_paths)
    gamma = float(gamma)
    status = np.zeros(n_paths, dtype=np.int8)
    k = np.zeros(n_paths, dtype=np.int16)
    prior = np.zeros(n_paths, dtype=float)
    hist = np.ones((n_paths, 7), dtype=float)
    gap_start = np.zeros(n_paths, dtype=np.int16)
    theta_gap = np.ones(n_paths, dtype=float)
    n_gaps = np.zeros(n_paths, dtype=np.int16)
    inactive_days = np.zeros(n_paths, dtype=np.int16)
    dropout_day = np.full(n_paths, -1, dtype=np.int16)
    ended: list[int] = []
    censored: list[int] = []
    terminal14 = np.zeros(n_paths, dtype=bool)
    mu, alpha = fit["mu"], fit["alpha"]
    for day in range(1, n_days + 1):
        inactive_now = (status == 1) | (status == 2)
        inactive_days[inactive_now] += 1
        if day == n_days:
            lengths = np.zeros(n_paths, dtype=int)
            lengths[inactive_now] = n_days - gap_start[inactive_now] + 1
            censored.extend(int(v) for v in lengths[inactive_now])
            terminal14 = inactive_now & (lengths >= 14)
            break
        next_status = status.copy()
        next_k = k.copy()
        next_prior = prior.copy()
        active_idx = np.flatnonzero(status == 0)
        if len(active_idx):
            act7 = hist[active_idx].mean(axis=1)
            u_active = 0.0 if frailty is None else frailty[active_idx]
            eta = (
                coef[0]
                + coef[1] * month_of[day]
                + coef[2] * act7
                + coef[3] * prior[active_idx]
                + u_active
            )
            starters = active_idx[rng.random(len(active_idx)) < expit(eta)]
            if len(starters):
                u_start = 0.0 if frailty is None else frailty[starters]
                theta = np.exp(fit["c0"] + fit["c1"] * prior[starters] - gamma * u_start)
                perm = rng.random(len(starters)) < np.exp(-theta)
                perm_idx = starters[perm]
                temp_idx = starters[~perm]
                next_status[perm_idx] = 2
                next_status[temp_idx] = 1
                next_k[temp_idx] = 1
                theta_gap[temp_idx] = theta[~perm]
                gap_start[starters] = day + 1
                n_gaps[starters] += 1
                dropout_day[perm_idx] = day + 1
        inactive_idx = np.flatnonzero(status == 1)
        if len(inactive_idx):
            hu = hazard_uncured(k[inactive_idx].astype(float), theta_gap[inactive_idx], mu, alpha)
            returning = rng.random(len(inactive_idx)) < hu
            ret_idx = inactive_idx[returning]
            stay_idx = inactive_idx[~returning]
            ended.extend(int(v) for v in k[ret_idx])
            next_status[ret_idx] = 0
            next_prior[ret_idx] = 1.0
            next_k[ret_idx] = 0
            next_k[stay_idx] = k[stay_idx] + 1
        hist = np.concatenate([hist[:, 1:], (status == 0).astype(float)[:, None]], axis=1)
        status, k, prior = next_status, next_k, next_prior
    return {
        "n_gaps": n_gaps,
        "inactive_days": inactive_days,
        "dropout_day": dropout_day,
        "ended": np.asarray(ended, dtype=int),
        "censored": np.asarray(censored, dtype=int),
        "terminal14": terminal14,
        "n_days": n_days,
        "n_paths": n_paths,
    }


def sanity_check_simulator(fit: dict) -> None:
    rng_fit = dict(fit)
    zero = simulate_checked(400, 84, np.array([-40.0, 0.0, 0.0, 0.0]), rng_fit, N_WEEKS_FIT, seed=1)
    if int(zero["n_gaps"].sum()) != 0 or int(zero["inactive_days"].sum()) != 0:
        raise SystemExit("Sanity check failed: lambda ≈ 0 still produced gaps.")
    certain = dict(fit)
    certain["c0"] = -20.0
    certain["c1"] = 0.0
    dropped = simulate_checked(400, 84, np.array([40.0, 0.0, 0.0, 0.0]), certain, N_WEEKS_FIT, seed=2)
    if not np.all(dropped["dropout_day"] == 2):
        raise SystemExit("Sanity check failed: lambda ≈ 1 and pi ≈ 1 did not drop everyone on day 2.")
    theta = np.array([1.2])
    total = 0.0
    prev = 1.0
    for day in range(1, 400):
        cur = float(survival_uncured(np.array([day]), theta, fit["mu"], fit["alpha"])[0])
        total += prev - cur
        prev = cur
    if abs(total - 1.0) > 1e-3:
        raise SystemExit(f"S_u is not a proper survival function on returners (mass {total:.4f}).")


def km_quantile(time: np.ndarray, event: np.ndarray, p: float) -> float | None:
    km = kaplan_meier(time, event)
    hit = km.loc[km["surv"] <= 1.0 - p]
    if hit.empty:
        return None
    return float(hit["time"].iloc[0])


def summarize_sim(sim: dict) -> dict:
    n = sim["n_paths"]
    gaps = sim["n_gaps"]
    ended = sim["ended"]
    cens = sim["censored"]
    time = np.concatenate([ended, cens]) if len(cens) else ended
    event = np.concatenate([np.ones(len(ended), dtype=int), np.zeros(len(cens), dtype=int)])
    def q(arr, p):
        if len(arr) == 0:
            return None
        return float(np.quantile(arr, p))
    return {
        "mean_gaps": float(gaps.mean()),
        "p0": float(np.mean(gaps == 0)),
        "p1": float(np.mean(gaps == 1)),
        "p2": float(np.mean(gaps == 2)),
        "p3plus": float(np.mean(gaps >= 3)),
        "ended_p50": q(ended, 0.5),
        "ended_p75": q(ended, 0.75),
        "ended_p90": q(ended, 0.9),
        "km_p50": km_quantile(time, event, 0.5),
        "km_p90": km_quantile(time, event, 0.9),
        "inactive_fraction": float(sim["inactive_days"].mean() / sim["n_days"]),
        "terminal14": float(sim["terminal14"].mean()),
        "permanent": float(np.mean(sim["dropout_day"] > 0)),
        "n_paths": n,
    }


def step_xy(km: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    t = np.concatenate([[0.0], km["time"].to_numpy()])
    s = np.concatenate([[1.0], km["surv"].to_numpy()])
    return t, s


def fitted_mean_survival(gaps: pd.DataFrame, fit: dict, dmax: int, mask: np.ndarray | None = None) -> np.ndarray:
    sub = gaps if mask is None else gaps.loc[mask]
    if sub.empty:
        return np.full(dmax + 1, np.nan)
    theta = theta_of(sub["prior_gap"], fit)
    days = np.arange(dmax + 1)
    surv = np.array(
        [float(np.mean(gap_survival(np.full(len(theta), day), theta, fit["mu"], fit["alpha"]))) for day in days]
    )
    return surv


def save_figures(gaps: pd.DataFrame, fit: dict, by_week: pd.DataFrame, by_decile: pd.DataFrame, derived: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    time = gaps["length"].to_numpy(dtype=float)
    event = gaps["delta"].to_numpy(dtype=int)
    km = kaplan_meier(time, event)
    t, s = step_xy(km)
    fitted = fitted_mean_survival(gaps, fit, 80)
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.step(t, s, where="post", color="black", label="Kaplan–Meier")
    ax.step(np.arange(81), fitted, where="post", color="#b22222", label="Fitted mean S(d)")
    ax.set_xlim(0, 80)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Days since the gap started")
    ax.set_ylabel("Probability the gap has not ended")
    ax.set_title(f"All {len(gaps)} gaps")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "km_overall.png", dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.8), sharey=True)
    for ax, value, title in zip(axes, (0, 1), ("First gap", "After a completed gap")):
        mask = gaps["prior_gap"].eq(value).to_numpy()
        km_s = kaplan_meier(time[mask], event[mask])
        tt, ss = step_xy(km_s)
        fit_s = fitted_mean_survival(gaps, fit, 80, gaps["prior_gap"].eq(value))
        ax.step(tt, ss, where="post", color="black", label="Kaplan–Meier")
        ax.step(np.arange(81), fit_s, where="post", color="#b22222", label="Fitted")
        ax.set_title(f"{title} (n={int(mask.sum())})")
        ax.set_xlim(0, 80)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("Days since the gap started")
    axes[0].set_ylabel("Probability the gap has not ended")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "km_by_prior_gap.png", dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.8), sharey=True)
    for ax, value, title in zip(axes, (1, 0), ("Start in weeks 1–6", "Start in weeks 7–12")):
        mask = gaps["early"].eq(value)
        km_s = kaplan_meier(time[mask.to_numpy()], event[mask.to_numpy()])
        tt, ss = step_xy(km_s)
        fit_s = fitted_mean_survival(gaps, fit, 80, mask)
        ax.step(tt, ss, where="post", color="black", label="Kaplan–Meier")
        ax.step(np.arange(81), fit_s, where="post", color="#b22222", label="Fitted")
        ax.set_title(f"{title} (n={int(mask.sum())})")
        ax.set_xlim(0, 80)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("Days since the gap started")
    axes[0].set_ylabel("Probability the gap has not ended")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "km_by_start.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    days = [row["day"] for row in derived["hazard"]]
    for cell in derived["cells"]:
        ax.plot(days, [row[cell["label"]] for row in derived["hazard"]], label=cell["label"])
    ax.set_xlabel("Day of the gap")
    ax.set_ylabel("Return hazard h(d)")
    ax.set_title("Promotion-time hazard, including the permanent fraction")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "return_hazard.png", dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.8))
    axes[0].plot(by_week["week_number"], by_week["observed"], "o", color="black", label="Observed")
    axes[0].plot(by_week["week_number"], by_week["fitted"], "s", color="#b22222", label="Fitted")
    axes[0].set_xlabel("Study week of the active day")
    axes[0].set_ylabel("P(gap starts tomorrow)")
    axes[0].set_title("By study week")
    axes[0].legend(frameon=False)
    axes[1].plot(by_decile["fitted"], by_decile["observed"], "o", color="black")
    hi = max(float(by_decile["fitted"].max()), float(by_decile["observed"].max())) * 1.05
    axes[1].plot([0, hi], [0, hi], color="0.6", lw=1)
    axes[1].set_xlabel("Mean fitted λ")
    axes[1].set_ylabel("Observed rate")
    axes[1].set_title("Deciles of fitted λ")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "model1_calibration.png", dpi=140)
    plt.close(fig)


def fmt(x, digits=3) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{x:.{digits}f}"


def _stack_paths(parts: list[dict]) -> dict:
    def cat(key):
        return np.concatenate([np.atleast_1d(p[key]) for p in parts])

    return {
        "n_gaps": cat("n_gaps"),
        "inactive_days": cat("inactive_days"),
        "dropout_day": cat("dropout_day"),
        "ended": cat("ended"),
        "censored": cat("censored"),
        "terminal14": cat("terminal14"),
        "n_days": int(parts[0]["n_days"]),
        "n_paths": len(parts),
    }


def _uid_paths(n_paths, n_days, n_weeks, coef, fit, uids, base_seed) -> dict:
    """One path per episode seed, with ``SeedSequence([seed, uid, 771337])``."""
    from activity_process import simulate_activity_paths

    uids = [int(u) for u in uids]
    parts = []
    for i in range(int(n_paths)):
        uid = uids[i % len(uids)]
        seed = int(base_seed) + i // len(uids)
        rng = np.random.default_rng(dropout_seed_sequence(seed, uid))
        parts.append(simulate_activity_paths(1, n_days, coef, fit, n_weeks, rng))
    return _stack_paths(parts)


def simulator_check_lines(data: dict, sim84_ref: dict, sim252_ref: dict) -> list[str]:
    """2000-path check of the episode generator against the 10,000-path validation."""
    from activity_process import (
        coefficients_from_record,
        prepare_activity,
        simulate_activity_paths,
        summarize_paths,
    )

    record = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))["activity_process"]
    coef, fit = coefficients_from_record(record)
    meta = pd.read_csv(META_PATH)
    uids = sorted(
        int(u) for u in meta.loc[meta["in_modelled"].astype(bool), "ParticipantIdentifier"]
    )
    sim84 = summarize_paths(_uid_paths(2000, 84, N_WEEKS_FIT, coef, fit, uids, 20261022))
    sim252 = summarize_paths(_uid_paths(2000, 252, N_WEEKS_LONG, coef, fit, uids, 20261023))
    examples = []
    for uid in (13, 22, 18):
        rng = np.random.default_rng(dropout_seed_sequence(1, uid))
        coef_u, fit_u, u, gamma = prepare_activity(record, "pooled", uid, rng)
        sim = simulate_activity_paths(
            1, 252, coef_u, fit_u, N_WEEKS_LONG, rng,
            u=None if u is None else np.asarray([u], dtype=float),
            gamma=gamma,
        )
        n_gaps = int(sim["n_gaps"][0])
        gap_word = "gap" if n_gaps == 1 else "gaps"
        examples.append(
            f"{uid}: dropout day {int(sim['dropout_day'][0])}, "
            f"{n_gaps} {gap_word}, {int(sim['inactive_days'][0])} inactive days"
        )
    lines = []
    a = lines.append
    a("## Simulator check")
    a("")
    a("The episode draws the activity path once, before the day loop, from the dropout")
    a("RNG stream (`SeedSequence` `[episode seed, participant, 771337]`). The same stream")
    a("is the legacy constant-hazard draw when no activity record is loaded. The path is")
    a("two days longer than the episode so the last Sunday's Monday and Tuesday are known.")
    a("`lambda = 0` is all active and does not draw on that stream. The main episode RNG")
    a("is `numpy.random.seed(episode seed)` and does not include the participant.")
    a("Users 13, 22, and 18 with episode seed 1, in the default pooled mode, on a 252-day path: " + "; ".join(examples) + ".")
    a("")
    a("The table is 2000 pooled-mode paths. The modelled participants cycle through")
    a("the uid slot, and the episode seed starts at 20261022 (84 days, `n_weeks=12`) or")
    a("20261023 (252 days, `n_weeks=36`) and increases after each full pass. The middle")
    a("column is the 10,000-path validation above (seeds 20261020 and 20261021).")
    a("")
    a(f"| Quantity | Data ({len(uids)} people) | 10,000 paths | 2000 paths |")
    a("|---|---:|---:|---:|")
    a(f"| Mean gaps per person | {fmt(data['mean_gaps'], 2)} | {fmt(sim84_ref['mean_gaps'], 2)} | {fmt(sim84['mean_gaps'], 2)} |")
    a(f"| Proportion with 0 gaps | {fmt(data['p0'], 3)} | {fmt(sim84_ref['p0'], 3)} | {fmt(sim84['p0'], 3)} |")
    a(f"| Proportion with 1 gap | {fmt(data['p1'], 3)} | {fmt(sim84_ref['p1'], 3)} | {fmt(sim84['p1'], 3)} |")
    a(f"| Proportion with 2 gaps | {fmt(data['p2'], 3)} | {fmt(sim84_ref['p2'], 3)} | {fmt(sim84['p2'], 3)} |")
    a(f"| Proportion with 3 or more | {fmt(data['p3plus'], 3)} | {fmt(sim84_ref['p3plus'], 3)} | {fmt(sim84['p3plus'], 3)} |")
    a(f"| Ended-gap length, median | {fmt(data['ended_p50'], 1)} | {fmt(sim84_ref['ended_p50'], 1)} | {fmt(sim84['ended_p50'], 1)} |")
    a(f"| Ended-gap length, 75th | {fmt(data['ended_p75'], 1)} | {fmt(sim84_ref['ended_p75'], 1)} | {fmt(sim84['ended_p75'], 1)} |")
    a(f"| Ended-gap length, 90th | {fmt(data['ended_p90'], 1)} | {fmt(sim84_ref['ended_p90'], 1)} | {fmt(sim84['ended_p90'], 1)} |")
    a(f"| Kaplan–Meier 90th percentile | {fmt(data['km_p90'], 1)} | {fmt(sim84_ref['km_p90'], 1)} | {fmt(sim84['km_p90'], 1)} |")
    a(f"| Fraction of inactive days | {fmt(data['inactive_fraction'], 3)} | {fmt(sim84_ref['inactive_fraction'], 3)} | {fmt(sim84['inactive_fraction'], 3)} |")
    a(f"| Terminal gap of 14 days or more | {fmt(data['terminal14'], 3)} | {fmt(sim84_ref['terminal14'], 3)} | {fmt(sim84['terminal14'], 3)} |")
    a(f"| Latent permanent by day 84 | — | {fmt(sim84_ref['permanent'], 3)} | {fmt(sim84['permanent'], 3)} |")
    a(f"| Latent permanent by day 252 | — | {fmt(sim252_ref['permanent'], 3)} | {fmt(sim252['permanent'], 3)} |")
    a("")
    a("At n = 2000 a proportion near 0.10 has a Monte Carlo standard error of about 0.007,")
    a("and a proportion near 0.60 has one of about 0.011. The 2000 paths use the pooled block.")
    a("The default episode mode is pooled. The 10,000 paths are the pooled validation above.")
    a("The Step 1 constant-hazard extrapolation to day 252 is 0.233.")
    a("")
    return lines


def write_report(payload: dict) -> None:
    m1 = payload["model1"]
    m2 = payload["model2"]
    week = payload["by_week"]
    dec = payload["by_decile"]
    derived = payload["derived"]
    data = payload["data_summary"]
    sim = payload["sim84"]
    long = payload["sim252"]
    mz = payload["mz"]
    mz0 = payload["mz_without_112"]
    lines = []
    a = lines.append
    a(f"# Activity process fit ({SOURCE_DATE})")
    a("")
    cohort = payload["cohort"]
    a(f"Two transition models on daily activity for the {cohort['n_users']} modelled participants.")
    a("The process is exogenous: no latent perceived utility and no action terms.")
    a("Scripts 01–03 and 06–07 were not rerun. The simulator and the script-06 likelihood are not in this fit.")
    a("The activity process is pooled across participants; dropout is calibrated in aggregate but not assigned to the participants who disengaged in the MRT. The shared-frailty mode is experimental (gamma is bimodal) and not used.")
    a("")
    a("## Data")
    a("")
    a(f"Source: `{SOURCE_FILE}`, restricted with `in_modelled` in `participant_dropout.csv`.")
    a("`active_d` is a page view, wear, or a daily or weekly survey submission on that local day.")
    a("Study day 1 is the first local end-of-day date and is active.")
    a(f"The fit window is protocol weeks 1–12 (Monday–Sunday): days 1–84 for the {cohort['n_monday']} Monday starters,")
    a(f"and days 2–85 for the {cohort['n_sunday']} Sunday starters. Study day 1 of a Sunday starter is the partial")
    a("week-0 Sunday. It is history for `act7` and is not inside a gap. `study_day` and `dropout_day` are unchanged.")
    a("A gap is a run of inactive days inside that window. It is ended when an active day follows inside the window;")
    a("otherwise it is censored at the week-12 Sunday. A Sunday starter's gap may begin on day 2.")
    a("")
    a("| Check | Value |")
    a("|---|---:|")
    a(f"| Modelled users | {cohort['n_users']} |")
    a(f"| Gaps | {cohort['n_gaps']} ({cohort['n_ended']} ended, {cohort['n_open']} censored) |")
    a(f"| Users with / without a gap | {cohort['n_with_gap']} / {cohort['n_without_gap']} |")
    a(f"| Active person-days in the model 1 risk set | {payload['n_risk']} |")
    a(f"| Gap starts in that risk set | {payload['n_events']} |")
    a("")
    open_txt = ", ".join(str(x) for x in cohort["open_lengths"]) or "none"
    a("The risk set is an observed active day inside the window whose next day is also in the window.")
    a(f"The censored lengths are {open_txt}.")
    a("A gap may start on the window's first day for a Sunday starter; that gap is in model 2 and not in the model 1 risk set.")
    a("None of the fitted gaps start there. The previous protocol-week fit, with activity equal to page view or wear only,")
    a("had a risk set of 3536 and 57 gaps (48 ended, 9 censored; censored lengths 1, 1, 3, 9, 12, 23, 24, 27, 76).")
    a("This fit uses the same weeks. The one added gap is user 248: study day 55 (2026-05-02) is a survey submission")
    a("with no page view and no wear, so it was inactive before. It splits one ended 3-day gap (days 54–56) into")
    a("two ended 1-day gaps. The censored lengths are unchanged, and the risk set gains that Saturday.")
    a("")
    a("`month_c` is the protocol month minus 2. Protocol month is `ceil(week / 4)`:")
    a("weeks 1–4, 5–8, and 9–12 are months 1, 2, and 3, so `month_c` is −1, 0, and 1.")
    a("A Monday-start simulator episode uses the same months. Over 36 weeks the month runs 1–9")
    a("(`month_c` from −1 to 7), with no stretch and no cap. Week 0, a Sunday starter's first day,")
    a("is month 1. It is history for `act7` and is not a model row.")
    a("`act7` is the fraction of the previous seven days that were active. Days before study day 1 count as active.")
    a("The observed week-0 Sunday is used when it falls in those seven days.")
    a("`prior_gap` becomes 1 on the day activity resumes and stays 1. That is the simulator rule")
    a("(\"on return, set prior_gap = 1\"), so the return day itself already carries it.")
    a("")
    a("## Model 1: active → inactive")
    a("")
    a("One row per active day `d` with day `d+1` observed. Outcome: day `d+1` is inactive.")
    a("Logistic maximum likelihood, cluster-robust standard errors by participant.")
    a("")
    a(f"Intercept-only rate λ̄ = {fmt(m1['lambda_bar'], 4)} per active day "
      f"(cluster SE {fmt(m1['lambda_bar_se'], 4)}; logit intercept {fmt(m1['intercept_only'], 3)}, "
      f"SE {fmt(m1['intercept_only_se'], 3)}).")
    a(f"Crude rate {payload['n_events']}/{payload['n_risk']} = {fmt(payload['n_events']/payload['n_risk'], 4)}.")
    a(f"Among days with `prior_gap = 0`, the first-gap rate is {fmt(m1['first_gap_rate'], 4)} "
      f"({m1['first_gap_events']} events / {m1['first_gap_days']} days).")
    a("")
    a("| Term | Coefficient | Cluster SE | Odds ratio | 95% CI | p |")
    a("|---|---:|---:|---:|---|---:|")
    for name, coef, se, odd, lo, hi, pval in zip(
        MODEL1_NAMES, m1["coef"], m1["se"], m1["or"], m1["or_low"], m1["or_high"], m1["p"]
    ):
        a(f"| {name} | {fmt(coef, 3)} | {fmt(se, 3)} | {fmt(odd, 3)} | {fmt(lo, 3)}–{fmt(hi, 3)} | {fmt(pval, 3)} |")
    a("")
    a(f"Joint Wald test of the three slopes, using the cluster covariance: "
      f"statistic {fmt(m1['wald_stat'], 2)} on {m1['wald_df']} df, p = {fmt(m1['wald_p'], 3)}.")
    a(f"Log likelihood {fmt(m1['llf'], 1)} versus {fmt(m1['llf_null'], 1)} for the intercept-only model.")
    a("")
    a("Reference predictions from the fitted logit:")
    a("")
    a("| Month covariate | act7 | prior_gap | Fitted λ |")
    a("|---:|---:|---:|---:|")
    for row in m1["references"]:
        a(f"| {fmt(row['month'], 2)} | {fmt(row['act7'], 1)} | {row['prior_gap']} | {fmt(row['lambda'], 4)} |")
    a("")
    a("### Calibration")
    a("")
    a("Fitted probabilities against the observed gap-start rate. Week is the study week of the active day.")
    a("")
    a("| Week | Days | Events | Observed | Fitted |")
    a("|---:|---:|---:|---:|---:|")
    for row in week.itertuples(index=False):
        a(f"| {int(row.week_number)} | {int(row.n)} | {int(row.events)} | {fmt(row.observed, 4)} | {fmt(row.fitted, 4)} |")
    a("")
    a("| Decile | Days | Observed | Mean fitted | Fitted range |")
    a("|---:|---:|---:|---:|---|")
    for row in dec.itertuples(index=False):
        a(
            f"| {int(row.decile)} | {int(row.n)} | {fmt(row.observed, 4)} | {fmt(row.fitted, 4)} | "
            f"{fmt(row.fitted_min, 4)}–{fmt(row.fitted_max, 4)} |"
        )
    a("")
    a("![Model 1 calibration](model1_calibration.png)")
    a("")
    a("## Model 2: inactive → active, with a permanent fraction")
    a("")
    a(f"Promotion-time cure model on the {cohort['n_gaps']} gaps. Discrete Weibull kernel `F(d) = 1 - exp(-(d/μ)^α)`,")
    a("`S(d) = exp(-θ F(d))`, `π = exp(-θ)`. Covariates sit on `log θ` at the gap-start day;")
    a("`F` has no covariates. An ended gap of length `t` contributes `S(t-1) - S(t)`; a gap still open")
    a("after `c` days contributes `S(c)`.")
    a("")
    a(f"Numerical maximum likelihood from {m2['n_starts']} starts. "
      f"{m2['n_starts_near_best']} starts landed on this solution (log likelihood {fmt(m2['loglik'], 2)}).")
    a(f"Hessian standard errors treat the {cohort['n_gaps']} gaps as independent. "
      f"They are too small: the gaps come from {cohort['n_with_gap']} people.")
    a("No bootstrap and no profile interval. The pooled block stores these point estimates.")
    a("The pooled block is what the simulator loads.")
    a("")
    a("| Parameter | Estimate | Hessian SE |")
    a("|---|---:|---:|")
    a(f"| c0 | {fmt(m2['c0'], 3)} | {fmt(m2['se']['c0'], 3)} |")
    a(f"| c1 (prior_gap) | {fmt(m2['c1'], 3)} | {fmt(m2['se']['c1'], 3)} |")
    a(f"| μ | {fmt(m2['mu'], 3)} | {fmt(m2['se']['mu'], 3)} |")
    a(f"| α | {fmt(m2['alpha'], 3)} | {fmt(m2['se']['alpha'], 3)} |")
    a("")
    a("Log theta is `c0 + c1 * prior_gap`. Month is in model 1 only.")
    a("μ and α standard errors are the delta method from the SEs of log μ and log α.")
    a(f"α {'<' if m2['alpha'] < 1 else '≥'} 1, so the Weibull return hazard is "
      f"{'decreasing' if m2['alpha'] < 1 else 'not decreasing'}.")
    if abs(m2["c1"]) < m2["se"]["c1"]:
        a("c1 is smaller than its standard error. π barely moves with prior gap.")
        a("μ itself is imprecise relative to its point estimate.")
    a("")
    a("### Fitted survival against Kaplan–Meier")
    a("")
    a("The fitted curve in each panel is the average of `S(d | x_i)` over the gaps in that panel,")
    a("not the survival at the average covariate.")
    a("")
    a("![All gaps](km_overall.png)")
    a("")
    a("![By prior gap](km_by_prior_gap.png)")
    a("")
    a("![By early or late start](km_by_start.png)")
    a("")
    a("The overall curve follows the Kaplan–Meier through the first two weeks and ends on the same plateau.")
    a("Between the early returns and user 112's day-56 return the data are flat while the Weibull keeps declining;")
    a("that is the proportional-hazards constraint, which ties return speed to π.")
    a("The first-gap panel is where that stretch shows. Later gaps and late starts sit on the fitted curve,")
    a("and the strata do not separate much, in line with the small c1.")
    a("")
    a("### Sufficient follow-up")
    a("")
    a("Maller–Zhou (1994): `t_n` is the largest gap time, censored gaps included; `t*_n` is the largest")
    a("ended gap; `N_n` counts ended gaps in `(2 t*_n - t_n, t*_n]`; `q_n = N_n / n`.")
    a("The second column drops user 112's ended 56-day gap and does not refit.")
    a("")
    a(f"| | All {cohort['n_gaps']} gaps | Without 112's 56-day gap |")
    a("|---|---:|---:|")
    a(f"| n | {mz['n']} | {mz0['n']} |")
    a(f"| t_n | {fmt(mz['t_n'], 0)} | {fmt(mz0['t_n'], 0)} |")
    a(f"| t*_n | {fmt(mz['t_star'], 0)} | {fmt(mz0['t_star'], 0)} |")
    a(f"| N_n | {mz['N_n']} | {mz0['N_n']} |")
    a(f"| q_n | {fmt(mz['q_n'], 3)} | {fmt(mz0['q_n'], 3)} |")
    a(f"| S_KM(t_n) | {fmt(mz['S_tn'], 3)} | {fmt(mz0['S_tn'], 3)} |")
    a(f"| Greenwood 95% CI | {fmt(mz['ci_low'], 3)}–{fmt(mz['ci_high'], 3)} | {fmt(mz0['ci_low'], 3)}–{fmt(mz0['ci_high'], 3)} |")
    a("")
    a(f"S_KM(t_n) {'>' if mz['S_tn'] > 0 else '='} 0 because the largest recorded gap time is "
      f"{'censored' if mz['S_tn'] > 0 else 'an ended gap'}.")
    a("The Greenwood interval includes 0 when the 56-day return is kept and excludes 0 when that gap is dropped.")
    a("Maller and Zhou (1992) read a positive plateau at the largest time as evidence of a permanent fraction")
    a(f"when follow-up reaches past the return distribution. With the 56-day return included, q_n is {mz['N_n']}/{mz['n']}:")
    a("that single late return sets the plateau. Without it, q_n is large because the next longest ended gap")
    a("is much shorter than the longest censored gap, so the follow-up window covers essentially every return.")
    a("The fit keeps 112.")
    a("")
    a("### Permanent fraction, return hazard, and the 14-day count")
    a("")
    a("`π(x)` depends on prior_gap only. The month-2 and month-9 rows share θ.")
    a("The median is the gap length among people who do return, from `S_u`.")
    a("")
    a("| Covariates | θ | π | Median length if they return |")
    a("|---|---:|---:|---:|")
    for cell in derived["cells"]:
        a(f"| {cell['label']} | {fmt(cell['theta'], 3)} | {fmt(cell['pi'], 3)} | {cell['median_return']} |")
    a("")
    a(f"Average of π_i over the {cohort['n_gaps']} observed gaps: {fmt(derived['mean_pi'], 3)}.")
    a("")
    a("`h(d) = 1 - S(d)/S(d-1)` includes the permanent fraction, so it falls toward 0 even if returners")
    a("would still come back. Selected days:")
    a("")
    a("| Day | " + " | ".join(cell["label"] for cell in derived["cells"]) + " |")
    a("|---:|" + "|".join(["---:"] * len(derived["cells"])) + "|")
    for row in derived["hazard"]:
        if row["day"] not in (1, 2, 3, 7, 14, 30):
            continue
        vals = " | ".join(fmt(row[cell["label"]], 3) for cell in derived["cells"])
        a(f"| {row['day']} | {vals} |")
    a("")
    a("![Return hazard](return_hazard.png)")
    a("")
    a("For each gap still open at the week-12 Sunday, `P(permanent | open c days) = π / S(c)`.")
    a("")
    a("| User | Open days | Start week | prior_gap | π | S(c) | P(permanent) | 14-day rule |")
    a("|---:|---:|---:|---:|---:|---:|---:|---|")
    for row in derived["open"]:
        a(
            f"| {row['uid']} | {row['c']} | {row['week_number']} | {row['prior_gap']} | "
            f"{fmt(row['pi'], 3)} | {fmt(row['S_c'], 3)} | {fmt(row['p_permanent'], 3)} | "
            f"{'yes' if row['rule_14'] else 'no'} |"
        )
    a("")
    a(f"Sum of these {len(derived['open'])} probabilities: **{fmt(derived['expected_dropouts'], 2)}** expected permanent gaps.")
    ruled = [row for row in derived["open"] if row["rule_14"]]
    ruled_ids = ", ".join(str(row["uid"]) for row in ruled)
    ruled_lengths = ", ".join(str(row["c"]) for row in ruled)
    a(f"The 14-day rule calls {len(ruled)} of these gaps dropouts ({ruled_ids}; lengths {ruled_lengths}).")
    a("")
    long_open = [row for row in derived["open"] if 14 <= row["c"] < 70]
    if long_open:
        lo = min(row["p_permanent"] for row in long_open)
        hi = max(row["p_permanent"] for row in long_open)
        span = f"{min(row['c'] for row in long_open)}–{max(row['c'] for row in long_open)}-day gaps to {lo:.2f}–{hi:.2f}"
    else:
        span = "the multi-week open gaps"
    a("The long-gap tail of model 2 rests on one return (112, day 56): the Weibull declines")
    a("over days 16–56 where the Kaplan–Meier is flat, which lowers P(permanent) for the")
    a(f"{span} and makes 3–4-week temporary gaps more frequent in simulation")
    a(f"(simulated 90th percentile {fmt(sim['km_p90'], 0)} versus {fmt(data['km_p90'], 0)} in the data).")
    a(f"Aggregate dropout is calibrated ({fmt(derived['expected_dropouts'], 2)} versus 4; "
      f"{fmt(100 * sim['terminal14'], 1)}% versus {fmt(100 * data['terminal14'], 1)}%).")
    a("")
    active_fraction = 1.0 - data["inactive_fraction"]
    per_person_day = derived["lambda_bar_times_mean_pi"] * active_fraction
    a(f"λ̄ × (average π over the {cohort['n_gaps']} gaps) = {fmt(derived['lambda_bar_times_mean_pi'] * 100, 3)}% per active day.")
    a(f"λ̄ × π × the active-day fraction ({fmt(active_fraction, 3)}) is "
      f"{fmt(per_person_day * 100, 3)}% per person-day.")
    a(f"The constant-hazard calibration is {HAZARD_CURRENT * 100:.3f}% per person-day (4/3809).")
    a("The person-day rates use that same denominator.")
    a("")
    prev = payload.get("previous")
    if prev:
        a("## Beside the previous fit")
        a("")
        a("The previous fit put month in model 2 as well (`c2` on log θ). This fit drops that term.")
        a("Pooled model 1 is the same likelihood. The frailty block is refit, so its model 1 means move.")
        a("")
        old_m1 = prev["model1"]["coefficients"]
        old_m2 = prev["model2"]
        old_names = prev.get("frailty", {}).get("names", [])
        old_mean = prev.get("frailty", {}).get("posterior_mean", [])
        old_fr = dict(zip(old_names, old_mean))
        a("| Term | Previous | This fit |")
        a("|---|---:|---:|")
        for i, name in enumerate(MODEL1_NAMES):
            a(f"| pooled {name} | {fmt(old_m1[i], 3)} | {fmt(m1['coef'][i], 3)} |")
        a(f"| pooled c0 | {fmt(old_m2['c0'], 3)} | {fmt(m2['c0'], 3)} |")
        a(f"| pooled c1 | {fmt(old_m2['c1'], 3)} | {fmt(m2['c1'], 3)} |")
        a(f"| pooled c2 | {fmt(old_m2.get('c2'), 3)} | — |")
        a(f"| pooled μ | {fmt(old_m2['mu'], 3)} | {fmt(m2['mu'], 3)} |")
        a(f"| pooled α | {fmt(old_m2['alpha'], 3)} | {fmt(m2['alpha'], 3)} |")
        fr_now = payload.get("frailty")
        if fr_now is not None:
            new_fr = dict(zip(fr_now["names"], fr_now["mean"]))
            for name in ("a0", "a1_month", "a2_act7", "a3_prior_gap", "c0", "c1_prior_gap", "c2_month", "mu", "alpha", "tau", "gamma"):
                new_name = name
                shown = new_fr.get(new_name)
                if name == "c2_month":
                    shown = None
                a(f"| frailty {name} | {fmt(old_fr.get(name), 3)} | {fmt(shown, 3)} |")
        a("")
    a("## Paths simulated from the fitted process")
    a("")
    a("10,000 paths. Modelled users cycle through the uid slot. The episode seed starts at 20261020")
    a("(84 days) or 20261021 (252 days). Frailty paths draw `u` from that user's posterior on the activity stream.")
    a("Day 1 starts active. `month_c` is computed from the path's protocol month and enters model 1 only.")
    a("A permanent draw stays inactive through day 84 and is censored for the gap-length comparison, as an open gap is in the data.")
    a("")
    a(f"| Quantity | Data ({cohort['n_users']} people) | Simulated paths |")
    a("|---|---:|---:|")
    a(f"| Mean gaps per person | {fmt(data['mean_gaps'], 2)} | {fmt(sim['mean_gaps'], 2)} |")
    a(f"| Proportion with 0 gaps | {fmt(data['p0'], 3)} | {fmt(sim['p0'], 3)} |")
    a(f"| Proportion with 1 gap | {fmt(data['p1'], 3)} | {fmt(sim['p1'], 3)} |")
    a(f"| Proportion with 2 gaps | {fmt(data['p2'], 3)} | {fmt(sim['p2'], 3)} |")
    a(f"| Proportion with 3 or more | {fmt(data['p3plus'], 3)} | {fmt(sim['p3plus'], 3)} |")
    a(f"| Ended-gap length, median | {fmt(data['ended_p50'], 1)} | {fmt(sim['ended_p50'], 1)} |")
    a(f"| Ended-gap length, 75th | {fmt(data['ended_p75'], 1)} | {fmt(sim['ended_p75'], 1)} |")
    a(f"| Ended-gap length, 90th | {fmt(data['ended_p90'], 1)} | {fmt(sim['ended_p90'], 1)} |")
    a(f"| Kaplan–Meier median length | {data['km_p50'] if data['km_p50'] is not None else '—'} | {sim['km_p50'] if sim['km_p50'] is not None else '—'} |")
    a(f"| Kaplan–Meier 90th percentile | {data['km_p90'] if data['km_p90'] is not None else '—'} | {sim['km_p90'] if sim['km_p90'] is not None else '—'} |")
    a(f"| Fraction of inactive days | {fmt(data['inactive_fraction'], 3)} | {fmt(sim['inactive_fraction'], 3)} |")
    a(f"| Terminal gap of 14 days or more | {fmt(data['terminal14'], 3)} | {fmt(sim['terminal14'], 3)} |")
    a("")
    a(f"Monte Carlo standard errors at n = {sim['n_paths']} are about "
      f"{fmt(math.sqrt(0.25/sim['n_paths']), 3)} for a proportion near 1/2 and smaller for rarer events.")
    a("The zero-gap share, the median ended gap, the inactive-day fraction, and the 14-day terminal share")
    a("are the close comparisons. The data Kaplan–Meier 90th percentile is the single day-56 return;")
    a("the simulated paths put fewer people on exactly one gap and more on three or more.")
    a(f"The latent fraction permanently gone by day 84 in these paths is {fmt(sim['permanent'], 3)}.")
    a("That fraction is not observed in the data; the 14-day row above is the comparable quantity.")
    a("")
    a("### 36-week extrapolation")
    a("")
    a("Separate 10,000 paths of 252 days. Protocol month runs 1 through 9. `month_c` runs from −1 to 7")
    a("on the same scale as the 12-week fit: month 2 is 0 in both, and month 9 is not folded back.")
    odds9 = math.exp(7.0 * float(m1["coef"][1]))
    a(f"Month 9 versus month 2 multiplies the model-1 odds of starting a gap by {fmt(odds9, 3)}.")
    a("Model 2 has no month term, so θ does not change between those months.")
    a(f"Fraction permanently gone by day 252: **{fmt(long['permanent'], 3)}**.")
    a("The stored constant-hazard extrapolation over the same horizon is 0.233 (`dropout_cumulative_day252`).")
    a(f"The 36-week figure is driven by prior_gap (odds ratio {fmt(m1['or'][3], 1)}, permanent once set):")
    a("event dependence and stayer heterogeneity are not distinguishable in 84 days")
    a("(see the time-to-first-gap Kaplan–Meier). This is an assumption-driven extrapolation.")
    a(f"The simulated share with 3 or more gaps ({fmt(sim['p3plus'], 2)} versus {fmt(data['p3plus'], 2)})")
    a("suggests prior_gap slightly overstates event dependence.")
    a("The month slope is the linear extrapolation past month 3, the last month in the fit.")
    a("")
    fr = payload.get("frailty")
    fr84 = payload.get("frailty_sim84")
    fr252 = payload.get("frailty_sim252")
    if fr is not None and fr84 is not None and fr252 is not None:
        a("## Shared frailty")
        a("")
        a("The same two regressions share one user frailty `u_p`. Model 1 adds `u_p` to the logit.")
        a("Model 2 subtracts `gamma * u_p` from log theta at the gap start.")
        a("Priors: regression coefficients N(0, 5²), log μ ~ N(log 5, 1), log α ~ N(0, 1),")
        a("τ ~ HalfNormal(1), γ ~ N(0, 1). Four NUTS chains, 5000 tuning steps, 30000 draws,")
        a(f"target acceptance 0.99, seed {fr['seed']}. Chains start at the pooled maximum likelihood,")
        a("with τ = 0.5 and the frailties at 0. The mass matrix is dense. While sampling, u on the gap rows is residualized")
        a("on the intercept and prior_gap; c0 and c1 are then mapped back to")
        a("log theta = c0 + c1*prior_gap − gamma*u_p.")
        a(f"Divergences {fr['divergences']}. Maximum R-hat {fmt(fr['max_rhat'], 3)}. "
          f"Minimum bulk ESS {fmt(fr['min_ess'], 0)}.")
        a("")
        a("| Term | Posterior mean | SD | 95% interval |")
        a("|---|---:|---:|---|")
        for name, mean, sd, lo, hi in zip(
            fr["names"], fr["mean"], fr["sd"], fr["q025"], fr["q975"]
        ):
            a(f"| {name} | {fmt(mean, 3)} | {fmt(sd, 3)} | {fmt(lo, 3)}–{fmt(hi, 3)} |")
        a("")
        a(f"Month 9 versus month 2 multiplies the frailty model-1 odds by {fmt(fr['odds_month9'], 3)}.")
        a("Model 2 has no month term, so θ does not change between those months.")
        modes = fr.get("gamma_modes")
        if (
            modes
            and modes["positive_weight"] > 0.1
            and modes["negative_weight"] > 0.1
        ):
            a(
                f"The marginal posterior of γ is bimodal: weight {fmt(modes['negative_weight'], 2)} "
                f"near {fmt(modes['negative_mean'], 2)}, and weight {fmt(modes['positive_weight'], 2)} "
                f"near {fmt(modes['positive_mean'], 2)}. The stored γ is the posterior mean, between the modes."
            )
        a("Each path draws one index into that user's 1000 posterior draws of `u_p`, from the activity stream,")
        a("then uses the posterior-mean coefficients.")
        a("")
        a("| Quantity | Data | Pooled paths | Frailty paths |")
        a("|---|---:|---:|---:|")
        a(f"| Mean gaps per person | {fmt(data['mean_gaps'], 2)} | {fmt(sim['mean_gaps'], 2)} | {fmt(fr84['mean_gaps'], 2)} |")
        a(f"| Proportion with 0 gaps | {fmt(data['p0'], 3)} | {fmt(sim['p0'], 3)} | {fmt(fr84['p0'], 3)} |")
        a(f"| Proportion with 1 gap | {fmt(data['p1'], 3)} | {fmt(sim['p1'], 3)} | {fmt(fr84['p1'], 3)} |")
        a(f"| Proportion with 2 gaps | {fmt(data['p2'], 3)} | {fmt(sim['p2'], 3)} | {fmt(fr84['p2'], 3)} |")
        a(f"| Proportion with 3 or more | {fmt(data['p3plus'], 3)} | {fmt(sim['p3plus'], 3)} | {fmt(fr84['p3plus'], 3)} |")
        a(f"| Ended-gap length, median | {fmt(data['ended_p50'], 1)} | {fmt(sim['ended_p50'], 1)} | {fmt(fr84['ended_p50'], 1)} |")
        a(f"| Ended-gap length, 75th | {fmt(data['ended_p75'], 1)} | {fmt(sim['ended_p75'], 1)} | {fmt(fr84['ended_p75'], 1)} |")
        a(f"| Ended-gap length, 90th | {fmt(data['ended_p90'], 1)} | {fmt(sim['ended_p90'], 1)} | {fmt(fr84['ended_p90'], 1)} |")
        a(f"| Kaplan–Meier 90th percentile | {data['km_p90'] if data['km_p90'] is not None else '—'} | {sim['km_p90'] if sim['km_p90'] is not None else '—'} | {fr84['km_p90'] if fr84['km_p90'] is not None else '—'} |")
        a(f"| Fraction of inactive days | {fmt(data['inactive_fraction'], 3)} | {fmt(sim['inactive_fraction'], 3)} | {fmt(fr84['inactive_fraction'], 3)} |")
        a(f"| Terminal gap of 14 days or more | {fmt(data['terminal14'], 3)} | {fmt(sim['terminal14'], 3)} | {fmt(fr84['terminal14'], 3)} |")
        a(f"| Permanently gone by day 84 | — | {fmt(sim['permanent'], 3)} | {fmt(fr84['permanent'], 3)} |")
        a(f"| Permanently gone by day 252 | — | {fmt(long['permanent'], 3)} | {fmt(fr252['permanent'], 3)} |")
        a("")
    a("")
    a("## Parameters written for the simulator")
    a("")
    a("`env_para_vanilla/generator_calibration.json`, key `activity_process`. Existing `dropout_*` keys are unchanged.")
    a("Model 1 coefficients are intercept, month, act7, prior_gap. Model 2 stores c0, c1, μ, α.")
    a("The shared-frailty block stores posterior-mean fixed effects, γ, and 1000 thinned")
    a("posterior draws of `u` for each modelled user. That block is selectable and unused.")
    a("")
    a("The local `06_fit_pu.py` run on 2026-10-19 (`python 06_fit_pu.py --chains 4 --cores 4 --tune 2000 --draws 3000 --target-accept 0.95`, finished 21:12) predates the active-day filter and is not used. `env_para_vanilla/` was restored to commit `65a9fed`, plus the `activity_process` block.")
    a("")
    lines.extend(simulator_check_lines(data, sim, long))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def data_summary(panel: pd.DataFrame, gaps: pd.DataFrame) -> dict:
    counts = gaps.groupby("uid").size()
    window = panel.loc[panel["in_window"]] if "in_window" in panel.columns else panel
    # users with no gap contribute 0
    users = window["ParticipantIdentifier"].unique()
    all_counts = counts.reindex(users, fill_value=0)
    ended = gaps.loc[gaps["delta"].eq(1), "length"].to_numpy(dtype=float)
    time = gaps["length"].to_numpy(dtype=float)
    event = gaps["delta"].to_numpy(dtype=int)
    inactive = 1.0 - float(window["active"].mean())
    terminal14 = float(
        (gaps["delta"].eq(0) & gaps["length"].ge(14)).sum()
        / window["ParticipantIdentifier"].nunique()
    )
    return {
        "mean_gaps": float(all_counts.mean()),
        "p0": float(np.mean(all_counts == 0)),
        "p1": float(np.mean(all_counts == 1)),
        "p2": float(np.mean(all_counts == 2)),
        "p3plus": float(np.mean(all_counts >= 3)),
        "ended_p50": float(np.quantile(ended, 0.5)),
        "ended_p75": float(np.quantile(ended, 0.75)),
        "ended_p90": float(np.quantile(ended, 0.9)),
        "km_p50": km_quantile(time, event, 0.5),
        "km_p90": km_quantile(time, event, 0.9),
        "inactive_fraction": inactive,
        "terminal14": terminal14,
    }


def fit_shared_frailty(frame: pd.DataFrame, gaps: pd.DataFrame, model1: dict, model2: dict) -> dict:
    """Shared-frailty joint model with the tight priors. Posterior means are plug-in values."""
    import os
    import sys

    sys.path = [p for p in sys.path if p not in ("", "/tmp", "/private/tmp")]
    os.environ.setdefault("PYTENSOR_FLAGS", "compiledir=/tmp/pytensor_compile_activity")
    import pymc as pm
    import pytensor.tensor as pt

    uids = list(dict.fromkeys(list(frame["uid"]) + list(gaps["uid"])))
    uid_index = {uid: i for i, uid in enumerate(uids)}
    month = frame["month"].to_numpy(dtype=float)
    act7 = frame["act7"].to_numpy(dtype=float)
    prior1 = frame["prior_gap"].to_numpy(dtype=float)
    y = frame["y"].to_numpy(dtype=int)
    i1 = frame["uid"].map(uid_index).to_numpy(dtype=int)
    g_prior = gaps["prior_gap"].to_numpy(dtype=float)
    length = gaps["length"].to_numpy(dtype=float)
    delta = gaps["delta"].to_numpy(dtype=float)
    i2 = gaps["uid"].map(uid_index).to_numpy(dtype=int)
    n_users = len(uids)

    def gap_loglik(theta, mu, alpha):
        def weibull_cdf(d):
            base = pt.maximum(d, 1.0)
            on = pt.cast(pt.gt(d, 0.0), "float64")
            return on * (1.0 - pt.exp(-pt.pow(base / mu, alpha)))

        f_now = weibull_cdf(length)
        f_prev = weibull_cdf(length - 1.0)
        step = theta * (f_now - f_prev)
        log_one_minus = pt.switch(
            pt.gt(step, math.log(2.0)),
            pt.log1p(-pt.exp(-step)),
            pt.log(-pt.expm1(-step)),
        )
        ended = -theta * f_prev + log_one_minus
        opened = -theta * f_now
        return pt.sum(pt.switch(pt.eq(delta, 0.0), opened, ended))

    a_hat = np.asarray(model1["coef"], dtype=float)
    c_hat = np.array([model2["c0"], model2["c1"]], dtype=float)
    eta0 = a_hat[0] + a_hat[1] * month + a_hat[2] * act7 + a_hat[3] * prior1
    logit_ll = float(np.sum(y * eta0 - np.logaddexp(0.0, eta0)))
    if abs(logit_ll - float(model1["llf"])) > 1e-4:
        raise SystemExit(
            f"Frailty model-1 log likelihood {logit_ll} disagrees with the pooled fit {model1['llf']}."
        )

    theta0 = np.exp(c_hat[0] + c_hat[1] * g_prior)
    gap_value = float(
        gap_loglik(
            pt.as_tensor_variable(np.asarray(theta0, dtype=float)),
            pt.as_tensor_variable(float(model2["mu"])),
            pt.as_tensor_variable(float(model2["alpha"])),
        ).eval()
    )
    expected_gap = -model2_nll(
        np.array(
            [c_hat[0], c_hat[1], math.log(model2["mu"]), math.log(model2["alpha"])],
            dtype=float,
        ),
        g_prior,
        length,
        delta,
    )
    if abs(gap_value - expected_gap) > 1e-4 or abs(expected_gap - float(model2["loglik"])) > 1e-4:
        raise SystemExit(
            "Frailty gap log likelihood "
            f"{gap_value} disagrees with the pooled cure fit {model2['loglik']} ({expected_gap})."
        )
    print(
        f"Frailty log likelihood at the pooled MLE with u=0: {logit_ll + expected_gap:.4f}",
        flush=True,
    )
    p_center = np.asarray(g_prior - float(np.mean(g_prior)), dtype=float)
    p_ss = float(np.dot(p_center, p_center))
    if p_ss <= 0.0:
        raise SystemExit("prior_gap does not vary across gaps; cannot residualize the frailty.")
    p_mean = float(np.mean(g_prior))

    with pm.Model() as model:
        a = pm.Normal("a", 0.0, 5.0, shape=4)
        c = pm.Normal("c", 0.0, 5.0, shape=2)
        log_mu = pm.Normal("log_mu", math.log(5.0), 1.0)
        log_alpha = pm.Normal("log_alpha", 0.0, 1.0)
        tau = pm.HalfNormal("tau", 1.0)
        gamma = pm.Normal("gamma", 0.0, 1.0)
        z = pm.Normal("z", 0.0, 1.0, shape=n_users)
        p_center_t = pt.as_tensor_variable(p_center)
        # User-mean centered, so a shared shift in u is not aliased with the logit intercept.
        u = tau * (z - pt.mean(z))
        eta = a[0] + a[1] * month + a[2] * act7 + a[3] * prior1 + u[i1]
        pm.Bernoulli("y", logit_p=eta, observed=y)
        mu = pt.exp(log_mu)
        alpha = pt.exp(log_alpha)
        # Residualize u on the gap rows. c0 and c1 are mapped back after sampling
        # to the simulator equation log theta = c0 + c1*prior_gap - gamma*u.
        u_g = u[i2]
        u_c = u_g - pt.mean(u_g)
        beta_u = pt.sum(p_center_t * u_c) / p_ss
        u_resid = u_c - beta_u * p_center_t
        theta = pt.exp(c[0] + c[1] * g_prior - gamma * u_resid)
        pm.Potential("gaps", gap_loglik(theta, mu, alpha))
        init_point = model.initial_point()
        init_logp = np.asarray(model.compile_logp()(init_point), dtype=float)
        init_grad = np.asarray(model.compile_dlogp()(init_point), dtype=float)
        if not np.isfinite(init_logp).all() or not np.isfinite(init_grad).all():
            raise SystemExit(
                "Shared-frailty log density or its gradient is not finite at the initial point."
            )
        print(
            f"Shared-frailty initial logp {float(init_logp):.3f}; gradient is finite.",
            flush=True,
        )
        print("Sampling shared frailty (4 chains, tune 5000, draws 30000)", flush=True)
        idata = pm.sample(
            draws=30000,
            tune=5000,
            chains=4,
            cores=4,
            target_accept=0.99,
            random_seed=2027,
            progressbar=True,
            init="adapt_full",
            initvals={
                "a": a_hat,
                "c": c_hat,
                "log_mu": float(math.log(model2["mu"])),
                "log_alpha": float(math.log(model2["alpha"])),
                "tau": 0.5,
                "gamma": 0.0,
                "z": np.zeros(n_users),
            },
            idata_kwargs={"log_likelihood": False},
        )
    import arviz as az

    draws = idata.posterior
    names = [
        "a0", "a1_month", "a2_act7", "a3_prior_gap",
        "c0", "c1_prior_gap",
        "mu", "alpha", "tau", "gamma",
    ]
    def _samples(name: str) -> np.ndarray:
        stacked = draws[name].stack(sample=("chain", "draw"))
        values = np.asarray(stacked)
        sample_axis = stacked.dims.index("sample")
        return np.moveaxis(values, sample_axis, 0)

    a_s = _samples("a")
    c_s = _samples("c")
    mu_s = np.exp(_samples("log_mu"))
    alpha_s = np.exp(_samples("log_alpha"))
    tau_s = _samples("tau")
    gamma_s = _samples("gamma")
    z_s = _samples("z")
    u_s = tau_s.reshape(-1, 1) * (z_s - z_s.mean(axis=1, keepdims=True))
    u_gap = u_s[:, i2]
    u_gap_c = u_gap - u_gap.mean(axis=1, keepdims=True)
    beta_s = (u_gap_c * p_center).sum(axis=1) / p_ss
    c0_sim = c_s[:, 0] + gamma_s * u_gap.mean(axis=1) - gamma_s * beta_s * p_mean
    c1_sim = c_s[:, 1] + gamma_s * beta_s
    linear_resid = (
        c_s[:, 0:1]
        + c_s[:, 1:2] * g_prior
        - gamma_s[:, None] * (u_gap_c - beta_s[:, None] * p_center)
    )
    linear_sim = c0_sim[:, None] + c1_sim[:, None] * g_prior - gamma_s[:, None] * u_gap
    if float(np.max(np.abs(linear_resid - linear_sim))) > 1e-6:
        raise SystemExit("Return-model reparameterization does not match the simulator equation.")
    c_s = np.column_stack([c0_sim, c1_sim])
    n_chains = int(idata.posterior.sizes["chain"])
    n_draws = int(idata.posterior.sizes["draw"])
    chain_of_sample = np.asarray(
        draws["gamma"].stack(sample=("chain", "draw")).coords["chain"]
    )
    if not np.array_equal(chain_of_sample.reshape(n_chains, n_draws)[:, 0], np.arange(n_chains)):
        raise SystemExit("Posterior stack is not chain-major; R-hat would use shuffled draws.")

    def _by_chain(flat: np.ndarray) -> np.ndarray:
        return np.asarray(flat, dtype=float).reshape(n_chains, n_draws, *np.shape(flat)[1:])

    summary = az.summary(
        az.from_dict(
            posterior={
                "a0": _by_chain(a_s[:, 0]),
                "a1_month": _by_chain(a_s[:, 1]),
                "a2_act7": _by_chain(a_s[:, 2]),
                "a3_prior_gap": _by_chain(a_s[:, 3]),
                "c0": _by_chain(c0_sim),
                "c1_prior_gap": _by_chain(c1_sim),
                "mu": _by_chain(mu_s),
                "alpha": _by_chain(alpha_s),
                "tau": _by_chain(tau_s),
                "gamma": _by_chain(gamma_s),
            }
        )
    )
    pick = np.linspace(0, u_s.shape[0], 1000, endpoint=False).astype(int)
    thinned = u_s[pick]
    if thinned.shape != (1000, len(uids)):
        raise SystemExit(f"Thinned frailty draws have shape {thinned.shape}, expected (1000, {len(uids)}).")
    u_by_uid = {
        str(int(uid)): [round(float(v), 6) for v in thinned[:, i]]
        for i, uid in enumerate(uids)
    }
    columns = np.column_stack([a_s, c_s, mu_s, alpha_s, tau_s, gamma_s])
    divergences = int(idata.sample_stats["diverging"].sum())
    max_rhat = float(summary["r_hat"].max())
    min_ess = float(summary["ess_bulk"].min())
    worst = summary.sort_values("r_hat", ascending=False).head(8)
    print(worst[["mean", "sd", "r_hat", "ess_bulk"]].to_string(), flush=True)
    if max_rhat > 1.01 or min_ess < 400:
        idata.to_netcdf("/tmp/frailty_idata.nc")
        print("per-chain gamma means", gamma_s.reshape(n_chains, n_draws).mean(axis=1), flush=True)
        print("per-chain c means", c_s.reshape(n_chains, n_draws, 2).mean(axis=1), flush=True)
        raise SystemExit(
            f"Shared frailty did not mix: max R-hat {max_rhat:.3f}, min bulk ESS {min_ess:.0f}."
        )
    mean = columns.mean(axis=0)
    positive = gamma_s > 0.5
    negative = gamma_s < -0.5
    gamma_modes = {
        "positive_weight": float(np.mean(gamma_s > 0.0)),
        "positive_mean": float(gamma_s[positive].mean()) if np.any(positive) else None,
        "negative_weight": float(np.mean(gamma_s < 0.0)),
        "negative_mean": float(gamma_s[negative].mean()) if np.any(negative) else None,
    }
    return {
        "seed": 2027,
        "priors": (
            "coefficients N(0, 5^2); log mu N(log 5, 1); log alpha N(0, 1); "
            "tau HalfNormal(1); gamma N(0, 1)"
        ),
        "sampler": (
            "NUTS, 4 chains, tune 5000, draws 30000, target_accept 0.99, dense mass matrix, started at the pooled MLE. "
            "u on the gap rows is residualized on the intercept and prior_gap while sampling; "
            "c0 and c1 are mapped back to log theta = c0 + c1*prior_gap - gamma*u."
        ),
        "names": names,
        "mean": [float(v) for v in mean],
        "sd": [float(v) for v in columns.std(axis=0, ddof=1)],
        "q025": [float(v) for v in np.quantile(columns, 0.025, axis=0)],
        "q975": [float(v) for v in np.quantile(columns, 0.975, axis=0)],
        "divergences": divergences,
        "max_rhat": max_rhat,
        "min_ess": min_ess,
        "gamma_modes": gamma_modes,
        "odds_month9": float(math.exp(7.0 * mean[1])),
        "M": 1000,
        "gamma": float(mean[9]),
        "tau": float(mean[8]),
        "model1_coefficients": [float(v) for v in mean[:4]],
        "model2": {
            "c0": float(mean[4]),
            "c1": float(mean[5]),
            "mu": float(mean[6]),
            "alpha": float(mean[7]),
        },
        "u_by_uid": u_by_uid,
    }


def write_calibration(model1: dict, model2: dict, frailty: dict, validation: dict) -> None:
    payload = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))
    kept = {key: payload[key] for key in payload if key.startswith("dropout_")}
    meta = pd.read_csv(META_PATH)
    n_modelled = int(meta["in_modelled"].astype(bool).sum())
    payload["activity_process"] = {
        "source_file": "ADAPR_COMBINED_DIR/participant_dropout_daily.csv",
        "source_date": SOURCE_DATE,
        "cohort": f"{n_modelled} modelled participants, study days 1-84",
        "covariates": {
            "month": (
                "protocol month m = ceil(week/4), covariate month_c = m - 2. "
                "Weeks 1-4, 5-8, and 9-12 are months 1, 2, and 3. "
                "A 36-week episode uses months 1-9 on this same scale, with no cap. "
                "Week 0 (a Sunday starter's first day) is month 1. "
                "Month enters model 1 only."
            ),
            "act7": (
                "fraction of the previous 7 days active; days before study day 1 count as active; "
                "the observed week-0 Sunday is used as history; unobserved days are skipped"
            ),
            "prior_gap": "1 on and after the day a completed gap ends (activity resumes); else 0",
            "activity": (
                "page view, wear, or a daily end-of-day or weekly check-in submission "
                "on the participant's local calendar day"
            ),
        },
        "model1": {
            "coefficient_names": list(MODEL1_NAMES),
            "coefficients": [float(v) for v in model1["coef"]],
            "cluster_robust_se": [float(v) for v in model1["se"]],
            "lambda_bar": float(model1["lambda_bar"]),
        },
        "model2": {
            "c0": model2["c0"],
            "c1": model2["c1"],
            "mu": model2["mu"],
            "alpha": model2["alpha"],
            "hessian_se": model2["se"],
        },
        "frailty": {
            "note": (
                "Experimental and unused. gamma is bimodal. Model 1 adds u_p to the logit; "
                "model 2 subtracts gamma * u_p from log theta. Fixed effects and gamma are "
                "posterior means. u_by_uid holds 1000 thinned posterior draws per modelled user. "
                "An episode draws one index from SeedSequence([seed, uid, 771337]) "
                "and then continues that stream for the path. "
                "The default activity_process_mode is 'pooled', which uses the model1 and model2 blocks."
            ),
            "seed": frailty["seed"],
            "priors": frailty["priors"],
            "sampler": frailty["sampler"],
            "M": frailty["M"],
            "gamma": frailty["gamma"],
            "tau": frailty["tau"],
            "names": frailty["names"],
            "posterior_mean": frailty["mean"],
            "posterior_sd": frailty["sd"],
            "q025": frailty["q025"],
            "q975": frailty["q975"],
            "divergences": frailty["divergences"],
            "max_rhat": frailty["max_rhat"],
            "min_ess": frailty["min_ess"],
            "gamma_modes": frailty["gamma_modes"],
            "model1_coefficients": frailty["model1_coefficients"],
            "model2": frailty["model2"],
            "odds_multiplier_month9_vs_month2": frailty["odds_month9"],
            "u_by_uid": frailty["u_by_uid"],
        },
        "validation_2000": validation,
    }
    CALIBRATION_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    written = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))
    for key, value in kept.items():
        if written.get(key) != value:
            raise SystemExit(f"dropout key {key} changed while writing activity_process.")


def frailty_from_calibration() -> dict:
    """Report fields for a frailty block already written into the calibration file."""
    block = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))["activity_process"]["frailty"]
    loaded = dict(block)
    loaded["mean"] = list(block["posterior_mean"])
    loaded["sd"] = list(block["posterior_sd"])
    loaded["odds_month9"] = float(block["odds_multiplier_month9_vs_month2"])
    return loaded


def activity_process_at_head() -> dict:
    """The committed activity-process block, used when the working file is already the new fit."""
    import subprocess

    raw = subprocess.check_output(
        ["git", "show", "HEAD:env_para_vanilla/generator_calibration.json"],
        cwd=ROOT,
    )
    block = json.loads(raw)["activity_process"]
    if "c2" not in block["model2"]:
        raise SystemExit(
            "HEAD calibration has no c2, so it is not the previous month-in-model-2 fit."
        )
    return block


def main() -> None:
    print(f"params dir: {resolve_params_dir()}", flush=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default=None,
        help="Directory for figures and report.md "
        "(default: docs/activity_process_<YYYYMMDD>/, created if absent)",
    )
    parser.add_argument(
        "--reuse-posterior",
        action="store_true",
        help="Skip NUTS and rewrite the report from the frailty block already stored.",
    )
    args = parser.parse_args()
    configure_paths(args.out)
    panel, _meta = load_panel()
    gaps = build_gaps(panel)
    cohort = assert_cohort(panel, gaps)
    frame = model1_frame(panel, gaps)
    n_risk = int(len(frame))
    if not 3000 <= n_risk <= 4200:
        raise SystemExit(f"Active person-days {n_risk} are outside 3000–4200. Stop.")

    full = fit_logit(frame, ["month", "act7", "prior_gap"])
    null = fit_logit(frame, [])
    fitted = np.asarray(full.predict(), dtype=float)
    by_week, by_decile = calibration_tables(frame, fitted)
    wald = full.wald_test(np.eye(4)[1:], scalar=True)
    first = frame["prior_gap"].eq(0)
    references = []
    for month, act7, prior in ((0.0, 1.0, 0), (0.0, 1.0, 1), (0.0, 0.5, 1), (1.0, 1.0, 0), (-1.0, 1.0, 0)):
        eta = float(full.params[0] + full.params[1] * month + full.params[2] * act7 + full.params[3] * prior)
        references.append({"month": month, "act7": act7, "prior_gap": prior, "lambda": float(expit(np.array([eta]))[0])})
    or_ci = np.exp(np.asarray(full.conf_int(), dtype=float))
    model1 = {
        "coef": [float(v) for v in full.params],
        "se": [float(v) for v in full.bse],
        "or": [float(v) for v in np.exp(full.params)],
        "or_low": [float(v) for v in or_ci[:, 0]],
        "or_high": [float(v) for v in or_ci[:, 1]],
        "p": [float(v) for v in full.pvalues],
        "llf": float(full.llf),
        "llf_null": float(null.llf),
        "lambda_bar": float(expit(np.array([null.params[0]]))[0]),
        "lambda_bar_se": float(expit(np.array([null.params[0]]))[0] * (1 - expit(np.array([null.params[0]]))[0]) * null.bse[0]),
        "intercept_only": float(null.params[0]),
        "intercept_only_se": float(null.bse[0]),
        "wald_stat": float(np.asarray(wald.statistic).reshape(-1)[0]),
        "wald_df": int(getattr(wald, "df_num", 3) or 3),
        "wald_p": float(np.asarray(wald.pvalue).reshape(-1)[0]),
        "first_gap_rate": float(frame.loc[first, "y"].mean()),
        "first_gap_events": int(frame.loc[first, "y"].sum()),
        "first_gap_days": int(first.sum()),
        "references": references,
    }
    model2 = fit_model2(gaps)
    if args.reuse_posterior:
        frailty = frailty_from_calibration()
    else:
        frailty = fit_shared_frailty(frame, gaps, model1, model2)
    sanity_check_simulator(model2)
    derived = derived_quantities(gaps, model2, model1["lambda_bar"])
    long_gap = gaps["length"].eq(56) & gaps["delta"].eq(1) & gaps["uid"].eq("112")
    if int(long_gap.sum()) != 1:
        raise SystemExit("Expected exactly one ended 56-day gap for user 112.")
    mz = maller_zhou(gaps["length"].to_numpy(dtype=float), gaps["delta"].to_numpy(dtype=int))
    held = gaps.loc[~long_gap]
    mz0 = maller_zhou(held["length"].to_numpy(dtype=float), held["delta"].to_numpy(dtype=int))
    data = data_summary(panel, gaps)
    meta = pd.read_csv(META_PATH)
    uids = sorted(
        int(u) for u in meta.loc[meta["in_modelled"].astype(bool), "ParticipantIdentifier"]
    )
    record = {
        "model1": {"coefficients": model1["coef"]},
        "model2": {key: model2[key] for key in ("c0", "c1", "mu", "alpha")},
        "frailty": {
            "model1_coefficients": frailty["model1_coefficients"],
            "model2": frailty["model2"],
            "gamma": frailty["gamma"],
            "u_by_uid": frailty["u_by_uid"],
        },
    }

    def run_cohort(mode, n_paths, n_days, n_weeks, base_seed):
        return summarize_paths(
            simulate_cohort_paths(record, mode, n_paths, n_days, n_weeks, uids, base_seed)
        )

    print("Simulating pooled and frailty paths", flush=True)
    sim84 = run_cohort("pooled", 10000, 84, N_WEEKS_FIT, 20261020)
    sim252 = run_cohort("pooled", 10000, 252, N_WEEKS_LONG, 20261021)
    frailty_sim84 = run_cohort("frailty", 10000, 84, N_WEEKS_FIT, 20261024)
    frailty_sim252 = run_cohort("frailty", 10000, 252, N_WEEKS_LONG, 20261025)
    pooled_2000 = run_cohort("pooled", 2000, 84, N_WEEKS_FIT, 20261022)
    pooled_252 = run_cohort("pooled", 2000, 252, N_WEEKS_LONG, 20261023)
    frailty_2000 = run_cohort("frailty", 2000, 84, N_WEEKS_FIT, 20261022)
    frailty_252 = run_cohort("frailty", 2000, 252, N_WEEKS_LONG, 20261023)

    def pack(summary, permanent_252, seed84, seed252):
        kept_keys = (
            "n_paths", "mean_gaps", "p0", "p1", "p2", "p3plus",
            "ended_p50", "ended_p75", "ended_p90", "km_p90",
            "inactive_fraction", "terminal14", "permanent",
        )
        out = {key: summary[key] for key in kept_keys}
        out["permanent_252"] = float(permanent_252)
        out["base_seed_84"] = int(seed84)
        out["base_seed_252"] = int(seed252)
        out["n_days_84"] = 84
        out["n_days_252"] = 252
        return out

    validation = {
        "note": (
            "2000 episode paths. Modelled users cycle; the seed increases after each full pass. "
            "Frailty draws u from the activity stream. Tests recompute this table."
        ),
        "uids": uids,
        "pooled": pack(pooled_2000, pooled_252["permanent"], 20261022, 20261023),
        "frailty": pack(frailty_2000, frailty_252["permanent"], 20261022, 20261023),
    }
    save_figures(gaps, model2, by_week, by_decile, derived)
    if args.reuse_posterior:
        previous = activity_process_at_head()
    else:
        previous = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))["activity_process"]
        write_calibration(model1, model2, frailty, validation)
    write_report(
        {
            "n_risk": n_risk,
            "n_events": int(frame["y"].sum()),
            "model1": model1,
            "model2": model2,
            "by_week": by_week,
            "by_decile": by_decile,
            "derived": derived,
            "data_summary": data,
            "sim84": sim84,
            "sim252": sim252,
            "frailty": frailty,
            "frailty_sim84": frailty_sim84,
            "frailty_sim252": frailty_sim252,
            "mz": mz,
            "mz_without_112": mz0,
            "cohort": cohort,
            "previous": previous,
        }
    )
    print(
        f"risk={n_risk} events={int(frame['y'].sum())} "
        f"c=({model2['c0']:.3f},{model2['c1']:.3f}) "
        f"mu={model2['mu']:.3f} alpha={model2['alpha']:.3f} "
        f"pooled perm252={sim252['permanent']:.3f} "
        f"frailty perm252={frailty_sim252['permanent']:.3f}"
    )
    print(f"Wrote {OUT_DIR / 'report.md'}")


if __name__ == "__main__":
    main()
