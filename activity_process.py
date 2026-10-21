"""Exogenous daily activity path for the simulator.

Monday-start episodes. Day 1 is active. Week is ``ceil(day/7)`` passed through
``study_week_norm(..., n_weeks=horizon)``. The path is drawn once, from its own
RNG, and the day loop only reads it.
"""

from __future__ import annotations

import math

import numpy as np

from vani_env import study_week_norm


def week_number(day: int) -> int:
    """Protocol week of a Monday-start episode. Day 1 is Monday of week 1."""
    return int(math.ceil(int(day) / 7.0))


def week_norm_of(day: int, n_weeks: int) -> float:
    if int(day) <= 0:
        return 0.0
    return float(study_week_norm(week_number(day), n_weeks=n_weeks))


DROPOUT_RNG_SALT = 771337


def dropout_seed_sequence(seed, uid) -> np.random.SeedSequence:
    """Activity-path RNG. The main episode seed does not include the participant."""
    dropout_seed = 0 if seed is None else int(seed)
    return np.random.SeedSequence([dropout_seed, int(uid), DROPOUT_RNG_SALT])


def expit(eta: np.ndarray) -> np.ndarray:
    eta = np.asarray(eta, dtype=float)
    out = np.empty(eta.shape, dtype=float)
    pos = eta >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-eta[pos]))
    ez = np.exp(eta[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def weibull_F(d: np.ndarray, mu: float, alpha: float) -> np.ndarray:
    d = np.asarray(d, dtype=float)
    out = np.zeros(d.shape, dtype=float)
    pos = d > 0
    out[pos] = 1.0 - np.exp(-np.power(d[pos] / mu, alpha))
    return out


def survival_uncured(d: np.ndarray, theta: np.ndarray, mu: float, alpha: float) -> np.ndarray:
    """S_u(d) = (exp(-theta F(d)) - exp(-theta)) / (1 - exp(-theta))."""
    pi = np.exp(-theta)
    numer = np.exp(-theta * weibull_F(d, mu, alpha)) - pi
    denom = np.maximum(1.0 - pi, 1e-15)
    return np.clip(numer / denom, 0.0, 1.0)


def hazard_uncured(k: np.ndarray, theta: np.ndarray, mu: float, alpha: float) -> np.ndarray:
    """h_u(k) = 1 - S_u(k) / S_u(k-1)."""
    k = np.asarray(k, dtype=float)
    theta = np.asarray(theta, dtype=float)
    s_prev = np.ones(k.shape, dtype=float)
    prev = k - 1.0
    has_prev = prev > 0
    if np.any(has_prev):
        s_prev[has_prev] = survival_uncured(prev[has_prev], theta[has_prev], mu, alpha)
    s_cur = survival_uncured(k, theta, mu, alpha)
    return 1.0 - s_cur / np.maximum(s_prev, 1e-15)


def coefficients_from_record(record: dict) -> tuple[np.ndarray, dict]:
    """Model 1 coefficients and model 2 parameters from calibration JSON."""
    coef = np.asarray(record["model1"]["coefficients"], dtype=float)
    model2 = record["model2"]
    fit = {
        "c0": float(model2["c0"]),
        "c1": float(model2["c1"]),
        "c2": float(model2["c2"]),
        "mu": float(model2["mu"]),
        "alpha": float(model2["alpha"]),
    }
    return coef, fit


def _empty_result(n_paths: int, n_state: int) -> dict:
    return {
        "status": np.zeros((n_paths, n_state), dtype=np.int8),
        "b": np.full((n_paths, n_state), np.nan),
        "dropout_day": np.full(n_paths, -1, dtype=np.int16),
        "n_gaps": np.zeros(n_paths, dtype=np.int16),
        "inactive_days": np.zeros(n_paths, dtype=np.int16),
        "ended": np.asarray([], dtype=int),
        "censored": np.asarray([], dtype=int),
        "terminal14": np.zeros(n_paths, dtype=bool),
        "n_days": 0,
        "n_paths": n_paths,
    }


def simulate_activity_paths(
    n_paths: int,
    n_days: int,
    coef: np.ndarray,
    fit: dict,
    n_weeks: int,
    rng: np.random.Generator,
    *,
    extra_days: int = 2,
    lambda_p: float | None = None,
    pi: float | None = None,
) -> dict:
    """Draw activity paths.

    ``status`` is 0 active, 1 temporary inactive, 2 permanent (B = 1).
    Columns are 0-based study days. ``extra_days`` extends the path past the
    episode so the last Sunday's Monday and Tuesday can be read.
    Gap summaries (ended, censored, inactive days, dropout) use ``n_days`` only.
    ``lambda_p = 0`` is all active and consumes no random numbers.
    ``lambda_p = 1`` and ``pi = 1`` drops every path on day 2.
    """
    n_paths = int(n_paths)
    n_days = int(n_days)
    n_state = n_days + int(extra_days)
    if lambda_p is not None and float(lambda_p) <= 0.0:
        out = _empty_result(n_paths, n_state)
        out["n_days"] = n_days
        return out
    if (
        lambda_p is not None
        and float(lambda_p) >= 1.0
        and pi is not None
        and float(pi) >= 1.0
    ):
        out = _empty_result(n_paths, n_state)
        out["status"][:, 1:] = 2
        out["b"][:, 1] = 1.0
        out["dropout_day"][:] = 2
        out["n_gaps"][:] = 1
        out["inactive_days"][:] = max(n_days - 1, 0)
        out["terminal14"][:] = n_days - 1 >= 14
        out["censored"] = np.full(n_paths, max(n_days - 1, 0), dtype=int)
        out["n_days"] = n_days
        return out

    coef = np.asarray(coef, dtype=float)
    week = np.array(
        [week_norm_of(day, n_weeks) for day in range(n_state + 2)], dtype=float
    )
    status = np.zeros(n_paths, dtype=np.int8)
    k = np.zeros(n_paths, dtype=np.int16)
    prior = np.zeros(n_paths, dtype=float)
    hist = np.ones((n_paths, 7), dtype=float)
    gap_start = np.zeros(n_paths, dtype=np.int16)
    theta_gap = np.ones(n_paths, dtype=float)
    status_hist = np.zeros((n_paths, n_state), dtype=np.int8)
    b_hist = np.full((n_paths, n_state), np.nan)
    n_gaps = np.zeros(n_paths, dtype=np.int16)
    inactive_days = np.zeros(n_paths, dtype=np.int16)
    dropout_day = np.full(n_paths, -1, dtype=np.int16)
    ended: list[int] = []
    censored: list[int] = []
    terminal14 = np.zeros(n_paths, dtype=bool)
    mu, alpha = float(fit["mu"]), float(fit["alpha"])

    for day in range(1, n_state + 1):
        status_hist[:, day - 1] = status
        inactive_now = status != 0
        if day <= n_days:
            inactive_days[inactive_now] += 1
            if day == n_days:
                lengths = np.zeros(n_paths, dtype=int)
                lengths[inactive_now] = n_days - gap_start[inactive_now] + 1
                censored.extend(int(v) for v in lengths[inactive_now])
                terminal14 = inactive_now & (lengths >= 14)
        if day == n_state:
            break

        next_status = status.copy()
        next_k = k.copy()
        next_prior = prior.copy()
        active_idx = np.flatnonzero(status == 0)
        if len(active_idx):
            act7 = hist[active_idx].mean(axis=1)
            if lambda_p is None:
                eta = (
                    coef[0]
                    + coef[1] * week[day]
                    + coef[2] * act7
                    + coef[3] * prior[active_idx]
                )
                p_start = expit(eta)
            else:
                p_start = np.full(len(active_idx), float(lambda_p))
            starters = active_idx[rng.random(len(active_idx)) < p_start]
            if len(starters):
                theta = np.exp(
                    fit["c0"]
                    + fit["c1"] * prior[starters]
                    + fit["c2"] * week[day + 1]
                )
                p_perm = np.exp(-theta) if pi is None else np.full(len(starters), float(pi))
                perm = rng.random(len(starters)) < p_perm
                perm_idx = starters[perm]
                temp_idx = starters[~perm]
                next_status[perm_idx] = 2
                next_status[temp_idx] = 1
                next_k[temp_idx] = 1
                theta_gap[temp_idx] = theta[~perm]
                gap_start[starters] = day + 1
                start_col = day  # 0-based index of day+1
                if start_col < n_state:
                    b_hist[temp_idx, start_col] = 0.0
                    b_hist[perm_idx, start_col] = 1.0
                inside = day + 1 <= n_days
                if inside:
                    n_gaps[starters] += 1
                    dropout_day[perm_idx] = day + 1
        inactive_idx = np.flatnonzero(status == 1)
        if len(inactive_idx):
            hu = hazard_uncured(
                k[inactive_idx].astype(float), theta_gap[inactive_idx], mu, alpha
            )
            returning = rng.random(len(inactive_idx)) < hu
            ret_idx = inactive_idx[returning]
            stay_idx = inactive_idx[~returning]
            if day + 1 <= n_days:
                ended.extend(int(v) for v in k[ret_idx])
            next_status[ret_idx] = 0
            next_prior[ret_idx] = 1.0
            next_k[ret_idx] = 0
            next_k[stay_idx] = k[stay_idx] + 1
        hist = np.concatenate(
            [hist[:, 1:], (status == 0).astype(float)[:, None]], axis=1
        )
        status, k, prior = next_status, next_k, next_prior

    return {
        "status": status_hist,
        "b": b_hist,
        "dropout_day": dropout_day,
        "n_gaps": n_gaps,
        "inactive_days": inactive_days,
        "ended": np.asarray(ended, dtype=int),
        "censored": np.asarray(censored, dtype=int),
        "terminal14": terminal14,
        "n_days": n_days,
        "n_paths": n_paths,
    }


def summarize_paths(sim: dict) -> dict:
    """Gap and dropout summaries comparable to the Step 1 validation table."""
    n = int(sim["n_paths"])
    gaps = np.asarray(sim["n_gaps"], dtype=float)
    ended = np.asarray(sim["ended"], dtype=float)
    cens = np.asarray(sim["censored"], dtype=float)
    n_days = int(sim["n_days"])

    def q(arr: np.ndarray, p: float):
        if arr.size == 0:
            return None
        return float(np.quantile(arr, p))

    time = np.concatenate([ended, cens]) if cens.size else ended
    event = np.concatenate(
        [np.ones(ended.size, dtype=int), np.zeros(cens.size, dtype=int)]
    ) if time.size else np.asarray([], dtype=int)
    km90 = None
    if time.size:
        order = np.argsort(time, kind="mergesort")
        time_o = time[order]
        event_o = event[order]
        surv = 1.0
        n_risk = float(time.size)
        i = 0
        while i < time_o.size:
            t = time_o[i]
            deaths = 0
            cens_n = 0
            while i < time_o.size and time_o[i] == t:
                if event_o[i] == 1:
                    deaths += 1
                else:
                    cens_n += 1
                i += 1
            if n_risk > 0 and deaths:
                surv *= 1.0 - deaths / n_risk
            n_risk -= deaths + cens_n
            if surv <= 0.1:
                km90 = float(t)
                break
    return {
        "n_paths": n,
        "mean_gaps": float(gaps.mean()),
        "p0": float(np.mean(gaps == 0)),
        "p1": float(np.mean(gaps == 1)),
        "p2": float(np.mean(gaps == 2)),
        "p3plus": float(np.mean(gaps >= 3)),
        "ended_p50": q(ended, 0.50),
        "ended_p75": q(ended, 0.75),
        "ended_p90": q(ended, 0.90),
        "km_p90": km90,
        "inactive_fraction": float(np.mean(sim["inactive_days"]) / n_days),
        "terminal14": float(np.mean(sim["terminal14"])),
        "permanent": float(np.mean(np.asarray(sim["dropout_day"]) > 0)),
    }
