"""Who is in the modelled cohort.

Script 01 has the span table, the weekly CAE panel, and the dropout table, so
it is the step that writes ``modelled_cohort.csv``. Later scripts read that
file. They do not apply the rule again.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CohortPolicy:
    """Thresholds and on/off switches for the modelled-cohort rule."""

    min_active_span_days: int = 83
    force_include_ids: tuple[str, ...] = ("379",)
    manual_override_ids: tuple[str, ...] = ()
    exclude_if_insufficient_wear: bool = False
    exclude_if_insufficient_foursc: bool = False
    exclude_if_all_weekly_surveys_missing: bool = False
    exclude_if_all_daily_surveys_missing: bool = False
    exclude_if_all_cae_missing: bool = False
    min_decision_window_wear_sum: int = 20
    min_foursc_non_nan: int = 20
    min_weekly_surveys_present: int = 1
    min_daily_surveys_present: int = 1
    min_cae_weeks_present: int = 1
    exclude_if_high_cae: bool = True
    min_weekly_cae_threshold: float = 6.0
    week1_cae_ceiling: float = 7.0


def modelled_membership(people: pd.DataFrame, policy: CohortPolicy) -> pd.DataFrame:
    """Return ``uid``, ``in_modelled``, and ``reason`` for each starter.

    A starter is modelled when the end-of-day task span covers
    ``min_active_span_days`` and no exclusion reason applies. Force-included
    ids are modelled even with a short span, and the force-include clears
    exclusion reasons before the high-CAE ceiling is considered. A manual
    override still removes someone after that clear, matching the audit.

    ``people`` needs ``ParticipantIdentifier`` and ``span_days``. Wear, survey,
    and CAE columns use the analysis-sample audit names. Missing count columns
    are zero and missing CAE values are unknown, so a disabled rule does not
    fire.
    """
    missing = {"ParticipantIdentifier", "span_days"} - set(people.columns)
    if missing:
        raise ValueError(
            "modelled_membership needs " + ", ".join(sorted(missing))
        )
    frame = people.copy()
    frame["ParticipantIdentifier"] = frame["ParticipantIdentifier"].astype(str)
    frame["span_days"] = pd.to_numeric(frame["span_days"], errors="coerce")
    count_defaults = {
        "decision_window_wear_sum": 0,
        "hourly_non_nan": 0,
        "weekly_present_sum": 0,
        "daily_present_sum": 0,
        "cae_weeks_present": 0,
    }
    for column, fill in count_defaults.items():
        if column not in frame.columns:
            frame[column] = fill
        else:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(fill)
    for column in ("min_weekly_cae", "week1_cae"):
        if column not in frame.columns:
            frame[column] = np.nan
        else:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    force = {str(pid) for pid in policy.force_include_ids}
    override = {str(pid) for pid in policy.manual_override_ids}
    rows: list[dict[str, object]] = []
    for rec in frame.itertuples(index=False):
        pid = str(rec.ParticipantIdentifier)
        n_wear = int(rec.decision_window_wear_sum)
        n_foursc = int(rec.hourly_non_nan)
        n_weekly = int(rec.weekly_present_sum)
        n_daily = int(rec.daily_present_sum)
        n_cae = int(rec.cae_weeks_present)
        low_wear = n_wear < policy.min_decision_window_wear_sum
        low_foursc = n_foursc < policy.min_foursc_non_nan
        no_daily = n_daily < policy.min_daily_surveys_present
        retain_low_engagement = low_wear or low_foursc or no_daily

        reasons: list[str] = []
        if policy.exclude_if_insufficient_wear and low_wear:
            reasons.append("insufficient_decision_window_wear")
        if policy.exclude_if_insufficient_foursc and low_foursc:
            reasons.append("no_foursc_step_outcomes")
        if (
            policy.exclude_if_all_weekly_surveys_missing
            and n_weekly < policy.min_weekly_surveys_present
            and not retain_low_engagement
        ):
            reasons.append("all_weekly_surveys_missing")
        if policy.exclude_if_all_daily_surveys_missing and no_daily:
            reasons.append("all_daily_surveys_missing")
        if policy.exclude_if_all_cae_missing and n_cae < policy.min_cae_weeks_present:
            reasons.append("all_cae_missing")
        if pid in force:
            reasons = []
        elif policy.exclude_if_high_cae:
            min_cae = rec.min_weekly_cae
            week1 = rec.week1_cae
            if np.isfinite(min_cae) and float(min_cae) >= float(
                policy.min_weekly_cae_threshold
            ):
                reasons.append("min_weekly_cae_ge_6")
            if np.isfinite(week1) and np.isclose(
                float(week1), float(policy.week1_cae_ceiling)
            ):
                reasons.append("week1_cae_eq_7")
        if pid in override and not reasons:
            reasons.append("manual_override")

        span_ok = np.isfinite(rec.span_days) and float(rec.span_days) >= float(
            policy.min_active_span_days
        )
        if not span_ok and pid not in force:
            rows.append({"uid": pid, "in_modelled": False, "reason": "span_below_min"})
        elif reasons:
            rows.append({"uid": pid, "in_modelled": False, "reason": "|".join(reasons)})
        elif pid in force:
            rows.append({"uid": pid, "in_modelled": True, "reason": "force_include"})
        else:
            rows.append({"uid": pid, "in_modelled": True, "reason": "span_eligible"})
    return pd.DataFrame(rows, columns=["uid", "in_modelled", "reason"])
