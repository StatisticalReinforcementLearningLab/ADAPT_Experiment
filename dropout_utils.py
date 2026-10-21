"""Shared observed/simulated dropout measurement and fit-time masking helpers."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd


STUDY_DAYS = 84
# Day 85 is the week-12 Sunday for a Sunday starter. Weekly check-ins are
# filled onto that Sunday from a completion up to this many days later
# (measured max of observed_date - Sunday slot; fill_weekly_12 tolerance).
# The activity table runs through that window. Dropout metrics stay on days 1-84.
DEFAULT_WEEKLY_COMPLETION_DAYS = 2
ACTIVITY_TABLE_DAYS = STUDY_DAYS + 1 + DEFAULT_WEEKLY_COMPLETION_DAYS
DEFAULT_DROPOUT_GAP_DAYS = 14
DEFAULT_DROPOUT_RISK_WINDOW_DAYS = STUDY_DAYS - DEFAULT_DROPOUT_GAP_DAYS + 1
RAW_OBSERVATION_TIMESTAMP_SPECS = (
    ("SurveyTasks.csv", "InsertedDate"),
    ("SurveyResults.csv", "InsertedDate"),
    ("AnalyticsEvents_ViewViewed.csv", "Timestamp"),
    ("AnalyticsEvents_PushNotificationSent.csv", "Timestamp"),
    ("AnalyticsEvents_PushNotificationOpened.csv", "Timestamp"),
    ("filtered_activities-heart.csv", "DateTime"),
    ("filtered_activities-steps.csv", "DateTime"),
)


def max_raw_observation_timestamp(
    folder: str | Path,
    *,
    specs: Sequence[tuple[str, str]] = RAW_OBSERVATION_TIMESTAMP_SPECS,
    chunksize: int = 500_000,
) -> pd.Timestamp:
    """Return the latest observed timestamp across the raw MRT event tables.

    Only event/creation timestamps are included; scheduled due dates are
    excluded because they can extend beyond the date of the data extract.
    Raw timestamps are interpreted as UTC, matching the extraction pipeline.
    """
    root = Path(folder)
    maxima: list[pd.Timestamp] = []
    for filename, column in specs:
        path = root / filename
        if not path.is_file():
            raise FileNotFoundError(f"raw timestamp source not found: {path}")
        source_max = pd.NaT
        for chunk in pd.read_csv(path, usecols=[column], chunksize=chunksize):
            values = pd.to_datetime(chunk[column], errors="coerce", utc=True)
            chunk_max = values.max()
            if pd.notna(chunk_max) and (pd.isna(source_max) or chunk_max > source_max):
                source_max = chunk_max
        if pd.notna(source_max):
            maxima.append(source_max)
    if not maxima:
        raise ValueError("no finite timestamps found in raw observation tables")
    return max(maxima)


def calibrate_constant_daily_hazard(
    participant_dropout: pd.DataFrame,
    *,
    risk_window_days: int = DEFAULT_DROPOUT_RISK_WINDOW_DAYS,
) -> dict[str, float | int | str]:
    """Calibrate a constant hazard from completed, detectably at-risk windows.

    With terminal inactivity defined as at least 14 days inside an 84-day
    study, dropout is observable only through day 71. A detected event
    contributes ``dropout_day - 1`` person-days; every other completed window
    contributes all 71 days.
    """
    required = {"window_complete", "dropout_day"}
    missing = sorted(required.difference(participant_dropout.columns))
    if missing:
        raise KeyError(f"participant dropout table is missing columns: {missing}")
    if risk_window_days < 1:
        raise ValueError("risk_window_days must be positive")

    complete = participant_dropout.loc[
        participant_dropout["window_complete"].astype(bool)
    ].copy()
    dropout_day = pd.to_numeric(complete["dropout_day"], errors="coerce")
    event = dropout_day.notna() & dropout_day.le(float(risk_window_days))
    exposure = np.where(
        event,
        dropout_day.to_numpy(dtype=float) - 1.0,
        float(risk_window_days),
    )
    if np.any(exposure < 0) or not np.all(np.isfinite(exposure)):
        raise ValueError("dropout exposure must be finite and nonnegative")

    events = int(event.sum())
    person_days = int(np.sum(exposure))
    if events < 1 or person_days < 1:
        raise ValueError("daily hazard calibration requires positive events and exposure")
    hazard_day = float(events / person_days)
    hazard_week = float(1.0 - (1.0 - hazard_day) ** 7)
    return {
        "dropout_hazard_day": hazard_day,
        "dropout_hazard_week": hazard_week,
        "dropout_events": events,
        "dropout_person_days": person_days,
        "dropout_risk_window_days": int(risk_window_days),
        "dropout_cumulative_day84": float(1.0 - (1.0 - hazard_day) ** 84),
        "dropout_cumulative_day252": float(1.0 - (1.0 - hazard_day) ** 252),
        "dropout_definition": (
            "Last active day followed by at least 14 inactive days through "
            "the end of the completed 84-day window."
        ),
        "dropout_source": "participant_dropout.csv 2026-10-17",
    }


def longest_true_run(values: Sequence[bool]) -> int:
    """Length of the longest consecutive True run."""
    best = current = 0
    for value in np.asarray(values, dtype=bool):
        current = current + 1 if value else 0
        best = max(best, current)
    return int(best)


def summarize_activity(
    active_any: Sequence[bool],
    *,
    gap_days: int = DEFAULT_DROPOUT_GAP_DAYS,
) -> dict[str, float | int]:
    """Summarize an 84-day activity vector using terminal inactivity dropout.

    Days are one-based. ``dropout_day`` is the day after the last active day
    when inactivity continues through day 84 for at least ``gap_days``.
    An all-inactive participant has ``last_active_day=0`` and dropout day 1.
    """
    active = np.asarray(active_any, dtype=bool).ravel()
    if active.size != STUDY_DAYS:
        raise ValueError(f"active_any must have {STUDY_DAYS} days, got {active.size}")
    if gap_days < 1:
        raise ValueError("gap_days must be positive")

    active_days = np.flatnonzero(active) + 1
    last_active = int(active_days[-1]) if active_days.size else 0
    inactive = ~active
    longest_gap = longest_true_run(inactive)
    terminal_gap = STUDY_DAYS - last_active
    dropout_day = float(last_active + 1) if terminal_gap >= gap_days else np.nan
    return {
        "last_active_day": last_active,
        "longest_inactive_gap": longest_gap,
        "terminal_inactive_gap": int(terminal_gap),
        "dropout_day": dropout_day,
    }


def weekly_checkin_completion_lag(weekly: pd.DataFrame) -> pd.Series:
    """Signed days from the Sunday slot to ``observed_date``, per completed check-in."""
    need = {"date", "observed_date", "week_present"}
    missing = need.difference(weekly.columns)
    if missing:
        raise KeyError(f"weekly check-ins are missing {sorted(missing)}")
    done = weekly.loc[pd.to_numeric(weekly["week_present"], errors="coerce").eq(1)].copy()
    lag = (
        pd.to_datetime(done["observed_date"], errors="coerce")
        - pd.to_datetime(done["date"], errors="coerce")
    ).dt.days
    lag = lag.dropna().astype(int)
    if lag.empty:
        raise ValueError("no completed weekly check-ins")
    return lag


def build_participant_dropout_table(
    anchors: pd.DataFrame,
    pageview_local: pd.DataFrame,
    heart_local: pd.DataFrame,
    steps_local: pd.DataFrame,
    *,
    survey_local: pd.DataFrame | None = None,
    completion_days: int = DEFAULT_WEEKLY_COMPLETION_DAYS,
    gap_days: int = DEFAULT_DROPOUT_GAP_DAYS,
    observation_end=None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build participant and participant-day activity tables.

    ``anchors`` must contain one row per participant with ``date_min``.
    Event tables must have local ``Date`` values. Wear is any positive
    ``Value`` in heart or steps; page-view activity is any event row.
    ``survey_local`` is any daily end-of-day or weekly check-in submission,
    already converted to the participant's local calendar date. A day is
    active if it has a page view, wear, or one of those submissions.
    The daily table includes day 85 and ``completion_days`` after it so a
    week-12 Sunday's completion window is covered. Dropout metrics use only
    days 1–84. Participants whose 84-day window extends beyond
    ``observation_end`` are right-censored as ``still_enrolled``.
    """
    anchor = anchors[["ParticipantIdentifier", "date_min"]].copy()
    anchor["ParticipantIdentifier"] = anchor["ParticipantIdentifier"].astype(str)
    anchor["date_min"] = pd.to_datetime(anchor["date_min"], errors="coerce").dt.normalize()
    if anchor["ParticipantIdentifier"].duplicated().any():
        raise ValueError("anchors must contain one row per participant")
    if anchor["date_min"].isna().any():
        raise ValueError("all participant anchors must have a finite date_min")
    observation_end = (
        pd.Timestamp(observation_end).normalize()
        if observation_end is not None
        else None
    )

    def _active_dates(df: pd.DataFrame, positive_value: bool) -> set[tuple[str, object]]:
        if df.empty:
            return set()
        dat = df[["ParticipantIdentifier", "Date"] + (["Value"] if positive_value else [])].copy()
        dat["ParticipantIdentifier"] = dat["ParticipantIdentifier"].astype(str)
        dat["Date"] = pd.to_datetime(dat["Date"], errors="coerce").dt.normalize()
        keep = dat["Date"].notna()
        if positive_value:
            keep &= pd.to_numeric(dat["Value"], errors="coerce").gt(0)
        return set(zip(dat.loc[keep, "ParticipantIdentifier"], dat.loc[keep, "Date"]))

    if completion_days < 0:
        raise ValueError("completion_days must be non-negative")
    page_dates = _active_dates(pageview_local, positive_value=False)
    heart_dates = _active_dates(heart_local, positive_value=True)
    step_dates = _active_dates(steps_local, positive_value=True)
    survey_frame = survey_local if survey_local is not None else pd.DataFrame(
        columns=["ParticipantIdentifier", "Date"]
    )
    survey_dates = _active_dates(survey_frame, positive_value=False)
    table_days = STUDY_DAYS + 1 + int(completion_days)

    daily_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for row in anchor.itertuples(index=False):
        uid = str(row.ParticipantIdentifier)
        dates = pd.date_range(row.date_min, periods=table_days, freq="D")
        page = np.array([(uid, date) in page_dates for date in dates], dtype=bool)
        heart = np.array([(uid, date) in heart_dates for date in dates], dtype=bool)
        steps = np.array([(uid, date) in step_dates for date in dates], dtype=bool)
        survey = np.array([(uid, date) in survey_dates for date in dates], dtype=bool)
        wear = heart | steps
        active = page | wear | survey
        # Dropout status uses the original 84-day window, not day 85.
        window_end = dates[STUDY_DAYS - 1]
        window_complete = observation_end is None or window_end <= observation_end
        followup_days = (
            STUDY_DAYS
            if observation_end is None
            else int(np.clip((observation_end - dates[0]).days + 1, 0, STUDY_DAYS))
        )
        observed_active = active[:followup_days]
        if window_complete:
            metrics = summarize_activity(active[:STUDY_DAYS], gap_days=gap_days)
            status = "dropout" if np.isfinite(metrics["dropout_day"]) else "completed_non_dropout"
        else:
            active_days = np.flatnonzero(observed_active) + 1
            last_active = int(active_days[-1]) if active_days.size else 0
            metrics = {
                "last_active_day": last_active,
                "longest_inactive_gap": longest_true_run(~observed_active),
                "terminal_inactive_gap": int(followup_days - last_active),
                "dropout_day": np.nan,
            }
            status = "still_enrolled"

        summary_rows.append(
            {
                "ParticipantIdentifier": uid,
                **metrics,
                "window_end": window_end,
                "window_complete": bool(window_complete),
                "followup_days": followup_days,
                "dropout_status": status,
                "n_active_days": int(active[:followup_days].sum()),
                "n_pageview_days": int(page[:followup_days].sum()),
                "n_wear_days": int(wear[:followup_days].sum()),
                "n_survey_days": int(survey[:followup_days].sum()),
            }
        )
        def _day_observed(day: int, date) -> bool:
            if day <= STUDY_DAYS:
                return day <= followup_days
            if observation_end is None:
                return True
            return pd.Timestamp(date).normalize() <= pd.Timestamp(observation_end).normalize()

        daily_rows.extend(
            {
                "ParticipantIdentifier": uid,
                "study_day": day,
                "Date": date,
                "observed": int(_day_observed(day, date)),
                "pageview_any": int(page[day - 1]) if _day_observed(day, date) else np.nan,
                "wear_any": int(wear[day - 1]) if _day_observed(day, date) else np.nan,
                "survey_any": int(survey[day - 1]) if _day_observed(day, date) else np.nan,
                "active_any": int(active[day - 1]) if _day_observed(day, date) else np.nan,
            }
            for day, date in enumerate(dates, start=1)
        )

    return pd.DataFrame(summary_rows), pd.DataFrame(daily_rows)


def in_study_mask(df: pd.DataFrame) -> np.ndarray:
    """Rows before participant dropout, using stable 1..84 ``study_day``."""
    if "dropout_day" not in df.columns:
        return np.ones(len(df), dtype=bool)
    if "study_day" not in df.columns:
        raise KeyError("dropout-aware data must contain study_day")
    day = pd.to_numeric(df["study_day"], errors="coerce").to_numpy(dtype=float)
    dropout = pd.to_numeric(df["dropout_day"], errors="coerce").to_numpy(dtype=float)
    return ~np.isfinite(dropout) | (day < dropout)


def mask_post_dropout_outcomes(
    df: pd.DataFrame,
    outcome_columns: Iterable[str],
) -> pd.DataFrame:
    """Return a copy with post-dropout outcomes set to NaN, never zero."""
    out = df.copy()
    post = ~in_study_mask(out)
    columns = [col for col in outcome_columns if col in out.columns]
    if columns and post.any():
        out.loc[post, columns] = np.nan
    return out


def simulated_daily_inactivity(
    morning_wear: Sequence[float],
    daily_survey: Sequence[float],
    pageview_slots: Sequence[float],
    *,
    pageview_zero: float,
) -> np.ndarray:
    """Daily inactivity from simulator arrays (two page-view slots per day)."""
    wear = np.asarray(morning_wear, dtype=float).ravel()
    survey = np.asarray(daily_survey, dtype=float).ravel()
    pageview = np.asarray(pageview_slots, dtype=float).ravel()
    if survey.size != wear.size or pageview.size != 2 * wear.size:
        raise ValueError("simulator arrays have incompatible daily/slot lengths")
    pv = pageview.reshape(-1, 2)
    no_pageview = np.isclose(pv, float(pageview_zero), rtol=0.0, atol=1e-10).all(axis=1)
    return (wear == 0.0) & (survey == 0.0) & no_pageview
