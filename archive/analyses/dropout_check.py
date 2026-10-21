"""Targeted raw-data dropout measurement without running the fit pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dropout_utils import (
    STUDY_DAYS,
    build_participant_dropout_table,
    max_raw_observation_timestamp,
    summarize_activity,
    weekly_checkin_completion_lag,
)

SURVEY_ACTIVITY_NAMES = (
    "MRT - Daily End of day survey and Planning Exercise",
    "MRT - Weekly Check-in survey",
)


DEFAULT_TIMEZONE = "America/New_York"
TARGET_UIDS = (379, 138, 82, 259)


def _first_nonempty(values):
    for value in values:
        if pd.notna(value) and str(value).strip():
            return str(value).strip()
    return np.nan


def _participant_timezones(folder: Path) -> dict[str, str]:
    device = pd.read_csv(folder / "ProjectDeviceData_selected_fields_combined.csv")
    device = device.rename(columns={"participantidentifier": "ParticipantIdentifier"})
    device["ParticipantIdentifier"] = device["ParticipantIdentifier"].astype(str)
    zones = (
        device.groupby("ParticipantIdentifier", sort=False)["timeZone"]
        .agg(_first_nonempty)
        .to_dict()
    )
    return {
        uid: (zone if pd.notna(zone) and zone else DEFAULT_TIMEZONE)
        for uid, zone in zones.items()
    }


def _local_dates(
    timestamp: pd.Series,
    participant: pd.Series,
    zones: dict[str, str],
) -> pd.Series:
    """Convert UTC timestamps to local normalized dates by participant IANA zone."""
    uid = participant.astype(str)
    ts = pd.to_datetime(timestamp, errors="coerce", utc=True)
    out = pd.Series(pd.NaT, index=timestamp.index, dtype="datetime64[ns]")
    zone_series = uid.map(zones).fillna(DEFAULT_TIMEZONE)
    for zone, idx in zone_series.groupby(zone_series, sort=False).groups.items():
        try:
            out.loc[idx] = ts.loc[idx].dt.tz_convert(str(zone)).dt.tz_localize(None).dt.normalize()
        except Exception:
            out.loc[idx] = (
                ts.loc[idx]
                .dt.tz_convert(DEFAULT_TIMEZONE)
                .dt.tz_localize(None)
                .dt.normalize()
            )
    return out


def _tester_ids(folder: Path) -> set[str]:
    testers = pd.read_csv(folder / "Testers.csv")
    return set(testers["ParticipantIdentifier"].astype(str)) | {"test-Yuxuan"}


def _starter_anchors(folder: Path, zones: dict[str, str]) -> pd.DataFrame:
    tasks = pd.read_csv(
        folder / "SurveyTasks.csv",
        usecols=["ParticipantIdentifier", "SurveyName", "InsertedDate"],
    )
    tasks["ParticipantIdentifier"] = tasks["ParticipantIdentifier"].astype(str)
    tasks = tasks.loc[
        tasks["SurveyName"].eq("MRT - Daily End of day survey and Planning Exercise")
        & ~tasks["ParticipantIdentifier"].isin(_tester_ids(folder))
    ].copy()
    tasks["Date"] = _local_dates(
        tasks["InsertedDate"], tasks["ParticipantIdentifier"], zones
    )
    anchors = (
        tasks.dropna(subset=["Date"])
        .groupby("ParticipantIdentifier", as_index=False)["Date"]
        .agg(date_min="min", date_max="max", n_task_days="nunique")
    )
    anchors["span_days"] = (
        anchors["date_max"] - anchors["date_min"]
    ).dt.days + 1
    return anchors


def _event_days(
    path: Path,
    *,
    timestamp_col: str,
    starter_ids: set[str],
    zones: dict[str, str],
    value_col: str | None = None,
    chunksize: int = 500_000,
) -> pd.DataFrame:
    usecols = ["ParticipantIdentifier", timestamp_col]
    if value_col:
        usecols.append(value_col)
    pieces = []
    for chunk in pd.read_csv(
        path,
        usecols=usecols,
        chunksize=chunksize,
        low_memory=False,
    ):
        chunk["ParticipantIdentifier"] = chunk["ParticipantIdentifier"].astype(str)
        chunk = chunk.loc[chunk["ParticipantIdentifier"].isin(starter_ids)].copy()
        if value_col:
            chunk = chunk.loc[
                pd.to_numeric(chunk[value_col], errors="coerce").gt(0)
            ].copy()
        if chunk.empty:
            continue
        chunk["Date"] = _local_dates(
            chunk[timestamp_col], chunk["ParticipantIdentifier"], zones
        )
        keep_cols = ["ParticipantIdentifier", "Date"]
        if value_col:
            chunk["Value"] = 1.0
            keep_cols.append("Value")
        pieces.append(chunk[keep_cols].dropna(subset=["Date"]).drop_duplicates())
    if not pieces:
        columns = ["ParticipantIdentifier", "Date"] + (["Value"] if value_col else [])
        return pd.DataFrame(columns=columns)
    return pd.concat(pieces, ignore_index=True).drop_duplicates()


def _distribution(values: pd.Series) -> dict[str, float | int | None]:
    x = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if not x.size:
        return {"n": 0, "min": None, "q25": None, "median": None, "q75": None, "max": None}
    q = np.quantile(x, [0.0, 0.25, 0.5, 0.75, 1.0])
    return {
        "n": int(x.size),
        "min": float(q[0]),
        "q25": float(q[1]),
        "median": float(q[2]),
        "q75": float(q[3]),
        "max": float(q[4]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--combined-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gap-days", type=int, default=14)
    parser.add_argument(
        "--extract-date",
        default=None,
        help="Optional cutoff; must not precede the latest raw observation",
    )
    args = parser.parse_args()

    folder = args.combined_dir.expanduser().resolve()
    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    max_raw_timestamp = max_raw_observation_timestamp(folder)
    derived_extract_date = max_raw_timestamp.tz_convert(
        DEFAULT_TIMEZONE
    ).tz_localize(None).normalize()
    if args.extract_date is None:
        extract_date = derived_extract_date
    else:
        extract_date = pd.Timestamp(args.extract_date).normalize()
        if extract_date < derived_extract_date:
            raise ValueError(
                f"--extract-date {extract_date.date()} precedes latest raw "
                f"observation {max_raw_timestamp} "
                f"({derived_extract_date.date()} in {DEFAULT_TIMEZONE})"
            )

    zones = _participant_timezones(folder)
    anchors = _starter_anchors(folder, zones)
    starter_ids = set(anchors["ParticipantIdentifier"].astype(str))
    page = _event_days(
        folder / "AnalyticsEvents_ViewViewed.csv",
        timestamp_col="Timestamp",
        starter_ids=starter_ids,
        zones=zones,
    )
    heart = _event_days(
        folder / "filtered_activities-heart.csv",
        timestamp_col="DateTime",
        starter_ids=starter_ids,
        zones=zones,
        value_col="Value",
    )
    steps = _event_days(
        folder / "filtered_activities-steps.csv",
        timestamp_col="DateTime",
        starter_ids=starter_ids,
        zones=zones,
        value_col="Value",
    )
    surveys = pd.read_csv(
        folder / "SurveyResults.csv",
        usecols=["ParticipantIdentifier", "SurveyName", "InsertedDate"],
    )
    surveys["ParticipantIdentifier"] = surveys["ParticipantIdentifier"].astype(str)
    surveys = surveys.loc[
        surveys["ParticipantIdentifier"].isin(starter_ids)
        & surveys["SurveyName"].isin(SURVEY_ACTIVITY_NAMES)
    ].copy()
    surveys["Date"] = _local_dates(
        surveys["InsertedDate"], surveys["ParticipantIdentifier"], zones
    )
    survey_local = surveys.loc[surveys["Date"].notna(), ["ParticipantIdentifier", "Date"]]
    checkin_lag = weekly_checkin_completion_lag(
        pd.read_csv(
            folder / "df_weekly_filled.csv",
            usecols=["date", "observed_date", "week_present"],
        )
    )
    completion_days = int(checkin_lag.max())
    print(
        "weekly check-in lag (observed_date - Sunday slot): "
        f"{checkin_lag.value_counts().sort_index().to_dict()} max={completion_days}"
    )
    table, daily = build_participant_dropout_table(
        anchors,
        page,
        heart,
        steps,
        survey_local=survey_local,
        completion_days=completion_days,
        gap_days=args.gap_days,
        observation_end=extract_date,
    )
    table = table.merge(
        anchors[["ParticipantIdentifier", "span_days", "n_task_days"]],
        on="ParticipantIdentifier",
        how="left",
        validate="1:1",
    )

    fitted_ids: set[str] = set()
    fit_path = folder / "df_fit.csv"
    if fit_path.is_file():
        fitted_ids = set(
            pd.read_csv(fit_path, usecols=["ParticipantIdentifier"])[
                "ParticipantIdentifier"
            ].astype(str)
        )
    modelled_47_ids = fitted_ids | {"379"}
    table["in_modelled_47"] = table["ParticipantIdentifier"].isin(modelled_47_ids)

    table.to_csv(folder / "participant_dropout.csv", index=False)
    table.to_csv(out_dir / "participant_dropout.csv", index=False)
    daily.to_csv(folder / "participant_dropout_daily.csv", index=False)
    daily.to_csv(out_dir / "participant_dropout_daily.csv", index=False)

    sensitivity = {}
    active_by_uid = {
        uid: g.loc[g["observed"].eq(1) & g["study_day"].le(STUDY_DAYS)]
        .sort_values("study_day")["active_any"]
        .to_numpy(dtype=bool)
        for uid, g in daily.groupby("ParticipantIdentifier")
    }
    complete_ids = set(
        table.loc[table["window_complete"], "ParticipantIdentifier"].astype(str)
    )
    for threshold in (7, 10, 14, 21, 28):
        flags = {
            uid: np.isfinite(
                summarize_activity(active, gap_days=threshold)["dropout_day"]
            )
            for uid, active in active_by_uid.items()
            if uid in complete_ids
        }
        sensitivity[str(threshold)] = {
            "n": int(sum(flags.values())),
            "fraction": float(np.mean(list(flags.values()))),
        }

    completed = table["window_complete"]
    is_dropout = completed & table["dropout_day"].notna()
    modelled = table["in_modelled_47"]
    modelled_completed = modelled & completed
    target_table = table.loc[
            table["ParticipantIdentifier"].isin({str(x) for x in TARGET_UIDS}),
            [
                "ParticipantIdentifier",
                "last_active_day",
                "longest_inactive_gap",
                "terminal_inactive_gap",
                "dropout_day",
                "span_days",
                "window_end",
                "window_complete",
                "dropout_status",
                "in_modelled_47",
            ],
        ].sort_values("ParticipantIdentifier")
    target_table["window_end"] = pd.to_datetime(
        target_table["window_end"], errors="coerce"
    ).dt.strftime("%Y-%m-%d")
    targets = target_table.to_dict(orient="records")
    completed_table = table.loc[completed].copy()
    completed_table["dropout_week"] = np.ceil(
        completed_table["dropout_day"] / 7.0
    )
    weekly_rows = []
    for week in range(1, 13):
        events = int(completed_table["dropout_week"].eq(week).sum())
        prior_events = int(completed_table["dropout_week"].lt(week).sum())
        at_risk = int(len(completed_table) - prior_events)
        weekly_rows.append(
            {
                "week": week,
                "at_risk": at_risk,
                "dropout_events": events,
                "weekly_hazard": events / at_risk if at_risk else np.nan,
            }
        )
    weekly = pd.DataFrame(weekly_rows)
    weekly.to_csv(out_dir / "observed_weekly_dropout_hazard.csv", index=False)
    early = weekly["week"].le(4)
    early_events = int(weekly.loc[early, "dropout_events"].sum())
    late_events = int(weekly.loc[~early, "dropout_events"].sum())
    early_exposure = int(weekly.loc[early, "at_risk"].sum())
    late_exposure = int(weekly.loc[~early, "at_risk"].sum())
    early_hazard = early_events / early_exposure
    late_hazard = late_events / late_exposure

    summary = {
        "extract_date": str(extract_date.date()),
        "max_raw_observation_timestamp": str(max_raw_timestamp),
        "gap_days": int(args.gap_days),
        "n_starters": int(len(table)),
        "n_completed_windows": int(completed.sum()),
        "n_still_enrolled": int((~completed).sum()),
        "still_enrolled_ids": sorted(
            table.loc[~completed, "ParticipantIdentifier"].astype(str)
        ),
        "n_dropouts_completed_windows": int(is_dropout.sum()),
        "dropout_fraction_completed_windows": float(
            is_dropout.sum() / completed.sum()
        ),
        "n_modelled_47": int(modelled.sum()),
        "n_modelled_47_completed": int(modelled_completed.sum()),
        "n_dropouts_modelled_47": int((is_dropout & modelled).sum()),
        "dropout_fraction_modelled_47": float(
            (is_dropout & modelled).sum() / modelled_completed.sum()
        ) if modelled_completed.any() else None,
        "longest_gap_dropouts": _distribution(
            table.loc[is_dropout, "longest_inactive_gap"]
        ),
        "longest_gap_non_dropouts": _distribution(
            table.loc[completed & ~is_dropout, "longest_inactive_gap"]
        ),
        "threshold_sensitivity": sensitivity,
        "weekly_dropout_events": weekly_rows,
        "piecewise_weekly_hazard": {
            "definition": "early weeks 1-4 versus late weeks 5-12",
            "early_events": early_events,
            "early_person_weeks": early_exposure,
            "early_hazard": early_hazard,
            "late_events": late_events,
            "late_person_weeks": late_exposure,
            "late_hazard": late_hazard,
            "early_to_late_rate_ratio": (
                early_hazard / late_hazard if late_hazard > 0 else None
            ),
        },
        "targets": targets,
    }
    with open(out_dir / "observed_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, allow_nan=False)
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
