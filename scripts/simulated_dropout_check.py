"""Provisional fixed-policy dropout check on existing participant models."""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from dropout_utils import longest_true_run, simulated_daily_inactivity


POLICIES = ("never_send", "random_send", "always_send")
FLAGGED_UIDS = {82, 138, 259}
BASE_SEED = 20261017


def _seed(uid: int, episode: int) -> int:
    return int(
        np.random.SeedSequence([BASE_SEED, int(uid), int(episode)])
        .generate_state(1, dtype=np.uint32)[0]
    )


def _run_uid_policy(
    uid: int,
    policy: str,
    episodes: int,
    params_dir: str,
) -> list[dict[str, object]]:
    os.environ["NWEEK"] = "12"
    import experiment
    from vani_env import _standardize_pageview_count

    runners = {
        "never_send": experiment.run_never_send,
        "random_send": experiment.run_random_send,
        "always_send": experiment.run_always_send,
    }
    rows = []
    for episode in range(episodes):
        seed = _seed(uid, episode)
        _result, online_env = runners[policy](
            uid, seed=seed, params_dir=params_dir
        )
        cfg = online_env.env.cfg
        pv_zero = _standardize_pageview_count(
            0.0,
            cfg.pageview_log_shift,
            cfg.pageview_log_scale,
            zero_count=cfg.pageview_zero_count,
        )
        inactive = simulated_daily_inactivity(
            online_env.morningFitbitWearAll,
            online_env.dailySurveyCompleteAll,
            online_env.pageViewNext4HourAll,
            pageview_zero=pv_zero,
        )
        longest = longest_true_run(inactive)
        rows.append(
            {
                "uid": int(uid),
                "policy": policy,
                "episode": int(episode),
                "seed": seed,
                "longest_inactive_run": longest,
                "simulated_dropout": int(longest >= 14),
            }
        )
    return rows


def _distribution(values: pd.Series) -> dict[str, float | int]:
    x = values.to_numpy(dtype=float)
    q = np.quantile(x, [0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0])
    return {
        "n": int(x.size),
        "min": float(q[0]),
        "q25": float(q[1]),
        "median": float(q[2]),
        "q75": float(q[3]),
        "q90": float(q[4]),
        "q95": float(q[5]),
        "max": float(q[6]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--params-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    params_dir = args.params_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    user_ids = np.atleast_1d(
        np.loadtxt(params_dir / "user_ids.txt", dtype=int)
    ).astype(int)
    missing_flagged = FLAGGED_UIDS.difference(set(map(int, user_ids)))
    if missing_flagged:
        raise ValueError(f"flagged participant models missing: {sorted(missing_flagged)}")

    futures = []
    rows: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for policy in POLICIES:
            for uid in user_ids:
                futures.append(
                    pool.submit(
                        _run_uid_policy,
                        int(uid),
                        policy,
                        int(args.episodes),
                        str(params_dir),
                    )
                )
        for future in as_completed(futures):
            rows.extend(future.result())

    episodes = pd.DataFrame(rows).sort_values(
        ["policy", "uid", "episode"]
    ).reset_index(drop=True)
    episodes["group"] = np.where(
        episodes["uid"].isin(FLAGGED_UIDS), "flagged_dropout_model", "non_dropout_39"
    )
    episodes.to_csv(output_dir / "simulated_dropout_episodes.csv", index=False)

    summary: dict[str, object] = {
        "n_models": int(len(user_ids)),
        "n_non_dropout_models": int((~np.isin(user_ids, list(FLAGGED_UIDS))).sum()),
        "flagged_models": sorted(FLAGGED_UIDS),
        "episodes_per_model_policy": int(args.episodes),
        "nweek": 12,
        "policies": {},
    }
    for policy in POLICIES:
        policy_df = episodes.loc[episodes["policy"].eq(policy)]
        groups: dict[str, object] = {}
        for label, group_df in (
            ("pooled_42", policy_df),
            (
                "non_dropout_39",
                policy_df.loc[policy_df["group"].eq("non_dropout_39")],
            ),
        ):
            groups[label] = {
                "n_episodes": int(len(group_df)),
                "dropout_fraction": float(group_df["simulated_dropout"].mean()),
                "longest_inactive_run": _distribution(
                    group_df["longest_inactive_run"]
                ),
                "n_models_with_any_dropout_episode": int(
                    group_df.groupby("uid")["simulated_dropout"].max().sum()
                ),
            }
        for uid in sorted(FLAGGED_UIDS):
            group_df = policy_df.loc[policy_df["uid"].eq(uid)]
            groups[f"uid_{uid}"] = {
                "n_episodes": int(len(group_df)),
                "dropout_fraction": float(group_df["simulated_dropout"].mean()),
                "longest_inactive_run": _distribution(
                    group_df["longest_inactive_run"]
                ),
            }
        summary["policies"][policy] = groups

    with open(output_dir / "simulated_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, allow_nan=False)
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
