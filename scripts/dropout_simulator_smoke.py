"""Matched-seed smoke of calibrated daily dropout vs hazard zero."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from dropout_utils import calibrate_constant_daily_hazard
import experiment
from experiment import cumulative_true_cae


UIDS = (13, 22, 118)
BASE_SEED = 20261017


def _seed(uid: int, episode: int) -> int:
    return int(
        np.random.SeedSequence([BASE_SEED, int(uid), int(episode)])
        .generate_state(1, dtype=np.uint32)[0]
    )


def resolve_dropout_table(explicit: Path | None) -> Path:
    """Participant table from script 1, not the gitignored docs copy.

    ``--dropout-table`` wins. Otherwise the file is
    ``ADAPR_COMBINED_DIR/participant_dropout.csv``. When
    ``ADAPR_ACTIVITY_DAILY`` is set, the table is that file's sibling.
    """
    if explicit is not None:
        path = explicit.expanduser().resolve()
    else:
        override = os.environ.get("ADAPR_ACTIVITY_DAILY", "").strip()
        combined = os.environ.get("ADAPR_COMBINED_DIR", "").strip()
        if override:
            path = Path(override).expanduser().resolve().parent / "participant_dropout.csv"
        elif combined:
            path = Path(combined).expanduser().resolve() / "participant_dropout.csv"
        else:
            raise SystemExit(
                "Set ADAPR_COMBINED_DIR to the folder with participant_dropout.csv "
                "(written by script 1), or pass --dropout-table."
            )
    if not path.is_file():
        raise SystemExit(
            f"Missing {path}. Script 1 writes participant_dropout.csv in ADAPR_COMBINED_DIR."
        )
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--params-dir", type=Path, default=Path("env_para_vanilla"))
    parser.add_argument(
        "--dropout-table",
        type=Path,
        default=None,
        help="participant_dropout.csv (default: ADAPR_COMBINED_DIR, "
        "or the directory of ADAPR_ACTIVITY_DAILY)",
    )
    parser.add_argument("--episodes", type=int, default=80)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/dropout_check_20261017"),
    )
    args = parser.parse_args()
    params_dir = args.params_dir.expanduser().resolve()
    dropout_table = resolve_dropout_table(args.dropout_table)
    calibration = calibrate_constant_daily_hazard(pd.read_csv(dropout_table))
    hazard = float(calibration["dropout_hazard_day"])

    rows = []
    for uid in UIDS:
        for episode in range(args.episodes):
            seed = _seed(uid, episode)
            for label, h in (("off", 0.0), ("on", hazard)):
                _res, oenv = experiment.run_random_send(
                    uid, seed=seed, params_dir=params_dir, dropout_hazard_day=h
                )
                rows.append(
                    {
                        "uid": int(uid),
                        "episode": int(episode),
                        "seed": int(seed),
                        "hazard": label,
                        "dropout_day": int(oenv.dropout_day),
                        "cumulative_true_cae": cumulative_true_cae(oenv),
                    }
                )
            print(
                f"uid={uid} episode={episode} "
                f"off={rows[-2]['cumulative_true_cae']:.4f} "
                f"on={rows[-1]['cumulative_true_cae']:.4f} "
                f"day={rows[-1]['dropout_day']}",
                flush=True,
            )

    table = pd.DataFrame(rows)
    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "dropout_simulator_smoke.csv", index=False)

    summary = {"uids": list(UIDS), "episodes": int(args.episodes), "hazard_day": hazard}
    for uid in UIDS:
        sub = table.loc[table["uid"].eq(uid)]
        on = sub.loc[sub["hazard"].eq("on")]
        off = sub.loc[sub["hazard"].eq("off")]
        dropped = on["dropout_day"] > 0
        dropped12 = dropped & on["dropout_day"].le(84)
        dropped36 = dropped & on["dropout_day"].le(252)
        summary[str(uid)] = {
            "fraction_dropped_week12": float(dropped12.mean()),
            "fraction_dropped_week36": float(dropped36.mean()),
            "mean_dropout_week_among_dropouts": (
                float(np.mean(np.ceil(on.loc[dropped, "dropout_day"] / 7.0)))
                if dropped.any()
                else None
            ),
            "mean_cumulative_true_cae_off": float(off["cumulative_true_cae"].mean()),
            "mean_cumulative_true_cae_on": float(on["cumulative_true_cae"].mean()),
        }
    on = table.loc[table["hazard"].eq("on")]
    off = table.loc[table["hazard"].eq("off")]
    dropped = on["dropout_day"] > 0
    summary["pooled"] = {
        "fraction_dropped_week12": float((dropped & on["dropout_day"].le(84)).mean()),
        "fraction_dropped_week36": float((dropped & on["dropout_day"].le(252)).mean()),
        "mean_cumulative_true_cae_off": float(off["cumulative_true_cae"].mean()),
        "mean_cumulative_true_cae_on": float(on["cumulative_true_cae"].mean()),
        "expected_week12": float(1.0 - (1.0 - hazard) ** 84),
        "expected_week36": float(1.0 - (1.0 - hazard) ** 252),
    }
    (out_dir / "dropout_simulator_smoke.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
