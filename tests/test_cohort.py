import numpy as np
import pandas as pd

from cohort import CohortPolicy, modelled_membership

POLICY = CohortPolicy()


def _people(rows):
    return pd.DataFrame(rows)


def _one(**kwargs):
    base = {
        "ParticipantIdentifier": "1",
        "span_days": 84,
        "min_weekly_cae": 4.0,
        "week1_cae": 4.0,
    }
    base.update(kwargs)
    got = modelled_membership(_people([base]), POLICY)
    return bool(got.loc[0, "in_modelled"]), got.loc[0, "reason"]


def test_span_force_include_and_high_cae_ceiling():
    assert _one() == (True, "span_eligible")
    assert _one(ParticipantIdentifier="9", span_days=10) == (False, "span_below_min")
    assert _one(ParticipantIdentifier="379", span_days=10) == (True, "force_include")
    assert _one(min_weekly_cae=6.0) == (False, "min_weekly_cae_ge_6")
    assert _one(week1_cae=7.0) == (False, "week1_cae_eq_7")
    assert _one(min_weekly_cae=6.5, week1_cae=7.0) == (
        False,
        "min_weekly_cae_ge_6|week1_cae_eq_7",
    )
    # Force-include clears the ceiling.
    assert _one(ParticipantIdentifier="379", span_days=10, min_weekly_cae=6.5) == (
        True,
        "force_include",
    )


def test_manual_override_removes_after_force_include_clears_reasons():
    people = _people(
        [
            {
                "ParticipantIdentifier": "8",
                "span_days": 90,
                "min_weekly_cae": np.nan,
                "week1_cae": np.nan,
            },
            {
                "ParticipantIdentifier": "379",
                "span_days": 10,
                "min_weekly_cae": 6.5,
                "week1_cae": 7.0,
            },
        ]
    )
    policy = CohortPolicy(manual_override_ids=("8", "379"))
    got = modelled_membership(people, policy).set_index("uid")
    assert bool(got.loc["8", "in_modelled"]) is False
    assert got.loc["8", "reason"] == "manual_override"
    assert bool(got.loc["379", "in_modelled"]) is False
    assert got.loc["379", "reason"] == "manual_override"
