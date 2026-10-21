"""Estimate the RLSVI prior and the leave-one-user-out priors.

``experiment.py`` imports ``load_estimated_priors`` from ``est_prior.py``,
so the fitting module keeps that name. Run this after ``08_est_ew_weights.py``.
"""

from est_prior import cli

if __name__ == "__main__":
    cli()
