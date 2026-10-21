"""Check that the engagement and outcome maps stay inside the unit circle.

``tune_ste.py`` imports ``analytic_gains_from`` and ``growth_factor`` from
``stability.py``, so the module keeps that name.
"""

from stability import main

if __name__ == "__main__":
    main()
