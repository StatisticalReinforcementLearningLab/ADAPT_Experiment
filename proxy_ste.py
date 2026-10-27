"""Fast proxy STE for a testbed spec; screen Layer-3 folders.

Given a testbed, this reports:

* the no-gap proxy STE (activity forced off), the number used to screen;
* the effective proxy STE under the testbed's own activity setting.

Layer-3 screening flags any spec whose no-gap proxy STE is below 0.1::

    python proxy_ste.py --screen-layer3
    python proxy_ste.py --params-dir env_para_ste0.5 --activity on --misspec A

Uses the existing ``tune_ste.evaluate`` proxy (Bernoulli 0.5 vs never-send
by default; pass ``--dqn-exp`` to add transferred DiscreteCQL).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from testbed import (
    MISSPEC_BLURBS,
    add_testbed_arguments,
    catalog_24,
    log_ste_activity_off,
    resolve_testbed_spec,
)
from tune_ste import (
    PARAMS_DIR,
    ProxySpec,
    evaluate,
    load_user_ids,
    make_spec,
    require_fitted_blocks,
    resolve_dqn_dir,
    summarise,
    write_report,
)

SCREEN_THRESHOLD = 0.1


def _proxy_spec(args) -> ProxySpec:
    grid = tuple(float(p) for p in (getattr(args, "policy_grid", None) or []))
    if not grid and not getattr(args, "dqn_exp", None) and not getattr(args, "dqn_model_dir", None):
        grid = (0.5,)
    args.policy_grid = list(grid)
    if not hasattr(args, "dqn_source_dir"):
        args.dqn_source_dir = str(PARAMS_DIR)
    if not hasattr(args, "dqn_exp"):
        args.dqn_exp = None
    if not hasattr(args, "dqn_model_dir"):
        args.dqn_model_dir = None
    return make_spec(args)


def measure_spec(spec, args) -> dict:
    params_dir = Path(spec.params_dir).expanduser().resolve()
    user_ids = load_user_ids(params_dir)
    require_fitted_blocks(params_dir, user_ids)
    proxy = _proxy_spec(args)
    ste_spec = spec.ste_spec()
    log_ste_activity_off(spec)
    nogap = evaluate(
        params_dir,
        user_ids,
        proxy,
        n_jobs=args.jobs,
        proxy_to_true=getattr(args, "proxy_to_true", 1.0),
        activity="off",
        misspec=ste_spec.misspec,
        stayer_fraction=ste_spec.stayer_fraction,
        log_activity=False,
    )
    effective = None
    if spec.activity != "off" or spec.misspec != ste_spec.misspec:
        effective = evaluate(
            params_dir,
            user_ids,
            proxy,
            n_jobs=args.jobs,
            proxy_to_true=getattr(args, "proxy_to_true", 1.0),
            activity=spec.activity,
            misspec=spec.misspec,
            stayer_fraction=spec.stayer_fraction,
            log_activity=False,
        )
    flag = float(nogap["mean_ste"]) < SCREEN_THRESHOLD
    return {
        "testbed": spec.as_dict(),
        "proxy_ste": nogap["mean_ste"],
        "effective_proxy_ste": None if effective is None else effective["mean_ste"],
        "flag_below_0.1": flag,
        "n_users": nogap["n_users"],
        "nogap": {k: nogap[k] for k in ("mean_ste", "median_ste", "min_ste", "max_ste")},
        "effective": (
            None
            if effective is None
            else {k: effective[k] for k in ("mean_ste", "median_ste", "min_ste", "max_ste")}
        ),
    }


def screen_layer3(args) -> int:
    rows = [s for s in catalog_24() if s.layer == 3]
    print(f"Screening {len(rows)} Layer-3 testbeds (flag proxy STE < {SCREEN_THRESHOLD})")
    flagged = []
    reports = []
    for spec in rows:
        print(
            f"\n=== {spec.suffix}  ({MISSPEC_BLURBS.get(spec.misspec, spec.misspec)}) ===",
            flush=True,
        )
        try:
            report = measure_spec(spec, args)
        except FileNotFoundError as exc:
            print(f"  SKIP {exc}")
            reports.append({"testbed": spec.as_dict(), "error": str(exc)})
            continue
        reports.append(report)
        print(
            f"  proxy STE (activity off) = {report['proxy_ste']:.4f}"
            + (
                ""
                if report["effective_proxy_ste"] is None
                else f"  effective = {report['effective_proxy_ste']:.4f}"
            )
        )
        if report["flag_below_0.1"]:
            flagged.append(spec.suffix)
            print("  FLAG: proxy STE < 0.1")
    if args.report:
        write_report(Path(args.report), {"threshold": SCREEN_THRESHOLD, "rows": reports})
    print(f"\nflagged {len(flagged)}/{len(rows)}: {flagged or 'none'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    add_testbed_arguments(p, params_dir=True)
    p.add_argument("--screen-layer3", action="store_true")
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--seed", type=int, default=ProxySpec.seed0)
    p.add_argument("--noise", default="ar1", choices=["ar1", "random", "sequential"])
    p.add_argument("--policy-grid", type=float, nargs="*", default=[0.5])
    p.add_argument("--jobs", type=int, default=None)
    p.add_argument("--proxy-to-true", default="1.0")
    p.add_argument("--dqn-exp", default=None)
    p.add_argument("--dqn-model-dir", default=None)
    p.add_argument("--dqn-source-dir", default=str(PARAMS_DIR))
    p.add_argument("--report", default=None)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    from tune_ste import resolve_proxy_to_true

    args.proxy_to_true = resolve_proxy_to_true(args.proxy_to_true)
    if args.screen_layer3:
        return screen_layer3(args)
    spec = resolve_testbed_spec(
        args.params_dir,
        activity=args.activity,
        stayer_fraction=args.stayer_fraction,
        misspec=args.misspec,
    )
    report = measure_spec(spec, args)
    print(
        f"{spec.suffix}: proxy STE (off) = {report['proxy_ste']:.4f}"
        + (
            ""
            if report["effective_proxy_ste"] is None
            else f"  effective = {report['effective_proxy_ste']:.4f}"
        )
        + ("  FLAG < 0.1" if report["flag_below_0.1"] else "")
    )
    summarise(
        {
            "mean_ste": report["proxy_ste"],
            "median_ste": report["nogap"]["median_ste"],
            "min_ste": report["nogap"]["min_ste"],
            "max_ste": report["nogap"]["max_ste"],
            "n_users": report["n_users"],
            "n_users_total": report["n_users"],
            "n_users_nan": 0,
            "users": [],
        }
    )
    if args.report:
        write_report(Path(args.report), report)
    return 1 if report["flag_below_0.1"] else 0


if __name__ == "__main__":
    sys.exit(main())
