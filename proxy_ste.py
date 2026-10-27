"""Fast proxy STE for a testbed spec; screen Layer-3 folders.

Given a testbed, this reports:

* the no-gap proxy STE (activity forced off), the number used to screen;
* the effective proxy STE under the testbed's own activity setting;
* a scaled proxy STE = proxy STE × the confirmed/proxy ratio in the
  params dir's ``ste_tuning.json`` (written by ``TUNE_PHASE=ratio``).
  That is an approximate full STE. Layer 3 uses only the three uniform
  folders, so the ratio is the uniform folder's. If the ratio is missing,
  the proxy is printed alone with the reason.

Layer-3 screening flags any spec whose no-gap proxy STE is below 0.1::

    python proxy_ste.py --screen-layer3
    python proxy_ste.py --params-dir env_para_ste0.5 --activity on --misspec A

Uses the existing ``tune_ste.evaluate`` proxy (Bernoulli 0.5 vs never-send
by default; pass ``--dqn-exp`` to add transferred DiscreteCQL). Layer 3
does not run ``ste_vanilla.py`` here; a full DiscreteCQL STE is optional
later if time allows.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from testbed import (
    MISSPEC_BLURBS,
    add_testbed_arguments,
    catalog,
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


def load_confirmation_ratio(params_dir) -> tuple[float | None, str]:
    """``proxy_to_true_estimate`` from ``<params_dir>/ste_tuning.json``, or why not."""
    path = Path(params_dir).expanduser().resolve() / "ste_tuning.json"
    if not path.is_file():
        return None, (
            f"{path.name} is missing in {path.parent.name}; "
            "run TUNE_PHASE=ratio after confirmation"
        )
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None, f"{path} is not valid JSON"
    conf = report.get("confirmation") or {}
    ratio = conf.get("proxy_to_true_estimate")
    if ratio is None:
        return None, (
            f"{path.parent.name}/ste_tuning.json has no confirmation.proxy_to_true_estimate; "
            "run TUNE_PHASE=ratio first"
        )
    try:
        value = float(ratio)
    except (TypeError, ValueError):
        return None, f"confirmation ratio in {path.parent.name} is not a number"
    if not math.isfinite(value) or value <= 0:
        return None, f"confirmation ratio in {path.parent.name} is {ratio!r}"
    return value, ""


def _format_proxy_line(report: dict) -> str:
    line = f"  proxy STE (activity off) = {report['proxy_ste']:.4f}"
    if report.get("scaled_proxy_ste") is not None:
        line += (
            f"  scaled proxy STE = {report['scaled_proxy_ste']:.4f}"
            f"  (ratio {report['confirmation_ratio']:.3f})"
        )
    else:
        reason = report.get("confirmation_ratio_reason") or "confirmation ratio missing"
        line += f"  (scaled proxy STE omitted: {reason})"
    if report.get("effective_proxy_ste") is not None:
        line += f"  effective = {report['effective_proxy_ste']:.4f}"
    return line


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
    ratio, ratio_reason = load_confirmation_ratio(params_dir)
    proxy_ste = float(nogap["mean_ste"])
    scaled = None if ratio is None else proxy_ste * ratio
    return {
        "testbed": spec.as_dict(),
        "proxy_ste": proxy_ste,
        "scaled_proxy_ste": scaled,
        "confirmation_ratio": ratio,
        "confirmation_ratio_reason": None if ratio is not None else ratio_reason,
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
    rows = [s for s in catalog() if s.layer == 3]
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
        print(_format_proxy_line(report))
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
        f"{spec.suffix}: {_format_proxy_line(report).lstrip()}"
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
