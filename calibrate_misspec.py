"""Calibrate misspecification A and B strengths once per params dir.

After a refit::

    python calibrate_misspec.py --params-dir env_para_ste0.2
    python calibrate_misspec.py --params-dir env_para_ste0.5
    python calibrate_misspec.py --params-dir env_para_ste0.8

or all three STE folders::

    python calibrate_misspec.py --all-ste

A. Fix ``lam_Y = 0.5``. Solve ``lam_SC``, ``lam_AA`` so that, under the
random-0.5 policy, the pooled OLS of ``Y_{w+1}`` on
``(1, Y_w, SC_ewma, AA_ewma)`` recovers the population fitted
``alpha^Y_SC`` and ``alpha^Y_AA`` within 10%. Default horizon is 12 weeks
(``--nweek 12``), matching the 11-week fitting panel those alphas come from.

B. Solve ``beta_D`` so the always-send minus never-send 36-week cumulative
CAE difference equals the original (misspec-off) difference. B always uses
36 weeks (``--nweek-b``).

Writes ``<params_dir>/misspec_calibration.json``. Does not retune STE.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from experiment import _make_online_env, run_always_send, run_never_send
from misspec import (
    CAE_AA_IDX,
    CAE_SC_IDX,
    CONFOUNDER_LAM_Y,
    CONFOUNDER_RHO,
    save_calibration,
)
from testbed import STE_PARAM_DIRS
from vani_env import EnvConfig, build_CAE_features, resolve_params_dir


def _user_ids(params_dir: Path) -> list[int]:
    return [int(x) for x in np.loadtxt(params_dir / "user_ids.txt", dtype=int)]


def population_cae_alphas(params_dir: Path, user_ids: list[int]) -> tuple[float, float]:
    """Mean fitted ``alpha^Y_SC``, ``alpha^Y_AA`` across participants."""
    sc, aa = [], []
    for uid in user_ids:
        cfg = EnvConfig(int(uid), params_dir=params_dir, activity="off", misspec="none")
        theta = np.asarray(cfg.theta_CAE, dtype=float)
        sc.append(float(theta[CAE_SC_IDX]))
        aa.append(float(theta[CAE_AA_IDX]))
    return float(np.mean(sc)), float(np.mean(aa))


def _ols_alphas(X: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(beta[CAE_SC_IDX]), float(beta[CAE_AA_IDX])


def _install_A(cfg, lam_sc: float, lam_aa: float) -> None:
    cfg.misspec_lam_Y = CONFOUNDER_LAM_Y
    cfg.misspec_lam_SC = float(lam_sc)
    cfg.misspec_lam_AA = float(lam_aa)
    cfg.misspec_rho = CONFOUNDER_RHO


def _run_A_episode(params_dir, uid, seed, nweek, lam_sc, lam_aa) -> tuple[np.ndarray, np.ndarray]:
    cfg, env, oenv = _make_online_env(
        int(uid), seed=seed, params_dir=params_dir,
        activity="off", misspec="A", nweek=nweek,
        misspec_strengths={"lam_SC": lam_sc, "lam_AA": lam_aa},
    )
    _install_A(cfg, lam_sc, lam_aa)
    rec_y: list[float] = []
    rec_x: list[np.ndarray] = []
    orig = env.gen_CAE

    def wrapped(cae_last, foursc_wk, antic_wk, week_idx):
        x = build_CAE_features(cae_last, foursc_wk, antic_wk)
        y = orig(cae_last, foursc_wk, antic_wk, week_idx)
        rec_y.append(float(y))
        rec_x.append(np.asarray(x, dtype=float))
        return y

    env.gen_CAE = wrapped
    oenv._reset_episode_state()
    rng = np.random.default_rng(seed)
    from experiment import N_RL_DAYS, N_RL_SLOTS

    for k in range(oenv.nweek):
        oenv.start_week(k, 1)
        for d in range(N_RL_DAYS):
            d_global = oenv._day_idx(k, d)
            oenv._ensure_day_started(k, d, d_global)
            for t in range(N_RL_SLOTS):
                a = 0 if oenv.dropped_out else int(rng.random() < 0.5)
                oenv.step_action(k, d, t, float(a), 1)
        oenv._finalize_week(k)
    if not rec_x:
        return np.empty((0, 4)), np.empty(0)
    return np.vstack(rec_x), np.asarray(rec_y, dtype=float)


def pooled_ols_A(params_dir, user_ids, lam_sc, lam_aa, *, nweek, episodes, seed0):
    xs, ys = [], []
    for e in range(int(episodes)):
        for uid in user_ids:
            x, y = _run_A_episode(
                params_dir, uid, int(seed0) + e, nweek, lam_sc, lam_aa
            )
            if y.size:
                xs.append(x)
                ys.append(y)
    X = np.vstack(xs)
    y = np.concatenate(ys)
    return _ols_alphas(X, y), int(y.size)


def calibrate_A(
    params_dir: Path,
    user_ids: list[int],
    *,
    nweek: int,
    episodes: int,
    seed0: int,
    tol: float = 0.10,
) -> dict:
    target_sc, target_aa = population_cae_alphas(params_dir, user_ids)
    print(
        f"  A targets: alpha_SC={target_sc:.4f}  alpha_AA={target_aa:.4f}  "
        f"(mean over {len(user_ids)} users, nweek={nweek})",
        flush=True,
    )

    def relerr(got, target):
        if abs(target) < 1e-8:
            return abs(got - target)
        return abs(got / target - 1.0)

    def loss(lams):
        (got_sc, got_aa), n = pooled_ols_A(
            params_dir, user_ids, lams[0], lams[1],
            nweek=nweek, episodes=episodes, seed0=seed0,
        )
        e_sc, e_aa = relerr(got_sc, target_sc), relerr(got_aa, target_aa)
        print(
            f"    lam_SC={lams[0]:.3f} lam_AA={lams[1]:.3f}  "
            f"OLS SC={got_sc:.4f} AA={got_aa:.4f}  "
            f"relerr={e_sc:.3f}/{e_aa:.3f}  n={n}",
            flush=True,
        )
        return np.array([got_sc - target_sc, got_aa - target_aa], dtype=float), (got_sc, got_aa, e_sc, e_aa)

    # Coarse grid, then one Nelder-Mead-style shrink around the best cell.
    grid = [0.15, 0.30, 0.45, 0.60]
    best = None
    best_pack = None
    for ls in grid:
        for la in grid:
            pack = loss((ls, la))
            score = float(np.mean(np.abs(pack[0])))
            if best is None or score < best[0]:
                best = (score, ls, la)
                best_pack = pack
    lam_sc, lam_aa = best[1], best[2]
    resid, (got_sc, got_aa, e_sc, e_aa) = best_pack
    if e_sc > tol or e_aa > tol:
        try:
            from scipy.optimize import least_squares

            def fun(v):
                return loss((float(v[0]), float(v[1])))[0]

            sol = least_squares(fun, [lam_sc, lam_aa], bounds=([0.0, 0.0], [0.95, 0.95]))
            lam_sc, lam_aa = float(sol.x[0]), float(sol.x[1])
            resid, (got_sc, got_aa, e_sc, e_aa) = loss((lam_sc, lam_aa))
        except Exception as exc:
            print(f"  [warn] refine failed ({exc}); keeping grid solution", flush=True)
    payload = {
        "lam_Y": CONFOUNDER_LAM_Y,
        "lam_SC": lam_sc,
        "lam_AA": lam_aa,
        "rho": CONFOUNDER_RHO,
        "target_alpha_SC": target_sc,
        "target_alpha_AA": target_aa,
        "ols_alpha_SC": got_sc,
        "ols_alpha_AA": got_aa,
        "relerr_SC": e_sc,
        "relerr_AA": e_aa,
        "within_10pct": bool(e_sc <= tol and e_aa <= tol),
        "nweek": int(nweek),
        "episodes": int(episodes),
        "seed0": int(seed0),
        "n_users": len(user_ids),
    }
    return payload


def _cum_cae(oenv) -> float:
    weekly = np.asarray(oenv.CAE_all[1 : oenv.nweek + 1], dtype=float)
    return float(np.nansum(weekly))


def _run_B_pair(params_dir, uid, seed, nweek, beta_d):
    """Always minus never cumulative CAE under misspec B with a live beta_D."""
    totals = []
    for policy, a_val in (("never_send", 0.0), ("always_send", 1.0)):
        cfg, env, oenv = _make_online_env(
            uid, seed=seed, params_dir=params_dir,
            activity="off", misspec="B", nweek=nweek,
            misspec_strengths={"beta_D": beta_d},
        )
        cfg.misspec_beta_D = float(beta_d)
        oenv._reset_episode_state()
        from experiment import N_RL_DAYS, N_RL_SLOTS

        for k in range(oenv.nweek):
            oenv.start_week(k, 1)
            for d in range(N_RL_DAYS):
                d_global = oenv._day_idx(k, d)
                oenv._ensure_day_started(k, d, d_global)
                for t in range(N_RL_SLOTS):
                    oenv.step_action(k, d, t, 0.0 if oenv.dropped_out else a_val, 1)
            oenv._finalize_week(k)
        totals.append(_cum_cae(oenv))
    return totals[1] - totals[0]


def _run_none_pair(params_dir, uid, seed, nweek):
    _n, never = run_never_send(
        uid, seed=seed, params_dir=params_dir, activity="off", misspec="none", nweek=nweek,
    )
    _a, always = run_always_send(
        uid, seed=seed, params_dir=params_dir, activity="off", misspec="none", nweek=nweek,
    )
    return _cum_cae(always) - _cum_cae(never)


def calibrate_B(
    params_dir: Path,
    user_ids: list[int],
    *,
    nweek: int,
    episodes: int,
    seed0: int,
) -> dict:
    original = []
    for e in range(int(episodes)):
        for uid in user_ids:
            original.append(
                _run_none_pair(params_dir, uid, int(seed0) + e, nweek)
            )
    target = float(np.mean(original))
    print(f"  B target always−never cumCAE = {target:.4f}", flush=True)

    def gap(beta):
        diffs = []
        for e in range(int(episodes)):
            for uid in user_ids:
                diffs.append(
                    _run_B_pair(params_dir, uid, int(seed0) + e, nweek, beta)
                )
        got = float(np.mean(diffs))
        print(f"    beta_D={beta:.4f}  gap={got:.4f}  err={got - target:.4f}", flush=True)
        return got - target, got

    # Bracket and bisection. Always−never under B at beta=0 is the halved path.
    lo, hi = -2.0, 2.0
    err_lo, _ = gap(lo)
    err_hi, _ = gap(hi)
    if err_lo * err_hi > 0:
        # Expand once.
        if abs(err_lo) < abs(err_hi):
            lo, err_lo = lo - 2.0, gap(lo - 2.0)[0]
        else:
            hi, err_hi = hi + 2.0, gap(hi + 2.0)[0]
    beta = 0.0
    got = target
    for _ in range(12):
        mid = 0.5 * (lo + hi)
        err_mid, got = gap(mid)
        beta = mid
        if abs(err_mid) < 0.02 * max(abs(target), 1.0):
            break
        if err_lo * err_mid <= 0:
            hi, err_hi = mid, err_mid
        else:
            lo, err_lo = mid, err_mid
    return {
        "beta_D": float(beta),
        "target_always_minus_never": target,
        "achieved_always_minus_never": float(got),
        "nweek": int(nweek),
        "episodes": int(episodes),
        "seed0": int(seed0),
        "n_users": len(user_ids),
    }


def calibrate_dir(params_dir: Path, args) -> dict:
    params_dir = resolve_params_dir(params_dir)
    user_ids = _user_ids(params_dir)
    if args.max_users and args.max_users < len(user_ids):
        user_ids = user_ids[: int(args.max_users)]
    print(f"Calibrating {params_dir}  ({len(user_ids)} users)", flush=True)
    t0 = time.perf_counter()
    payload = {"params_dir": str(params_dir)}
    if args.which in {"A", "both"}:
        payload["A"] = calibrate_A(
            params_dir, user_ids,
            nweek=args.nweek, episodes=args.episodes, seed0=args.seed,
        )
    if args.which in {"B", "both"}:
        payload["B"] = calibrate_B(
            params_dir, user_ids,
            nweek=args.nweek_b, episodes=args.episodes, seed0=args.seed,
        )
    path = save_calibration(params_dir, {k: payload[k] for k in payload if k in {"A", "B"}})
    print(f"  wrote {path}  ({time.perf_counter() - t0:.1f}s)", flush=True)
    return payload


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--params-dir", default=None)
    p.add_argument(
        "--all-ste",
        action="store_true",
        help="Calibrate env_para_ste0.2 / 0.5 / 0.8",
    )
    p.add_argument("--which", choices=("A", "B", "both"), default="both")
    p.add_argument(
        "--nweek",
        type=int,
        default=12,
        help="Weeks for misspec A only (default 12; matches the 11-week fit panel).",
    )
    p.add_argument(
        "--nweek-b",
        type=int,
        default=36,
        help="Weeks for misspec B (default 36).",
    )
    p.add_argument("--episodes", type=int, default=4)
    p.add_argument("--seed", type=int, default=20261026)
    p.add_argument("--max-users", type=int, default=None)
    return p


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    dirs = []
    if args.all_ste:
        dirs.extend(STE_PARAM_DIRS)
    if args.params_dir:
        dirs.append(args.params_dir)
    if not dirs:
        dirs.append(None)
    reports = []
    for d in dirs:
        reports.append(calibrate_dir(d, args))
    print(json.dumps(
        [{k: v for k, v in r.items() if k != "params_dir"} | {"params_dir": r.get("params_dir")}
         for r in reports],
        indent=2,
    ))


if __name__ == "__main__":
    main()
