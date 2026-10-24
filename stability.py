"""Stability of each fitted participant model.

Primary diagnostic: the analytic loop gains g^E and g^Y (``analytic_gains``).

    g^E = a_1 + sum_{k in PV, FW, PJ} a_k D_k,   g^Y = alpha_1 + alpha_SC D_SC + alpha_AA D_AA,

where D_k = s_k / (1 - r_k) is the long-run response of mediator k's weekly
summary to a unit change of the weekly state: s_k is the per-occasion
sensitivity (state slope plus the suggestion-by-state and check-in-by-state
terms; logistic slopes p(1-p) at the participant's mean fitted probability)
and r_k is the mediator's own feedback through its history covariates
(previous-slot / previous-day lag; for steps also yesterday's steps and the
prior-2-h steps, which depend on the previous slot). Each gain is evaluated with no suggestions and with suggestions at
every Monday--Saturday decision (6 of 7 days), and with J_w in {0, 1}; the
largest |g| is reported. The weekly map is triangular (Y does not enter the
E transition), so the model is locally stable if |g^E| < 1 and |g^Y| < 1.

Numerical check: the growth factor of the deterministic skeleton
(``growth_factor``). The simulator is run with residuals set to zero and
Bernoulli variables replaced by their probabilities, from two weekly states
that differ by a small perturbation of (Y_1, E_1); g = (d_W / d_w0)^(1/(W - w0))
with d_w = |Y_w - Y'_w| + |E_w - E'_w|.

    ADAPR_PARAMS_DIR=env_para_vanilla python stability.py [--weeks 40 --burn 5]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import experiment as ex
from vani_env import Env, EnvConfig


A_EFF = 6.0 / 7.0  # suggestions on Monday--Saturday only


def _coef(theta, names, name, default=0.0):
    return float(theta[names.index(name)]) if name in names else float(default)


def _mean_pred(pred, key, default=0.5):
    v = np.asarray([x for x in pred.get(key, []) if x is not None], dtype=float)
    return float(np.mean(v)) if v.size else float(default)


def analytic_gains(uid, params_dir) -> dict:
    """Closed-form loop gains g^E and g^Y of one participant folder entry."""
    from vani_env import _load_json
    params_dir = Path(params_dir)
    return analytic_gains_from(
        _load_json(params_dir / f"params_env_{uid}.json"),
        _load_json(params_dir / f"pred_{uid}.json"),
        _load_json(params_dir / "std_params.json"),
    )


def analytic_gains_from(P: dict, pred: dict, std: dict) -> dict:
    """Closed-form loop gains g^E and g^Y (see module docstring).

    ``P`` is a participant's parameter dict (possibly modified in memory, as in
    ``tune_ste.py``), ``pred`` its ``pred_<uid>.json`` (mean fitted
    probabilities of the binary mediators), and ``std`` the standardization
    constants. Returns ``{"gE": {"never", "always"}, "gY": {"never", "always"}}``.
    """
    from vani_env import (THETA_FOURSC_NAMES, THETA_ANTIC_NAMES, THETA_CAE_NAMES,
                          THETA_PRIOR2HOUR_STEP_COUNT_NAMES,
                          pv_hurdle_occurrence_z_gap, trim_theta_cae)

    def th(key):
        return np.asarray(P[key], dtype=float)

    # ---- g^E: E -> engagement -> E ------------------------------------------
    Ew, Ewn = th("theta_penalized_Ew"), P["theta_penalized_Ew_names"]
    a1 = _coef(Ew, Ewn, "a1")
    a = {"PV": _coef(Ew, Ewn, "a2_PV_lag_week"), "FW": _coef(Ew, Ewn, "a3_FW_lag_week"),
         "PJ": _coef(Ew, Ewn, "a4_PJ_lag_week")}
    pv, pvn = th("theta_penalized_PV"), P["theta_penalized_PV_names"]
    fw, fwn = th("theta_penalized_FW"), P["theta_penalized_FW_names"]
    pj, pjn = th("theta_penalized_PJ"), P["theta_penalized_PJ_names"]
    p_occ = _mean_pred(pred, "pred_penalized_hourly_pageview")
    p_fw = _mean_pred(pred, "pred_penalized_nextday_wearing")
    p_pj = _mean_pred(pred, "pred_penalized_daily_present")
    gap = float(pv_hurdle_occurrence_z_gap(std))  # positive mean minus zero-count value

    def D_E(A, J):
        v_occ = p_occ * (1 - p_occ) * gap
        s_pv = (v_occ * (_coef(pv, pvn, "alpha1_Ew") + A * _coef(pv, pvn, "alpha4_action_by_Ew")
                         + J * pv[pvn.index("query_Jw_Ew")])
                + p_occ * (_coef(pv, pvn, "gamma1_Ew") + A * _coef(pv, pvn, "gamma4_action_by_Ew")
                           + J * _coef(pv, pvn, "intensity_query_Jw_Ew")))
        r_pv = v_occ * _coef(pv, pvn, "alpha_ar1_hourly_pageview_lag1") + p_occ * _coef(
            pv, pvn, "gamma_ar1_hourly_pageview_lag1")
        out = {"PV": (s_pv, r_pv)}
        for k, t, n, p_, pre, ar in (("FW", fw, fwn, p_fw, "beta", "beta_ar1_morning_wearing"),
                                     ("PJ", pj, pjn, p_pj, "theta", "theta_ar1_daily_present_yesterday")):
            v = p_ * (1 - p_)
            s_k = v * (_coef(t, n, f"{pre}1_Ew") + A * (_coef(t, n, f"{pre}4_A0_morning_by_Ew")
                                                        + _coef(t, n, f"{pre}6_A1_afternoon_by_Ew"))
                       + J * _coef(t, n, "query_Jw_Ew"))
            out[k] = (s_k, v * _coef(t, n, ar))
        return {k: (np.inf if r >= 1 else s_ / (1 - r)) for k, (s_, r) in out.items()}

    gE = {}
    for A in (0.0, A_EFF):
        vals = []
        for J in (0.0, 1.0):
            D = D_E(A, J)
            vals.append(a1 + sum(a[k] * D[k] for k in a))
        gE["never" if A == 0 else "always"] = max(vals, key=abs)

    # ---- g^Y: CAE -> steps / anticipated affect -> CAE -------------------------
    sc = th("theta_fourSC"); scn = THETA_FOURSC_NAMES
    aa = th("theta_antic"); aan = THETA_ANTIC_NAMES
    cae = trim_theta_cae(th("theta_CAE")); caen = THETA_CAE_NAMES
    p2 = th("theta_prior2hour_step_count"); p2n = THETA_PRIOR2HOUR_STEP_COUNT_NAMES[: p2.size]
    s4, s_y = std["4hour_step_count_scale"], std["yesterday_step_count_scale"]
    s_p, s_pe = std["prior2hour_step_count_scale"], std["EMA_prior2hour_step_count_scale"]
    lam = 0.5 * (_coef(p2, p2n, "fourSC_lag1_morning") + _coef(p2, p2n, "fourSC_lag1_afternoon"))
    eps = _coef(p2, p2n, "EMA_Prior2HourStepCount")
    p2h_resp = lam / (1 - eps * s_p / s_pe) if eps * s_p / s_pe < 1 else np.inf

    gY = {}
    for A in (0.0, A_EFF):
        s_aa = _coef(aa, aan, "CAE_avg_lastweek") + A * (
            _coef(aa, aan, "A0_morning_by_CAE_avg_lastweek") + _coef(aa, aan, "A1_afternoon_by_CAE_avg_lastweek"))
        r_aa = _coef(aa, aan, "anticipated_affect_yesterday")
        D_aa = np.inf if r_aa >= 1 else s_aa / (1 - r_aa)
        s_sc = (_coef(sc, scn, "CAE_avg_lastweek") + A * _coef(sc, scn, "WalkingSuggestion_by_CAE_avg_lastweek")
                + (_coef(sc, scn, "anticipated_affect_yesterday")
                   + A * _coef(sc, scn, "WalkingSuggestion_by_anticipated_affect_yesterday")) * D_aa)
        r_sc = (_coef(sc, scn, "fourSC_lag1")
                + (_coef(sc, scn, "yesterday_step_count") + A * _coef(sc, scn, "WalkingSuggestion_by_yesterday_step_count")) * s4 / s_y
                + (_coef(sc, scn, "prior2hour_step_count") + A * _coef(sc, scn, "WalkingSuggestion_by_prior2hour_step_count")) * p2h_resp)
        D_sc = np.inf if r_sc >= 1 else s_sc / (1 - r_sc)
        gY["never" if A == 0 else "always"] = (_coef(cae, caen, "CAE_avg_lastweek")
                                               + _coef(cae, caen, "fourSC_ewma") * D_sc
                                               + _coef(cae, caen, "anticipated_affect_ewma") * D_aa)
    return {"gE": gE, "gY": gY}


def skeleton_trajectory(uid, params_dir, action: int, nweek: int, dY=0.0, dE=0.0):
    """Weekly (Y_w, E_w), w = 0..nweek, of the deterministic skeleton."""
    cfg = EnvConfig(uid, params_dir=params_dir, nweek=nweek)
    env = Env(cfg, noise="none")
    oenv = ex.OnlineEnv(env, nweek=nweek, seed=0, params_dir=params_dir)
    oenv._reset_episode_state()
    oenv.CAE_all[0] += dY
    oenv.s["caeAverageLastWeek"] = float(oenv.CAE_all[0])
    oenv.s["caeAverage"] = float(oenv.CAE_all[0])
    oenv.pu_all[0] += dE
    oenv.s["perceivedUtilityLastWeek"] = float(oenv.pu_all[0])
    for k in range(nweek):
        oenv.start_week(k, 1)
        for d in range(ex.N_RL_DAYS):
            for t in range(ex.N_RL_SLOTS):
                oenv.step_action(k, d, t, int(action), 1)
        oenv._finalize_week(k)
    return np.asarray(oenv.CAE_all[: nweek + 1], float), np.asarray(oenv.pu_all[: nweek + 1], float)


def growth_factor(uid, params_dir, action: int, *, weeks=40, burn=5, eps=1e-2) -> dict:
    Y0, E0 = skeleton_trajectory(uid, params_dir, action, weeks)
    Y1, E1 = skeleton_trajectory(uid, params_dir, action, weeks, dY=eps, dE=eps)
    d = np.abs(Y1 - Y0) + np.abs(E1 - E0)
    clipped = bool(
        np.any(np.isclose(E0, EnvConfig(uid, params_dir=params_dir).limits_perceivedUtility[0]))
        or np.any(np.isclose(E0, EnvConfig(uid, params_dir=params_dir).limits_perceivedUtility[1]))
    )
    if d[burn] <= 0 or not np.isfinite(d[burn]):
        g = float("nan")
    elif d[weeks] <= 0:
        g = 0.0
    else:
        g = float((d[weeks] / d[burn]) ** (1.0 / (weeks - burn)))
    return {"growth": g, "E_hits_limit": clipped, "d_first": float(d[burn]), "d_last": float(d[weeks])}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--params-dir", default=None)
    ap.add_argument("--weeks", type=int, default=40)
    ap.add_argument("--burn", type=int, default=5)
    ap.add_argument("--users", default="")
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)
    params_dir = Path(ex.resolve_params_dir(args.params_dir))
    print(f"params dir: {params_dir}", flush=True)
    uids = ([int(u) for u in args.users.split(",") if u.strip()] if args.users
            else [int(u) for u in np.loadtxt(params_dir / "user_ids.txt", dtype=int).ravel()])
    out = {}
    for uid in uids:
        gains = analytic_gains(uid, params_dir)
        out[str(uid)] = {
            "analytic": gains,
            "skeleton": {pol: growth_factor(uid, params_dir, a, weeks=args.weeks, burn=args.burn)
                         for pol, a in (("never", 0), ("always", 1))},
        }
        g, sk = gains, out[str(uid)]["skeleton"]
        print(f"{uid}: gE never={g['gE']['never']:+.3f} always={g['gE']['always']:+.3f} | "
              f"gY never={g['gY']['never']:+.3f} always={g['gY']['always']:+.3f} | "
              f"skeleton never={sk['never']['growth']:.3f} always={sk['always']['growth']:.3f}")
    worst = max(max(abs(v) for v in d["analytic"]["gE"].values()) for d in out.values())
    worst_y = max(max(abs(v) for v in d["analytic"]["gY"].values()) for d in out.values())
    worst_sk = max(max(d["skeleton"]["never"]["growth"], d["skeleton"]["always"]["growth"]) for d in out.values())
    print(f"max |gE| = {worst:.3f}, max |gY| = {worst_y:.3f} "
          f"({'stable' if max(worst, worst_y) < 1 else 'UNSTABLE'}); max skeleton growth = {worst_sk:.3f}")
    path = Path(args.out) if args.out else params_dir / "stability_skeleton.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"weeks": args.weeks, "burn": args.burn, "max_abs_gE": worst, "max_abs_gY": worst_y,
                   "max_skeleton_growth": worst_sk, "participants": out}, f, indent=2)
    return out


if __name__ == "__main__":
    main()
