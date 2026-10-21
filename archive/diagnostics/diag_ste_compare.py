"""Per-participant STE of the trained DiscreteCQL vs simple fixed policies.

Reads ``results_ste/exp<EXP>/res<EXP>_<uid>.txt`` (columns: never-send total,
CQL total on test seeds TEST_SEED0 + n) and simulates always-send and
Bernoulli(0.5) on the *same* test seeds, so all arms share common random
numbers. Reports, per participant, Δ/σ for CQL, always-send and random, and
the mean of the clipped values. If CQL is a good stand-in for the optimal
policy, its STE should be at least the best fixed policy's for most people.

    ADAPR_PARAMS_DIR=env_para_vanilla python diag_ste_compare.py --exp 1 --n 100 --jobs 8
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np


def _one(task):
    uid, exp, n, params_dir = task
    import numpy.random as rd

    import ste_vanilla as sv

    res = np.loadtxt(Path("results_ste") / f"exp{exp}" / f"res{exp}_{uid}.txt")
    res = res.reshape(-1, 2)[:n]
    g0, gc = res[:, 0], res[:, 1]
    nweek = int(sv.EnvConfig(uid, params_dir=Path(params_dir)).nweek)
    arms = {}
    for label, p in (("always", 1.0), ("random", 0.5)):
        tot = np.empty(len(g0))
        for i in range(len(g0)):
            seed = sv.TEST_SEED0 + i
            rd.seed(seed)
            tot[i] = sv.rollout_total_cae(uid, nweek=nweek, seed=seed, policy="bernoulli",
                                          walk_prob=p, params_dir=Path(params_dir))
        arms[label] = tot
    # Never-send must reproduce the stored column (checks common random numbers).
    rd.seed(sv.TEST_SEED0)
    z0 = sv.rollout_total_cae(uid, nweek=nweek, seed=sv.TEST_SEED0, policy="zero",
                              params_dir=Path(params_dir))
    sd = float(np.std(g0, ddof=1))
    out = {"uid": int(uid), "sigma": sd, "crn_check": float(z0 - g0[0]),
           "cql": float(np.mean(gc - g0)) / sd}
    for k, v in arms.items():
        out[k] = float(np.mean(v - g0)) / sd
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--n", type=int, default=100, help="test episodes per arm (first n test seeds)")
    ap.add_argument("--jobs", type=int, default=None)
    ap.add_argument("--params-dir", default=None)
    a = ap.parse_args()
    import ste_vanilla as sv

    params_dir = str(sv._resolve_params_dir(a.params_dir))
    uids = np.loadtxt(Path(params_dir) / "user_ids.txt", dtype=int)
    tasks = [(int(u), a.exp, a.n, params_dir) for u in uids]
    with ProcessPoolExecutor(max_workers=a.jobs) as pool:
        rows = list(pool.map(_one, tasks))
    print(f"{'uid':>5} {'sigma':>7} {'CQL':>7} {'always':>7} {'random':>7}  best_fixed  crn_check")
    for r in sorted(rows, key=lambda r: r["cql"] - max(r["always"], r["random"], 0.0)):
        best = max(r["always"], r["random"], 0.0)
        flag = "  <-- CQL below best fixed" if r["cql"] < best - 0.05 else ""
        print(f"{r['uid']:>5} {r['sigma']:>7.2f} {r['cql']:>+7.3f} {r['always']:>+7.3f} "
              f"{r['random']:>+7.3f}  {best:>+9.3f}  {r['crn_check']:>+8.4f}{flag}")
    clip = lambda k: np.mean([max(r[k], 0.0) for r in rows])  # noqa: E731
    best_fixed = np.mean([max(r["always"], r["random"], 0.0) for r in rows])
    best_all = np.mean([max(r["cql"], r["always"], r["random"], 0.0) for r in rows])
    print(f"\nmean clipped STE:  CQL {clip('cql'):.3f}   always {clip('always'):.3f}   "
          f"random {clip('random'):.3f}   best fixed {best_fixed:.3f}   best of all {best_all:.3f}")
    n_below = sum(r["cql"] < max(r["always"], r["random"], 0.0) - 0.05 for r in rows)
    print(f"CQL below the best fixed policy by > 0.05 for {n_below} of {len(rows)} participants")


if __name__ == "__main__":
    main()
