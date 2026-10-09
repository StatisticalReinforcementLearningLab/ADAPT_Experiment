"""Summaries and figures for ``diag_pf_compare.py`` outputs.

Usage::

    python diag_pf_summarize.py --runs pf_cmp --out pf_cmp_summary
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

LABEL = {"old": "Old: PL with parameter draws",
         "rbpl": "New: RBPL",
         "rbpl_obsEB": "New: RBPL + observed-pair EB"}
PHASES = [0, 5, 12, 24, 36]
PHASE_LAB = ["2-5", "6-12", "13-24", "25-36"]
RUNS = [-1, 0, 2, 5, 10, 40]
RUN_LAB = ["0", "1-2", "3-5", "6-10", ">10"]


def load(runs_dir):
    wk, rn = [], []
    for f in glob.glob(str(Path(runs_dir) / "*.json")):
        r = json.load(open(f))
        J, obs = r["J"], r["obs"]
        run = 0
        pdv = np.asarray(r["path_distinct"])
        T = len(obs)
        for k in range(1, T):
            wk.append(dict(method=r["method"], J=J, uid=r["uid"], seed=r["seed"], fs=r["fseed"], k=k,
                           obs=obs[k], prev_run=run, ess=r["ess"][k], essr=r["ess"][k] / J,
                           res=r["resampled"][k], anc=r["ancestors"][k] / J,
                           bhat=r["b_hat"][k], btil=r["b_tilde"][k], truth=r["truth"][k],
                           paths=pdv[k], age=T - 1 - k))
            run = run + 1 if obs[k] != "Y" else 0
        rn.append(dict(method=r["method"], J=J, uid=r["uid"], seed=r["seed"], fs=r["fseed"],
                       s2Y=r["sigma2_Y"], sec=r["seconds"]))
    W, R = pd.DataFrame(wk), pd.DataFrame(rn)
    W["phase"] = pd.cut(W.k, PHASES, labels=PHASE_LAB)
    W["runbin"] = pd.cut(W.prev_run, RUNS, labels=RUN_LAB)
    W["err"] = W.bhat - W.truth
    W["cov95"] = W.err.abs() <= 1.96 * W.btil
    return W, R


def tables(W, R):
    out = {}
    g = W.groupby(["method", "J"])
    out["ess"] = pd.DataFrame({
        "median ESS/J": g.essr.median(),
        "weeks with ESS<2 (%)": 100 * g.ess.apply(lambda x: (x < 2).mean()),
        "weeks with ESS/J<0.1 (%)": 100 * g.essr.apply(lambda x: (x < 0.1).mean()),
        "weeks resampled (%)": 100 * g.res.mean(),
        "distinct ancestors/J when resampled": W[W.res == 1].groupby(["method", "J"]).anc.median(),
    })
    out["collapse_phase"] = 100 * W.groupby(["method", "J", "phase"], observed=True).ess.apply(
        lambda x: (x < 2).mean()).unstack()
    out["ess_phase"] = W.groupby(["method", "J", "phase"], observed=True).essr.median().unstack()
    Y = W[W.obs == "Y"]
    out["collapse_run"] = 100 * Y.groupby(["method", "J", "runbin"], observed=True).ess.apply(
        lambda x: (x < 2).mean()).unstack()
    out["n_run"] = Y[(Y.method == "rbpl") & (Y.J == 200) & (Y.fs == 1)].groupby("runbin", observed=True).size()
    L = W[W.obs != "Y"]
    gl = L.groupby(["method", "J"])
    out["accuracy"] = pd.DataFrame({
        "RMSE": gl.err.apply(lambda x: np.sqrt((x ** 2).mean())),
        "RMSE short-CAE wks": L[L.obs == "Ytilde"].groupby(["method", "J"]).err.apply(lambda x: np.sqrt((x ** 2).mean())),
        "RMSE missing wks": L[L.obs == "none"].groupby(["method", "J"]).err.apply(lambda x: np.sqrt((x ** 2).mean())),
        "95% coverage": gl.cov95.mean(),
        "mean b_tilde": gl.btil.mean(),
        "overconfident (%)": 100 * gl.apply(lambda d: ((d.err.abs() > 0.5) & (d.btil < 0.1)).mean()),
    })
    mc = L.groupby(["method", "J", "uid", "seed", "k"]).bhat.std().groupby(["method", "J"]).mean()
    out["accuracy"]["MC sd of b_hat"] = mc
    out["cov_run"] = L.groupby(["method", "J", "runbin"], observed=True).cov95.mean().unstack()
    out["rmse_phase"] = L.groupby(["method", "J", "phase"], observed=True).err.apply(
        lambda x: np.sqrt((x ** 2).mean())).unstack()
    L = L.assign(agebin=pd.cut(L.age, [-1, 0, 3, 8, 16, 35], labels=["0", "1-3", "4-8", "9-16", ">16"]))
    out["paths"] = L.groupby(["method", "J", "agebin"], observed=True).paths.median().unstack()
    out["s2Y"] = R.groupby(["method", "J"]).s2Y.describe(percentiles=[0.1, 0.5])[["10%", "50%", "max"]]
    out["sec"] = R.groupby(["method", "J"]).sec.median()
    # Per-participant collapse frequency (J=200).
    pp = W[W.J == 200].groupby(["uid", "method"]).ess.apply(lambda x: 100 * (x < 2).mean()).unstack()
    out["per_participant"] = pp.sort_values("old", ascending=False)
    # True transition noise of the simulated CAE (AR(1) residual variance).
    tv = []
    for (u, s), d in W[(W.method == "rbpl") & (W.J == 200) & (W.fs == 1)].groupby(["uid", "seed"]):
        y = d.sort_values("k").truth.to_numpy()
        b = np.polyfit(y[:-1], y[1:], 1)
        tv.append(np.var(y[1:] - np.polyval(b, y[:-1])))
    out["true_ar1_var"] = float(np.median(tv))
    return out


def figures(W, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = {"old": "#c0392b", "rbpl": "#2471a3", "rbpl_obsEB": "#1e8449"}
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for m in ["old", "rbpl"]:
        d = W[(W.method == m) & (W.J == 200)].groupby("k").essr
        q = d.quantile([0.25, 0.5, 0.75]).unstack()
        ax[0].plot(q.index + 1, q[0.5], color=col[m], label=LABEL[m])
        ax[0].fill_between(q.index + 1, q[0.25], q[0.75], color=col[m], alpha=0.2)
        c = W[(W.method == m) & (W.J == 200)].groupby("k").ess.apply(lambda x: (x < 2).mean())
        ax[1].plot(c.index + 1, 100 * c, color=col[m], label=LABEL[m])
    ax[0].set(xlabel="study week w", ylabel="ESS / J", title="ESS / J (median, IQR), J = 200")
    ax[1].set(xlabel="study week w", ylabel="% of runs with ESS < 2", title="Collapse to one particle, J = 200")
    ax[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(Path(outdir) / "pf_ess_by_week.png", dpi=150)
    plt.close(fig)

    Y = W[(W.obs == "Y") & (W.J == 200)]
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    labs = RUN_LAB
    x = np.arange(len(labs))
    for i, m in enumerate(["old", "rbpl", "rbpl_obsEB"]):
        v = Y[Y.method == m].groupby("runbin", observed=False).ess.apply(lambda s: 100 * (s < 2).mean())
        ax.bar(x + (i - 1) * 0.27, v.reindex(labs).to_numpy(), width=0.27, color=col[m], label=LABEL[m])
    ax.set_xticks(x, labs)
    ax.set(xlabel="unobserved weeks before a full-survey week", ylabel="% with ESS < 2",
           title="Collapse when Y_w is observed, J = 200")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(Path(outdir) / "pf_collapse_by_gap.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="pf_cmp")
    ap.add_argument("--out", default="pf_cmp_summary")
    a = ap.parse_args()
    Path(a.out).mkdir(exist_ok=True)
    W, R = load(a.runs)
    T = tables(W, R)
    with open(Path(a.out) / "tables.txt", "w") as f:
        for k, v in T.items():
            f.write(f"== {k}\n{v.round(3).to_string() if hasattr(v, 'round') else v}\n\n")
    figures(W, a.out)
    print(open(Path(a.out) / "tables.txt").read())


if __name__ == "__main__":
    main()
