"""Compare the two CAE belief-state particle filters on the current testbed.

* ``old``  - particle learning with parameter draws: the filter implemented in
  ``algorithm_helpers.ParticleFilterRuntime.update_standard`` (and ported to
  ``RL_agents/rl/pf.py``). Each particle draws theta from its conjugate
  posterior, propagates Y_w from the transition (bootstrap proposal), and is
  weighted by the plug-in mediator likelihood.
* ``rbpl`` - Rao-Blackwellized particle learning, Algorithm
  ``alg:belief.state.rbpl`` of the manuscript: theta is integrated out of the
  mediator likelihood and of the transition (Gaussian predictive),
  short-CAE weeks use the locally optimal proposal, systematic resampling.

Both filters run on identical inputs: a random-policy (pi = 0.5) episode is
simulated once per (participant, seed); the policy does not depend on the
belief, so replaying the weekly PF packets offline reproduces the online
inputs exactly. Shared PF priors (``env_para_vanilla/rl_priors.json``) and the
online empirical-Bayes noise refit are used for both.

Usage::

    python diag_pf_compare.py --uids 13,18 --seeds 1,2 --J 50,200 --out pf_cmp
"""
from __future__ import annotations

import argparse
import json
import time
import types
from pathlib import Path

import os

import numpy as np

# Use the estimated PF priors (env_para_vanilla/rl_priors.json), as in the
# experiments and in deployment; est_prior (imported by the loader) only needs
# ADAPR_COMBINED_DIR to be set.
os.environ.setdefault("USE_ESTIMATED_PRIORS", "1")
os.environ.setdefault("ADAPR_COMBINED_DIR", str(Path(__file__).resolve().parent))

import algorithm_helpers as ah  # noqa: E402
import experiment as ex  # noqa: E402

LOG2PI = np.log(2.0 * np.pi)


def _mvn_logpdf(y, mean, cov):
    L = np.linalg.cholesky(cov)
    z = np.linalg.solve(L, y - mean)
    return float(-0.5 * (y.size * LOG2PI + z @ z) - np.sum(np.log(np.diag(L))))


def _systematic(w, rng):
    J = w.size
    u = (rng.random() + np.arange(J)) / J
    c = np.cumsum(w)
    c[-1] = 1.0
    return np.searchsorted(c, u)


class _RngProxy:
    """Delegates to a Generator and records the ancestor indices of ``choice``."""

    def __init__(self, rng):
        self._rng = rng
        self.last_choice = None

    def choice(self, *a, **kw):
        self.last_choice = self._rng.choice(*a, **kw)
        return self.last_choice

    def __getattr__(self, name):
        return getattr(self._rng, name)


class OldRuntime(ah.ParticleFilterRuntime):
    """Testbed filter, unchanged; only records the resampling ancestors."""

    def update_standard(self, k, packet, I_w):
        self.rng.last_choice = None
        super().update_standard(k, packet, I_w)
        g = self.rng.last_choice
        self.last_g = np.arange(self.J) if g is None else np.asarray(g)


class RBPLRuntime(ah.ParticleFilterRuntime):
    """Same storage/EB refit as the testbed runtime; RBPL weekly update."""

    def update_standard(self, k, packet, I_w):  # noqa: C901
        k = int(k)
        if k == 0:
            self.last_g = np.arange(self.J)
            return super().update_standard(k, packet, I_w)
        ag = self.agent
        pf = packet.pf_data
        J_w, Y_w, tY_w = ah.outcome_from_packet(packet, I_w)
        w_pf = k + 1
        sw_prev = w_pf - 2
        has_cumul = pf["X_cumul_MY_base"] is not None
        n_med = len(ag.nu_0_MY)
        if has_cumul and getattr(ag, "update_sigma2_online", True):
            self._update_sigma2_eb(pf, sw_prev, (self.v_hat @ self.y_hat).ravel())

        J = self.J
        log_w = np.log(np.maximum(self.v_hat, 1e-300))
        mt = np.zeros(J)   # propagation mean
        t2 = np.zeros(J)   # propagation variance
        for j in range(J):
            if has_cumul:
                (nu_MY, G_MY, nu_Y, G_Y, nu_t, G_t) = ah._per_particle_posterior(
                    j, sw_prev, self.y_hat, pf["cae_all_0"],
                    ag.nu_0_MY, ag.Gamma_0_MY, ag.sigma2_MY,
                    ag.nu_0_Y, ag.Gamma_0_Y, ag.sigma2_Y,
                    ag.nu_0_tilde_Y, ag.Gamma_0_tilde_Y, ag.sigma2_tilde_Y,
                    pf["X_cumul_MY_base"], pf["cae_delta_cumul_MY"],
                    pf["week_idx_cumul_MY"], pf["y_cumul_MY"],
                    pf["X_cumul_Y_base"], pf["cae_delta_cumul_Y"], pf["y_cumul_Y"],
                    pf["X_cumul_tY"], pf["cae_delta_cumul_tY"],
                    pf["week_idx_cumul_tY"], pf["y_cumul_tY"],
                )
            else:
                nu_MY, G_MY = list(ag.nu_0_MY), list(ag.Gamma_0_MY)
                nu_Y, G_Y = ag.nu_0_Y, ag.Gamma_0_Y
                nu_t, G_t = ag.nu_0_tilde_Y, ag.Gamma_0_tilde_Y
            cae_curr = ah._cae_by_sw(j, sw_prev, self.y_hat, pf["cae_all_0"])[sw_prev]
            # Rao-Blackwellized mediator likelihood (Eq. rb.mediator).
            for m in range(n_med):
                y = np.asarray(pf["M_Y_obs"][m], dtype=float).reshape(-1)
                X = (np.atleast_2d(pf["X_MY_base"][m])
                     + cae_curr * np.atleast_2d(pf["cae_delta_MY"][m]))
                mask = np.isfinite(y)
                if not mask.any():
                    continue
                Xo = X[mask]
                cov = ag.sigma2_MY[m] * np.eye(Xo.shape[0]) + Xo @ G_MY[m] @ Xo.T
                log_w[j] += _mvn_logpdf(y[mask], Xo @ nu_MY[m], cov)
            # Transition predictive.
            xY = ah._transition_feature_with_particle_mediators(pf["X_Y_base"], pf["cae_delta_Y"], cae_curr)
            mu = float(xY @ nu_Y)
            s2 = float(ag.sigma2_Y + xY @ G_Y @ xY)
            if J_w == 1 and I_w == 1:
                log_w[j] += -0.5 * (LOG2PI + np.log(s2) + (Y_w - mu) ** 2 / s2)
                mt[j], t2[j] = Y_w, 0.0
            elif J_w == 1:
                th = self.rng.multivariate_normal(nu_t, G_t)
                a = float(th @ np.asarray(pf["X_tY_base"], dtype=float))
                b = float(th @ np.asarray(pf["cae_delta_tY"], dtype=float))
                st2 = float(ag.sigma2_tilde_Y)
                v = st2 + b * b * s2
                log_w[j] += -0.5 * (LOG2PI + np.log(v) + (tY_w - a - b * mu) ** 2 / v)
                t2[j] = 1.0 / (1.0 / s2 + b * b / st2)
                mt[j] = t2[j] * (mu / s2 + b * (tY_w - a) / st2)
            else:
                mt[j], t2[j] = mu, s2

        log_w -= log_w.max()
        w = np.exp(log_w)
        w /= w.sum()
        ess = 1.0 / np.sum(w ** 2)
        self.ess_hist[k] = ess
        b_hat = float(w @ mt)
        b_var = float(w @ (t2 + mt ** 2) - b_hat ** 2)
        if ess < 0.5 * J:
            g = _systematic(w, self.rng)
            self.v_hat = np.full(J, 1.0 / J)
            self.resampled_hist[k] = True
        else:
            g = np.arange(J)
            self.v_hat = w.copy()
        self.last_g = g
        y_new = np.zeros((J, w_pf))
        y_new[:, :w_pf - 1] = self.y_hat[g]
        y_new[:, w_pf - 1] = mt[g] + np.sqrt(t2[g]) * self.rng.standard_normal(J)
        self.y_hat = y_new
        self.b_hat_hist[k] = b_hat
        self.b_tilde_hist[k] = np.sqrt(max(b_var, 0.0))
        self._record_standard_week(k)


class RBPLObservedEBRuntime(RBPLRuntime):
    """RBPL with the CAE-transition noise refit on observed transitions only.

    The default empirical-Bayes refit regresses the belief-mean path on its own
    lag. In unobserved weeks that path is the filter's prediction, so its
    residuals understate the transition noise and sigma2_Y shrinks. Here
    sigma2_Y is refit only on pairs (Y_{w-1}, Y_w) that were both observed
    (full survey completed); the mediator and short-CAE refits are unchanged.
    """

    MIN_ROWS = 3

    def update_standard(self, k, packet, I_w):
        if not hasattr(self, "_obsY"):
            self._obsY = {0: True}
        if int(k) >= 1:
            J_w, Y_w, _ = ah.outcome_from_packet(packet, I_w)
            self._obsY[int(k)] = bool(J_w == 1 and I_w == 1)
        return super().update_standard(k, packet, I_w)

    def _update_sigma2_eb(self, pf_data, sim_w_prev, ybar):
        s2_prev = float(self.agent.sigma2_Y)
        super()._update_sigma2_eb(pf_data, sim_w_prev, ybar)
        self.agent.sigma2_Y = s2_prev
        X_base = pf_data["X_cumul_Y_base"]
        if X_base is None or sim_w_prev <= 0:
            return
        keep = [i for i in range(sim_w_prev) if self._obsY.get(i, False) and self._obsY.get(i + 1, False)]
        if len(keep) < self.MIN_ROWS:
            return
        cae = np.empty(sim_w_prev + 1)
        cae[0] = pf_data["cae_all_0"]
        cae[1:] = ybar[1:sim_w_prev + 1]
        X = X_base + cae[:sim_w_prev][:, None] * pf_data["cae_delta_cumul_Y"]
        ag = self.agent
        ag.sigma2_Y = ah.empirical_bayes_sigma2(
            X[keep], ybar[1:sim_w_prev + 1][keep], ag.nu_0_Y, ag.Gamma_0_Y, ag.sigma2_Y)


def _agent(J, nweek, y1):
    return types.SimpleNamespace(
        W=nweek, J=J, Y_1=y1,
        nu_0_MY=[np.asarray(x, float) for x in ex.nu_0_MY],
        Gamma_0_MY=[np.asarray(x, float) for x in ex.Gamma_0_MY],
        sigma2_MY=np.asarray(ex.sigma2_MY, float).copy(),
        nu_0_Y=np.asarray(ex.nu_0_Y, float), Gamma_0_Y=np.asarray(ex.Gamma_0_Y, float),
        sigma2_Y=float(ex.sigma2_Y),
        nu_0_tilde_Y=np.asarray(ex.nu_0_tilde_Y, float),
        Gamma_0_tilde_Y=np.asarray(ex.Gamma_0_tilde_Y, float),
        sigma2_tilde_Y=float(ex.sigma2_tilde_Y),
        update_sigma2_online=True,
    )


def _distinct_rows(a):
    return int(np.unique(np.round(a, 12), axis=0).shape[0])


def run_filter(cls, packets, I_hist, J, nweek, y1, seed):
    ag = _agent(J, nweek, y1)
    ds = types.SimpleNamespace(b_hat_hist=np.full(nweek, np.nan),
                               b_tilde_hist=np.full(nweek, np.nan), pf_result=None)
    pf = cls(ag, ds, _RngProxy(np.random.default_rng(seed)))
    anc = np.full(nweek, np.nan)
    G = []
    for k in range(nweek):
        pf.update_standard(k, packets[k], int(I_hist[k]))
        G.append(pf.last_g)
        anc[k] = len(np.unique(pf.last_g))      # distinct ancestors kept at week k
    # Path degeneracy at the end of the episode: for each week c, the number of
    # distinct week-c ancestors among the final particles (backtracking).
    lin = np.arange(J)
    path_div = np.zeros(nweek, dtype=int)
    for c in range(nweek - 1, -1, -1):
        path_div[c] = len(np.unique(lin))
        lin = G[c][lin]
    return {
        "ess": pf.ess_hist.tolist(), "resampled": pf.resampled_hist.astype(int).tolist(),
        "ancestors": anc.tolist(), "b_hat": ds.b_hat_hist.tolist(), "b_tilde": ds.b_tilde_hist.tolist(),
        "path_distinct": path_div.tolist(),
        "y_final": pf.y_hat.tolist(), "v_final": np.asarray(pf.v_hat).tolist(),
        "sigma2_MY": [float(x) for x in ag.sigma2_MY], "sigma2_Y": float(ag.sigma2_Y),
    }


def simulate(uid, seed):
    res, oenv = ex.run_random_send(uid, seed=seed)
    nweek = oenv.nweek
    packets = [oenv.get_week_packet(k) for k in range(nweek)]
    I_hist = np.asarray(res["I"], dtype=int)
    truth = [float(oenv.CAE_all[0])] + [float(p.Y_prev) for p in packets[1:]]
    obs = ["Y"] + [("Y" if (p.J_w == 1 and I_hist[k] == 1) else "Ytilde" if p.J_w == 1 else "none")
                   for k, p in enumerate(packets) if k >= 1]
    return packets, I_hist, truth, obs, float(oenv.CAE_all[0]), nweek


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uids", required=True)
    ap.add_argument("--seeds", default="1")
    ap.add_argument("--J", default="50,200")
    ap.add_argument("--methods", default="old,rbpl")
    ap.add_argument("--fseeds", default="1", help="filter seeds (Monte Carlo replicates on the same data)")
    ap.add_argument("--out", default="pf_cmp")
    a = ap.parse_args()
    ex._ensure_priors_configured(None)
    out = Path(a.out)
    out.mkdir(exist_ok=True)
    cls = {"old": OldRuntime, "rbpl": RBPLRuntime, "rbpl_obsEB": RBPLObservedEBRuntime}
    for uid in [int(u) for u in a.uids.split(",")]:
        for seed in [int(s) for s in a.seeds.split(",")]:
            if all((out / f"{m}_J{J}_u{uid}_s{seed}_f{fs}.json").exists()
                   for m in a.methods.split(",") for J in a.J.split(",") for fs in a.fseeds.split(",")):
                continue
            packets, I_hist, truth, obs, y1, nweek = simulate(uid, seed)
            for J in [int(x) for x in a.J.split(",")]:
              for fs in [int(x) for x in a.fseeds.split(",")]:
                for meth in a.methods.split(","):
                    f = out / f"{meth}_J{J}_u{uid}_s{seed}_f{fs}.json"
                    if f.exists():
                        continue
                    t0 = time.time()
                    r = run_filter(cls[meth], packets, I_hist, J, nweek, y1, 1000 * fs + seed)
                    r.update(uid=uid, seed=seed, fseed=fs, J=J, method=meth, truth=truth, obs=obs,
                             prior_src=str(ex._priors_src),
                             seconds=round(time.time() - t0, 2))
                    f.write_text(json.dumps(r))
                    print(f"{meth} J={J} uid={uid} seed={seed}: {r['seconds']}s  "
                          f"median ESS/J={np.nanmedian(np.array(r['ess'][1:]) / J):.3f}", flush=True)


if __name__ == "__main__":
    main()
