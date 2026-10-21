# Activity process for the simulator: internal gaps and dropout (spec, 2026-10-20)

Two transition models on daily activity. Exogenous: no latent PU, no action terms, so the
process is independent of the policy and censoring by it stays ignorable. Fitted on the
observed daily table; used by the simulator as a daily state machine; the 4b emission
likelihood is restricted to active days so the two do not double count.

## 0. Data

Source: `docs/dropout_check_20261017/participant_dropout_daily.csv`
(ParticipantIdentifier, study_day, Date, observed, pageview_any, wear_any, active_any).

- The 47 modelled participants only (`in_modelled_47` in participant_dropout.csv): the simulator
  represents this cohort, so the activity process is fitted on it. The 8 high-CAE exclusions and
  the 9 still-enrolled starters are not used. Study day 1 = first local EOD task date. Window = days 1..84.
- `observed = 0` (after the extract date) is censoring, never inactivity.
- `active_d = pageview_any OR wear_any` (wear = HR>0 or steps>0 on the local day).
- A gap = maximal run of consecutive inactive observed days. Gap i of person p starts on
  day s_pi (first inactive day). Its length is t_pi if an active day follows inside the
  observed window (delta_pi = 1); otherwise it is censored at
  c_pi = (last observed day in window) - s_pi + 1 (delta_pi = 0).
- Day 1 is active by construction (first EOD task). A person's state on day d is
  active, inactive, or (after extract) unobserved.

Covariates available on every day d, computable in the simulator from its own history:

| name | definition |
|---|---|
| `week_d` | `vani_env.study_week_norm(ceil(d/7), n_weeks=12)` = (week - 6.5)/5.5, in [-1, 1]; the simulator passes its own horizon (n_weeks=36) as for the other models |
| `act7_d` | fraction of the previous 7 observed days (d-7..d-1) that were active; days before day 1 count as active; if fewer than 7 observed days use the available ones |
| `prior_gap_d` | 1 if at least one gap was completed before day d, else 0 (shared by both models) |

## 1. Model 1: active -> inactive (gap start)

Discrete-time hazard, person-period logistic regression (Allison 1982; Singer & Willett 2003
ch. 10-11).

- Risk set: every observed active day d with day d+1 observed and d+1 <= 84 (about 3580 active
  person-days; 18 of the 47 have at least one gap, 29 none).
- Outcome: y_d = 1 if day d+1 is inactive.
- Model:
    logit lambda_d = a0 + a1 week_d + a2 act7_d + a3 prior_gap_d
- Estimation: maximum likelihood (logistic regression). Standard errors cluster-robust by
  participant (repeated events within person; Cook & Lawless 2007 ch. 3). No frailty.
- Also fit the intercept-only model (lambda_bar) and report both.
- Model check: compare fitted vs observed gap-start rates by week (1..12)
  a calibration table with deciles of fitted lambda.
- Why logistic: everyone can start another gap, no cure needed; logistic hazard is the
  standard discrete-time form and takes covariates directly.

## 2. Model 2: inactive -> active, with a permanent fraction (promotion-time cure)

Unit = one gap. Promotion-time (bounded cumulative hazard) cure model
(Yakovlev et al. 1996; Chen, Ibrahim & Sinha 1999), discrete time.

- Kernel: discrete Weibull (Nakagawa & Osaki 1975) on gap length,
    F(d) = 1 - exp(-(d/mu)^alpha),  d = 0,1,2,...   (F(0)=0)
- Survival of the gap (probability the gap has not ended by the end of day d):
    S(d) = exp(-theta F(d)),   S(inf) = exp(-theta) = pi  (permanent fraction)
- Daily return hazard: h(d) = 1 - S(d)/S(d-1) = 1 - exp(-theta [F(d) - F(d-1)]).
- Likelihood per gap:
    delta = 1 (returned after t days):  S(t-1) - S(t)
    delta = 0 (censored after c days):  S(c)
  Log-likelihood summed over the 56 gaps of the 47 modelled users (47 completed; 9 censored at
  the window end: lengths 1, 1, 2, 9, 12, 22, 23, 27, 76). See identification_km.png in docs/activity_process_20261020
  for the KM curves: the gap-length curve plateaus after ~16 days (pi identified); the
  time-to-first-gap curve does not reach a plateau within 84 days (a stayer fraction for
  model 1 would be identified only through the parametric shape, hence no cure in model 1).
- Covariates on theta, as is standard for promotion-time models (Chen, Ibrahim & Sinha 1999);
  the kernel F is a baseline discrete Weibull (mu, alpha) without covariates:
    log theta = c0 + c1 prior_gap_s + c2 week_s   (s = gap start day; week_s = study_week_norm).
  Parameters (c0, c1, c2, log mu, log alpha). Proportional hazards: h(d|x) = theta(x) f(d), so
  c1, c2 are informed by the return speed of all 47 completed gaps, not only by the plateau;
  the price is that return speed and permanent fraction pi(x) = exp(-theta(x)) move together.
  Diagnostics: fitted S(d|x) vs KM stratified by prior_gap and by early/late start (weeks 1-6
  vs 7-12). Note the clear dropouts sit in weeks 2, 9, 9, 9.
- Estimation: numerical ML (scipy L-BFGS on (c0, c1, c2, log mu, log alpha)) from several
  starts. Report point estimates with Hessian SEs, noting they assume independent gaps and are
  therefore too small (56 gaps from 18 users). No bootstrap or profile CI: the simulator uses
  point estimates only.
- Checks:
  - Kaplan-Meier of gap length (censoring at window end / extract date) overlaid with the
    fitted S(d); the KM plateau should sit near pi.
  - Maller & Zhou (1994, JASA) sufficient-follow-up statistic: t_n = largest observed gap time
    (censored included), t*_n = largest uncensored gap time, N_n = number of uncensored gaps
    in (2 t*_n - t_n, t*_n], q_n = N_n / n. Report q_n with and without 112's gap (descriptive only, no refit; expected ~1/56 with,
    ~0.8 without: 112 is the single late return that sets the plateau height).
    Also Maller & Zhou (1992, Biometrika): is S_KM(t_n) significantly > 0 (immunes present).
- Derived quantities to report:
  - pi(x) at prior_gap = 0/1 and early/late week; median gap length among returners; h(d) for d = 1..30.
  - For each censored terminal gap: P(permanent | censored at c) = pi / S(c).
    Expected number of dropouts among the 47 users = sum of these; compare to the
    hard-threshold count 4.
  - Implied daily dropout rate lambda_bar * pi; compare to the current 0.1050 %/day.
- Why promotion-time: the permanent fraction is an explicit parameter; one
  standard survival likelihood; no latent indicator to integrate over.

## 3. Simulator mapping

Mixture representation of the fitted model 2 (exact):
    theta = exp(c0 + c1 prior_gap + c2 week_start),  pi = exp(-theta),
    S_u(d) = (exp(-theta F(d)) - exp(-theta)) / (1 - exp(-theta)),
    h_u(d) = 1 - S_u(d)/S_u(d-1).

Daily state machine. Observed states are only {active, inactive(d)}; 'dropped' below is the
mixture-representation latent B = 1 (a spell that never ends), booked as the existing absorbing
flag so the agent-freeze machinery is reused. Separate RNG stream
(SeedSequence child, as for the current hazard). The path is drawn once up front
for the episode, plus 2 days so the last Sunday's Monday and Tuesday are known;
the day loop only reads it:

1. active day d: compute covariates from the simulated history; draw Bernoulli(lambda_d).
   On success: draw B ~ Bernoulli(pi). B = 1 -> dropped on day d+1 (absorbing; existing
   dropout machinery, dropout_day = d+1, agent freeze as in (C)). B = 0 -> inactive with
   gap-day counter 1 on day d+1.
2. inactive day with counter k: draw Bernoulli(h_u(k)); success -> active next day
   (prior_gap = 1 from then on); else counter k+1.
3. inactive and dropped days: presence indicators are 0 both as truth and as observed (the system
   sees a real zero; E_hat and the PU weekly means use it). On inactive day d the forced zero is
   FW for the row of day d-1 (next-morning wear); PJ forced 0 on day d; page views and
   ws_interaction forced 0 on day d. J_w is a forced zero only if its whole completion window is
   inactive (the Sunday slot through Sunday plus the maximum of observed completion date minus
   that Sunday; 2 days in this extract). U1 and U2 follow J_w. Content variables (steps / 4SC /
   prior-2h / today steps, anticipated affect, self-reported active_status (its own morning
   model, not gated by PJ), U1, U2, CAE_short, full CAE) keep their latent draw under A = 0 and
   are NaN as observed. Active days: all streams drawn as now, no truncation; the existing gates
   (PJ for the daily survey, J_w and I_w for the weekly one) are reused.
4. Setting lambda = 0 reproduces the current simulator bit-for-bit (regression test).
   lambda = 1, pi = 1 -> everyone dropped on day 2 (replaces the h = 1 test).
5. Per-episode log: dropout_day, n_gaps, inactive_days, learned_weeks.
6. Calibration check (2000 episodes, 3 participants): gaps per person, gap-length quantiles,
   fraction of inactive days, fraction with >= 14-day terminal gap by day 84, fraction
   permanently gone by day 252; compare with the data and with 10,000 paths simulated from
   the fitted process alone.

Parameters stored in `generator_calibration.json` under `activity_process`:
model 1 coefficients and names; model 2 (c0, c1, c2, mu, alpha); covariate definitions;
source file and date. Existing `dropout_*` keys kept for comparison.

## 4. 4b change (goes into the cluster refit)

Each emission is gated on the date it refers to. PV and PJ on day d enter only if day d is
active. FW is `nextday_wearing` stored on the row of day d-1, so that row enters only if day d
is active. J_w, and U1/U2 on the same check-in, are a forced zero only when every day in the
completion window is inactive.
Weekly PV/FW/PJ means that enter the E_w transition are unchanged (fixed denominators,
inactive days count 0). Log the number of excluded person-days. Script 5 unchanged: its
outcomes are already NaN when the device is not worn or the survey is not answered.

## 5. References

- Allison, P. D. (1982). Discrete-time methods for the analysis of event histories.
  Sociological Methodology 13, 61-98.
- Singer, J. D. & Willett, J. B. (2003). Applied Longitudinal Data Analysis, ch. 10-12.
- Cook, R. J. & Lawless, J. F. (2007). The Statistical Analysis of Recurrent Events.
- Yakovlev, A. Y. & Tsodikov, A. D. (1996). Stochastic Models of Tumor Latency and Their
  Biostatistical Applications; Chen, M.-H., Ibrahim, J. G. & Sinha, D. (1999). A new Bayesian
  model for survival data with a surviving fraction. JASA 94, 909-919.
- Nakagawa, T. & Osaki, S. (1975). The discrete Weibull distribution. IEEE Trans. Reliability.
- Farewell, V. T. (1982). Mixture models for survival data with long-term survivors.
  Biometrics 38, 1041-1046 (mixture-cure alternative, not used).
- Amico, M. & Van Keilegom, I. (2018). Cure models in survival analysis. Annual Review of
  Statistics and Its Application 5, 311-342.
