"""
Experiment 2: AV entry into a human-driven major stream at a
stop-controlled junction, with a chance-constrained decision rule based on
an explicit prior over the approaching human driver's response.

Conditional risk psi(theta; d, v): probability that entering a lag of
distance d ahead of a human approaching at speed v produces a conflict
(min TTC < 1.5 s, follower deceleration > 3.35 m/s^2, or contact),
integrated over idiosyncratic variation given the driver's trait theta.

Population risk (linear functional of g):
  p_g(d, v) = int psi(theta; d, v) g(theta | v) dtheta,
  g(theta | v) proportional to g(theta) * l(v | theta)  (speed is observed
  and carries information about theta through the copula).
"""
import numpy as np, time
from scipy.stats import norm
from bodd_core import MARGINALS, map_params, POPS, Pop, LAPSE_EXTRA, as_mapping

S0 = 2.0
AV_U0, AV_A, AV_V0 = 1.5, 1.5, 13.9
DT, H = 0.05, 15.0
TTC_C, DEC_C = 1.5, 3.35
ALPHA = 0.05
TGAP_MARGIN = 0.3
TH = np.arange(-5.0, 3.0001, 0.25)
DG = np.arange(10.0, 90.001, 2.5)
VG = np.array([10.0, 12.0, 14.0, 16.0, 18.0, 20.0])
TF = np.linspace(-5.0, 3.0, 641)
M_EPS = 400
NAMES = ["T", "a", "b", "tau_pr", "tau_cf", "lapse"]


def sim_merge(d, v, tau, T, a, b, dt=DT, horizon=H):
    shp = np.broadcast_shapes(np.shape(d), np.shape(v), np.shape(tau), np.shape(T))
    s = np.array(np.broadcast_to(d, shp), float)
    vf = np.array(np.broadcast_to(v, shp), float)
    v0f = vf.copy()
    tau = np.broadcast_to(tau, shp); T = np.broadcast_to(T, shp)
    a = np.broadcast_to(a, shp); sqab = 2 * np.sqrt(a * np.broadcast_to(b, shp))
    u = np.full(shp, AV_U0)
    dv0 = vf - u
    minttc = np.where(dv0 > 0.01, s / np.maximum(dv0, 1e-6), np.inf)
    maxdrac = np.where(dv0 > 0.01, dv0 ** 2 / (2 * np.maximum(s, 0.01)), 0.0)
    maxdec = np.zeros(shp); coll = np.zeros(shp, bool)
    for k in range(int(horizon / dt)):
        t = k * dt
        au = AV_A * (1 - (u / AV_V0) ** 4)
        dvv = vf - u
        s_star = S0 + np.maximum(0.0, vf * T + vf * dvv / sqab)
        af = a * (1 - (vf / v0f) ** 4 - (s_star / np.maximum(s, 0.1)) ** 2)
        af = np.where(t >= tau, np.clip(af, -8.0, a), 0.0)
        un = u + au * dt
        vfn = np.maximum(vf + af * dt, 0.0)
        s = s + 0.5 * ((u + un) - (vf + vfn)) * dt
        u, vf = un, vfn
        dvv = vf - u
        ttc = np.where(dvv > 0.01, s / np.maximum(dvv, 1e-6), np.inf)
        minttc = np.minimum(minttc, np.where(s > 0, ttc, 0.0))
        drac = np.where(dvv > 0.01, dvv ** 2 / (2 * np.maximum(s, 0.01)), 0.0)
        maxdrac = np.maximum(maxdrac, drac)
        maxdec = np.maximum(maxdec, -af)
        coll |= s <= 0
        s = np.maximum(s, 0.0)
    conflict = (minttc < TTC_C) | (maxdrac > DEC_C) | coll
    return conflict, minttc, maxdrac, coll


def human_response(theta, eps, extra, rho):
    q = map_params(theta, eps, rho, NAMES)
    T = np.maximum(q["T"], q["tau_cf"] + TGAP_MARGIN)
    tau = q["tau_pr"] + q["lapse"] * extra
    return tau, T, q["a"], q["b"]


def eps_draws(seed=2024, M=M_EPS):
    rng = np.random.default_rng(seed)
    eps = {k: rng.standard_normal(M) for k in NAMES}
    m, s, lo, hi = LAPSE_EXTRA
    extra = np.clip(m * np.exp(s * rng.standard_normal(M)), lo, hi)
    return eps, extra


def crit_tau(d, v, T, a, b, dt=0.1, horizon=20.0, iters=15, hi0=14.0):
    """Largest response delay that still avoids a conflict (bisection).
    Conflict is monotone in the delay, so conflict <=> tau > tau*."""
    shp = np.broadcast_shapes(np.shape(d), np.shape(T))
    lo = np.full(shp, 0.02); hi = np.full(shp, hi0)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        c, *_ = sim_merge(d, v, mid, T, a, b, dt=dt, horizon=horizon)
        hi = np.where(c, mid, hi)
        lo = np.where(c, lo, mid)
    return 0.5 * (lo + hi)


XG = np.concatenate([np.linspace(0.02, 2.0, 200), np.linspace(2.02, 16.0, 300)])


def tau_survival(rho, n_quad=400):
    """S[itheta, ix] = P(response delay > XG[ix] | theta), analytic in the
    perception-response component, quadrature over the lapse delay.
    Valid for any Mapping (linear or nonlinear, scalar or per-parameter rho)."""
    M = as_mapping(rho)
    m = MARGINALS["tau_pr"]; lam = MARGINALS["lapse"]
    r_pr, r_la = M.rho("tau_pr"), M.rho("lapse")
    s_pr, s_la = m["sign"] * r_pr, lam["sign"] * r_la
    mt = M.m(TH)
    sc_pr = np.sqrt(1 - r_pr ** 2); sc_la = np.sqrt(1 - r_la ** 2)
    thr = M.zinv(norm.ppf(1 - lam["p"]), s_la)
    p_lapse = 1 - norm.cdf((thr - s_la * mt) / sc_la)
    med_e, s_e, lo_e, hi_e = LAPSE_EXTRA
    qs = (np.arange(n_quad) + 0.5) / n_quad
    extra = np.clip(med_e * np.exp(s_e * norm.ppf(qs)), lo_e, hi_e)

    def surv(x):                     # P(tau_pr > x | theta)
        c = np.log(np.clip(x, 1e-6, None) / m["med"]) / m["s"]
        cz = M.zinv(c, s_pr)
        s = 1 - norm.cdf((cz - s_pr * mt[:, None]) / sc_pr)
        return np.where(x < m["lo"], 1.0, np.where(x >= m["hi"], 0.0, s))

    base = surv(XG[None, :])
    conv = np.zeros_like(base)
    for e in extra:
        conv += surv(XG[None, :] - e)
    conv /= n_quad
    return (1 - p_lapse)[:, None] * base + p_lapse[:, None] * conv


def psi_table(rho, eps, extra=None, M=128, verbose=False, dt_ct=0.1):
    """psi[iv, itheta, id]; idiosyncratic T, a, b integrated by Monte Carlo,
    response delay integrated analytically."""
    S = tau_survival(rho)
    th = TH[:, None, None]
    e = {k: eps[k][None, None, :M] for k in ["T", "a", "b", "tau_cf"]}
    q = map_params(th, e, rho, ["T", "a", "b", "tau_cf"])
    T = np.maximum(q["T"], q["tau_cf"] + TGAP_MARGIN)
    a, b = q["a"], q["b"]
    out = np.zeros((len(VG), len(TH), len(DG)))
    for iv, v in enumerate(VG):
        ts = crit_tau(DG[None, :, None], v, T, a, b, dt=dt_ct)
        for it in range(len(TH)):
            out[iv, it] = np.interp(ts[it], XG, S[it]).mean(-1)
        if verbose:
            print(f"  v={v} done", flush=True)
    return out


# linear interpolation matrix from TH to TF
_W = np.zeros((len(TF), len(TH)))
for i, t in enumerate(TF):
    j = min(np.searchsorted(TH, t) - 1, len(TH) - 2); j = max(j, 0)
    f = (t - TH[j]) / (TH[j + 1] - TH[j])
    _W[i, j], _W[i, j + 1] = 1 - f, f


def v_weights(pop, rho, v):
    M = as_mapping(rho)
    r = M.rho("v0"); sr = MARGINALS["v0"]["sign"] * r
    zs = (v - MARGINALS["v0"]["mu"]) / MARGINALS["v0"]["sd"]
    z = M.zinv(zs, sr)
    like = norm.pdf((z - sr * M.m(TF)) / np.sqrt(1 - r ** 2))
    w = pop.pdf(TF) * like
    return w / w.sum()


def p_table(pop, rho, psi):
    """population conflict probability p[iv, id]."""
    P = np.zeros((len(VG), len(DG)))
    for iv, v in enumerate(VG):
        P[iv] = v_weights(pop, rho, v) @ (_W @ psi[iv])
    return P


INTERP = "loglag"


def interp_p(P, d, v):
    """Interpolate table P[iv, id] at arrays (d, v).
    "loglag" (default): interpolate log-probability, along distance at each grid
    speed, then across speed at a fixed time-lag d/v; risk decays roughly
    exponentially in lag and its scale moves with speed, so this is far more
    accurate than bilinear interpolation of P on (d, v).  "linear": bilinear."""
    d = np.atleast_1d(np.asarray(d, float)); v = np.atleast_1d(np.asarray(v, float))
    d, v = np.broadcast_arrays(d, v)
    v = np.clip(v, VG[0], VG[-1])
    iv = np.clip(np.searchsorted(VG, v) - 1, 0, len(VG) - 2)
    fv = (v - VG[iv]) / (VG[iv + 1] - VG[iv])
    if INTERP == "linear":
        dd = np.clip(d, DG[0], DG[-1])
        idd = np.clip(np.searchsorted(DG, dd) - 1, 0, len(DG) - 2)
        fd = (dd - DG[idd]) / (DG[idd + 1] - DG[idd])
        p00 = P[iv, idd]; p01 = P[iv, idd + 1]; p10 = P[iv + 1, idd]; p11 = P[iv + 1, idd + 1]
        return (1 - fv) * ((1 - fd) * p00 + fd * p01) + fv * ((1 - fd) * p10 + fd * p11)
    LP = np.log(np.clip(P, 1e-12, 1.0))
    t = d / v

    def at(j):
        dj = np.clip(t * VG[j], DG[0], DG[-1])
        k = np.clip(np.searchsorted(DG, dj) - 1, 0, len(DG) - 2)
        f = (dj - DG[k]) / (DG[k + 1] - DG[k])
        return (1 - f) * LP[j, k] + f * LP[j, k + 1]

    out = np.exp((1 - fv) * at(iv) + fv * at(iv + 1))
    return np.where(out <= 1.5e-12, 0.0, out)


def crit_lag(P, v, alpha=ALPHA, n=2000):
    dd = np.linspace(DG[0], DG[-1], n)
    p = interp_p(P, dd, np.full(n, v))
    ok = p <= alpha
    # smallest d such that all larger d are acceptable
    bad = np.where(~ok)[0]
    if len(bad) == 0:
        return dd[0]
    j = bad[-1] + 1
    return dd[min(j, n - 1)]


class Stream:
    """Major-stream vehicles from a true population; AV arrivals."""

    def __init__(self, pop, rho, n_veh=300000, n_av=30000, q=800, seed=7):
        rng = np.random.default_rng(seed)
        u, z = rng.random(n_veh), rng.standard_normal(n_veh)
        th, low = pop.sample(u, z)
        eps = {k: rng.standard_normal(n_veh) for k in NAMES + ["v0"]}
        m_, s_, lo_, hi_ = LAPSE_EXTRA
        extra = np.clip(m_ * np.exp(s_ * rng.standard_normal(n_veh)), lo_, hi_)
        self.tau, self.T, self.a, self.b = human_response(th, eps, extra, rho)
        self.v = np.clip(map_params(th, eps, rho, ["v0"])["v0"], VG[0], VG[-1])
        self.theta, self.low = th, low
        mean_h = 3600.0 / q
        self.h = 1.0 + rng.exponential(mean_h - 1.0, n_veh)
        self.j0 = rng.integers(0, n_veh - 500, n_av)
        self.lag0 = rng.random(n_av) * self.h[self.j0]
        self.cum = np.cumsum(self.h)

    def evaluate(self, P_prior, P_true, alpha=ALPHA, arrays=False):
        acc_full = interp_p(P_prior, self.v * self.h, self.v) <= alpha
        n = len(acc_full)
        idx = np.where(acc_full, np.arange(n), n - 1)
        nt = np.minimum.accumulate(idx[::-1])[::-1]     # next accepting index
        d0 = self.v[self.j0] * self.lag0
        acc_first = interp_p(P_prior, d0, self.v[self.j0]) <= alpha
        jstar = np.where(acc_first, self.j0, nt[np.minimum(self.j0 + 1, n - 1)])
        d_acc = np.where(acc_first, d0, self.v[jstar] * self.h[jstar])
        wait = np.where(acc_first, 0.0,
                        self.lag0 + self.cum[jstar - 1] - self.cum[self.j0])
        v_acc = self.v[jstar]
        p_true = interp_p(P_true, d_acc, v_acc)
        conf, minttc, maxdrac, coll = sim_merge(d_acc, v_acc, self.tau[jstar],
                                               self.T[jstar], self.a[jstar], self.b[jstar])
        if arrays:
            return dict(conf=conf, viol=p_true > alpha, wait=wait, p_true=p_true,
                        coll=coll, low=self.low[jstar])
        return dict(conf_rate=conf.mean(), coll_rate=coll.mean(),
                    viol=(p_true > alpha).mean(), exp_true=p_true.mean(),
                    wait_mean=wait.mean(), wait_p95=np.percentile(wait, 95),
                    low_share_conf=(self.low[jstar] & conf).sum() / max(1, conf.sum()),
                    low_share_acc=self.low[jstar].mean())


if __name__ == "__main__":
    eps, extra = eps_draws()
    t0 = time.time()
    psi = np.zeros((len(VG), len(TH), len(DG)))
    th = TH[:, None, None]
    e = {k: val[None, None, :] for k, val in eps.items()}
    tau, T, a, b = human_response(th, e, extra[None, None, :], 0.5)
    c, *_ = sim_merge(DG[None, :, None], 14.0, tau, T, a, b)
    print("one (rho,v) slice:", time.time() - t0, "s")
    ps = c.mean(-1)
    for it in [0, 8, 12, 16, 20, 24, 28]:
        print(f"theta={TH[it]:+.2f}", np.round(ps[it, ::4], 3))
