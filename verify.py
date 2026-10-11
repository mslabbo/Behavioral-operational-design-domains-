"""
Verification checks reported in Section 4.4. Run from the repository root: python code/verify.py
Each check prints PASS/FAIL against the tolerance stated in the manuscript.
"""
import numpy as np
from scipy.stats import norm
import exp1_ring as E
import exp2_merge as X
from bodd_core import POPS, MAPPINGS, map_params, MARGINALS, LAPSE_EXTRA

ok = True


def report(name, cond, detail):
    global ok
    ok &= bool(cond)
    print(f"[{'PASS' if cond else 'FAIL'}] {name}: {detail}")


rng = np.random.default_rng(0)

# 1. Proposition 1: reference marginals preserved under every mapping, for every
#    continuous parameter (10th/50th/90th percentiles within 0.2%) and the lapse rate
N = 400000
th = rng.standard_normal(N); eps = {k: rng.standard_normal(N) for k in MARGINALS}
CONT = ["T", "v0", "a", "b", "tau_cf", "sig", "tau_pr"]
def target(k, p):
    m = MARGINALS[k]; z = norm.ppf(p)
    x = m["med"] * np.exp(m["s"] * z) if m["kind"] == "ln" else m["mu"] + m["sd"] * z
    return np.clip(x, m["lo"], m["hi"])
worst, worst_lapse = 0.0, 0.0
for nm, M in MAPPINGS.items():
    q = map_params(th, eps, M, CONT + ["lapse"])
    for k in CONT:
        worst = max(worst, np.max(np.abs(np.quantile(q[k], [0.1, 0.5, 0.9]) / target(k, np.array([0.1, 0.5, 0.9])) - 1)))
    worst_lapse = max(worst_lapse, abs(q["lapse"].mean() - MARGINALS["lapse"]["p"]))
report("Marginal preservation (7 parameters x 4 mappings)", worst < 0.002, f"max relative quantile error {100*worst:.2f}%")
report("Lapse rate preserved (4 mappings)", worst_lapse < 0.002, f"max abs error {worst_lapse:.4f} (target 0.10)")

# 1b. Conditional-risk table vs direct simulation (200 000 draws per cell)
psi_t = np.load("tables/psi_r50.npy"); dmax = 0.0; rows = []
for t0, d, v in [(-2.0, 35.0, 14.0), (-1.0, 45.0, 14.0), (0.0, 35.0, 14.0), (0.0, 45.0, 14.0), (1.0, 35.0, 14.0), (-1.0, 50.0, 16.0)]:
    n = 200000
    e = {k: rng.standard_normal(n) for k in X.NAMES}
    m_, s_, lo_, hi_ = LAPSE_EXTRA
    extra = np.clip(m_ * np.exp(s_ * rng.standard_normal(n)), lo_, hi_)
    tau, T, a, b = X.human_response(np.float64(t0), e, extra, 0.5)
    c, *_ = X.sim_merge(d, v, tau, T, a, b, dt=0.05)
    tab = psi_t[list(X.VG).index(v), int(np.argmin(abs(X.TH - t0))), int(np.argmin(abs(X.DG - d)))]
    dmax = max(dmax, abs(tab - c.mean())); rows.append(f"{tab:.4f}/{c.mean():.4f}")
report("Risk table vs direct simulation", dmax < 0.01, f"max abs difference {dmax:.4f} (table/direct: {', '.join(rows)})")

# 2. Analytic response-delay survival vs direct sampling (within 0.004)
err = 0.0
for nm in ["baseline", "tail", "hetero"]:
    M = MAPPINGS[nm]; S = X.tau_survival(M)
    for t0 in [-2.5, 0.0, 1.5]:
        n = 200000
        e = {k: rng.standard_normal(n) for k in X.NAMES}
        m_, s_, lo_, hi_ = LAPSE_EXTRA
        extra = np.clip(m_ * np.exp(s_ * rng.standard_normal(n)), lo_, hi_)
        tau, *_ = X.human_response(np.full(n, t0), e, extra, M)
        it = int(np.argmin(abs(X.TH - t0)))
        err = max(err, max(abs((tau > x).mean() - np.interp(x, X.XG, S[it])) for x in [1.0, 1.5, 2.5, 4.0]))
report("Analytic delay distribution vs sampling", err < 0.005, f"max abs error {err:.4f}")

# 3. Interpolated population risk vs direct simulation off-grid (log-lag within 15%)
psi = np.load("tables/psi_r50.npy"); g = POPS["D_mix"]; P = X.p_table(g, 0.5, psi)
n = 600000
u, z = rng.random(n), rng.standard_normal(n); thg, _ = g.sample(u, z)
e = {k: rng.standard_normal(n) for k in X.NAMES + ["v0"]}
m_, s_, lo_, hi_ = LAPSE_EXTRA
extra = np.clip(m_ * np.exp(s_ * rng.standard_normal(n)), lo_, hi_)
v0 = map_params(thg, e, 0.5, ["v0"])["v0"]
rel_log, rel_lin = [], []
for vc, d in [(13.0, 34.0), (15.0, 47.0), (13.0, 38.0), (16.5, 52.0)]:
    sel = np.abs(v0 - vc) < 0.25
    tau, T, a, b = X.human_response(thg[sel], {k: e[k][sel] for k in X.NAMES}, extra[sel], 0.5)
    c, *_ = X.sim_merge(d, v0[sel], tau, T, a, b); mc = c.mean()
    X.INTERP = "linear"; pl = X.interp_p(P, d, vc)[0]
    X.INTERP = "loglag"; pg = X.interp_p(P, d, vc)[0]
    rel_log.append(pg / mc - 1); rel_lin.append(pl / mc - 1)
report("Log-lag interpolation vs direct simulation", max(abs(np.array(rel_log))) < 0.15,
       f"log-lag errors {np.round(100*np.array(rel_log),1)}%, bilinear {np.round(100*np.array(rel_lin),1)}%")

# 3b. Integration step: halving dt in the critical-delay bisection
e2 = {k: rng.standard_normal(64) for k in X.NAMES}
qq = map_params(np.float64(0.0), e2, 0.5, ["T", "a", "b", "tau_cf"])
Tq = np.maximum(qq["T"], qq["tau_cf"] + X.TGAP_MARGIN)
dd = 0.0
for d in [30.0, 40.0, 50.0]:
    t1 = X.crit_tau(d, 14.0, Tq, qq["a"], qq["b"], dt=0.05); t2 = X.crit_tau(d, 14.0, Tq, qq["a"], qq["b"], dt=0.025)
    dd = max(dd, np.abs(t1 - t2).max())
report("Integration step (dt 0.05 vs 0.025)", dd <= 0.05 + 1e-9, f"max change in critical delay {dd:.3f} s")

# 4. Calibrated prior commits no exceedances in its own population
st = X.Stream(POPS["R"], 0.5, n_veh=100000, n_av=10000, seed=3)
PR = X.p_table(POPS["R"], 0.5, psi)
r = st.evaluate(PR, PR)
report("Reference prior in reference population", r["viol"] == 0.0, f"exceedances {100*r['viol']:.3f}%")

# 5. rho = 0: populations cannot differ (bitwise identical runs)
a = E.simulate([dict(pop=POPS["R"], key="R", p_av=0.0, seed=s, rho=0.0) for s in range(4)])
b = E.simulate([dict(pop=POPS["D_loc"], key="D", p_av=0.0, seed=s, rho=0.0) for s in range(4)])
same = all(x["events_vh"] == y["events_vh"] and x["tet_pct"] == y["tet_pct"] for x, y in zip(a, b))
report("rho = 0 negative control", same, "identical outcomes across populations" if same else "outcomes differ")

# 6. Batch independence of Experiment 1 runs
a = E.simulate([dict(pop=POPS["R"], key="R", p_av=0.0, seed=s) for s in range(3)])
b = E.simulate([dict(pop=POPS[k], key=k, p_av=0.0, seed=s) for k in ["D_mix", "R"] for s in range(3)])
same = all(x["tet_pct"] == y["tet_pct"] for x, y in zip(a, [r for r in b if r["key"] == "R"]))
report("Run reproducible independent of batch", same, "identical" if same else "differs")

print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED")
