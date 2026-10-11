"""Experiment 1 analysis: effect sizes with paired-bootstrap 95% CIs,
descriptor sufficiency with population-bootstrap CIs. Writes
results/exp1_summary.json and prints the tables used in the manuscript."""
import json, collections
import numpy as np
from scipy import stats

rng = np.random.default_rng(2026)
NB = 10000
CANON = ["R", "D_loc", "D_disp", "D_mix", "D_tail"]
L = lambda d: json.load(open(f"results/exp1_{d}.json"))
S = {}


def by(rows, *keys):
    g = collections.defaultdict(dict)
    for r in rows:
        g[tuple(r[k] for k in keys)][r["seed"]] = r
    return g


def ratio_ci(a, b):
    """ratio of means with paired bootstrap over seeds, plus paired t p-value."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = len(a); idx = rng.integers(0, n, (NB, n))
    bs = a[idx].mean(1) / b[idx].mean(1)
    p = stats.ttest_rel(a, b).pvalue if np.any(a != b) else 1.0
    return dict(r=float(a.mean() / b.mean()), lo=float(np.percentile(bs, 2.5)),
                hi=float(np.percentile(bs, 97.5)), p=float(p))


def paired(g, k1, k2, f="events_vh", **kw):
    ext = tuple(kw.values())
    A = g[(k1,) + ext]; B = g[(k2,) + ext]
    seeds = sorted(set(A) & set(B))
    return ratio_ci([A[s][f] for s in seeds], [B[s][f] for s in seeds])


def fmt(c):
    return f"{c['r']:.2f} [{c['lo']:.2f}, {c['hi']:.2f}]"


# ---- A: canonical table and penetration sweep -----------------------
A = L("A")["rows"]
g = by(A, "key", "p_av")
tab = {}
for p in [0.0, 0.3]:
    for k in CANON:
        rs = list(g[(k, p)].values())
        m = lambda f: float(np.mean([r[f] for r in rs]))
        tab[f"{k}|{p}"] = dict(ev=m("events_vh"), tet=m("tet_pct"), tit=m("tit_vh"),
                               drac=m("drac_pct"), coll=m("coll_vh"), v=m("speed"))
        if k != "R":
            tab[f"{k}|{p}"]["ratio"] = paired(g, k, "R", p_av=p)
S["table3"] = tab
sweep = {}
for k in CANON:
    for p in [0.0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8]:
        ev = [r["events_vh"] for r in g[(k, p)].values()]
        d = dict(mean=float(np.mean(ev)), se=float(np.std(ev, ddof=1) / np.sqrt(len(ev))))
        if k != "R":
            d["ratio"] = paired(g, k, "R", p_av=p)
        sweep[f"{k}|{p}"] = d
S["sweep"] = sweep
S["shape"] = {f"{k2}|{p}": paired(g, k2, "D_loc", p_av=p)
              for k2 in ["D_mix", "D_tail"] for p in [0.0, 0.3]}
red = {}
for k in CANON:
    b0 = tab[f"{k}|0.0"]["ev"]
    red[k] = {str(p): 1 - sweep[f"{k}|{p}"]["mean"] / b0 for p in [0.2, 0.3, 0.4]}
S["reduction"] = red

print("== Table 3 (events/veh-h, ratio vs R with 95% CI) ==")
for p in [0.0, 0.3]:
    for k in CANON:
        t = tab[f"{k}|{p}"]
        rr = fmt(t["ratio"]) + f" p={t['ratio']['p']:.1e}" if "ratio" in t else "1.00"
        print(f" p={p} {k:7s} ev={t['ev']:.2f} tet={t['tet']:.3f} tit={t['tit']:.2f} drac={t['drac']:.3f} "
              f"coll={t['coll']:.3f} v={t['v']:.2f}  ratio={rr}")
print("== penetration sweep ratios ==")
for k in CANON[1:]:
    print(" ", k, " ".join(f"{p}:{fmt(sweep[f'{k}|{p}']['ratio'])}(p={sweep[f'{k}|{p}']['ratio']['p']:.2g})"
                           for p in [0.1, 0.2, 0.4, 0.6, 0.8]))
print("== shape (vs D_loc) ==", {k: fmt(v) + f" p={v['p']:.2g}" for k, v in S["shape"].items()})
print("== reductions ==", {k: {p: round(v, 3) for p, v in d.items()} for k, d in red.items()})


# ---- B: descriptor sufficiency --------------------------------------
def loo_r2(X, y):
    n = len(y); X1 = np.column_stack([np.ones(n), X]); pr = np.zeros(n)
    H = X1 @ np.linalg.pinv(X1.T @ X1) @ X1.T
    b = np.linalg.lstsq(X1, y, rcond=None)[0]
    res = y - X1 @ b
    loo = res / (1 - np.diag(H))
    return 1 - np.sum(loo ** 2) / np.sum((y - y.mean()) ** 2), float(np.sqrt(np.mean(loo ** 2)))


def descriptor_analysis(design):
    D = L(design); meta = {m["key"]: m for m in D["meta"]}
    gg = collections.defaultdict(list)
    for r in D["rows"]:
        gg[r["key"]].append(r["events_vh"])
    ks = sorted(gg)
    y = np.log(np.array([np.mean(gg[k]) for k in ks]))
    F = {f: np.array([meta[k][f] for k in ks]) for f in ["mean", "sd", "skew", "tail15", "lpm1"]}
    sets = {"mean": [F["mean"]], "sd": [F["sd"]], "mean+sd": [F["mean"], F["sd"]],
            "mean+sd+skew": [F["mean"], F["sd"], F["skew"]], "tail15": [F["tail15"]],
            "log lpm1": [np.log(F["lpm1"])]}
    out = {}
    for nm, cols in sets.items():
        r2, rmse = loo_r2(np.column_stack(cols), y)
        out[nm] = dict(r2=float(r2), rmse=rmse)
    X = np.column_stack([np.ones(len(y)), F["mean"], F["sd"], F["skew"]])
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    s2 = np.sum((y - X @ b) ** 2) / (len(y) - 4)
    se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    # population bootstrap for coefficients and incremental R2
    n = len(y); bs_b = []; bs_inc = []
    for _ in range(4000):
        i = rng.integers(0, n, n)
        Xi, yi = X[i], y[i]
        bi = np.linalg.lstsq(Xi, yi, rcond=None)[0]; bs_b.append(bi)
        X1 = Xi[:, :2]; b1 = np.linalg.lstsq(X1, yi, rcond=None)[0]
        tss = np.sum((yi - yi.mean()) ** 2)
        bs_inc.append((np.sum((yi - X1 @ b1) ** 2) - np.sum((yi - Xi @ bi) ** 2)) / tss)
    bs_b = np.array(bs_b)
    X1 = X[:, :2]; b1 = np.linalg.lstsq(X1, y, rcond=None)[0]
    tss = np.sum((y - y.mean()) ** 2)
    inc = (np.sum((y - X1 @ b1) ** 2) - np.sum((y - X @ b) ** 2)) / tss
    Fst = (inc * tss / 2) / (np.sum((y - X @ b) ** 2) / (n - 4))
    out["coef"] = {nm: dict(b=float(b[j]), t=float(b[j] / se[j]),
                            lo=float(np.percentile(bs_b[:, j], 2.5)),
                            hi=float(np.percentile(bs_b[:, j], 97.5)))
                   for j, nm in enumerate(["const", "mean", "sd", "skew"])}
    out["inc_r2"] = dict(v=float(inc), lo=float(np.percentile(bs_inc, 2.5)),
                         hi=float(np.percentile(bs_inc, 97.5)), F=float(Fst),
                         p=float(1 - stats.f.cdf(Fst, 2, n - 4)))
    b_m = np.polyfit(F["mean"], y, 1)
    out["slope_mean_only"] = float(b_m[0])
    out["points"] = dict(mean=F["mean"].tolist(), sd=F["sd"].tolist(), y=y.tolist())
    return out


S["B"] = {}
for d in ["B0", "B3", "L_hetero_0", "L_sat_0", "L_tail_0", "L_hetero_3", "L_sat_3", "L_tail_3"]:
    S["B"][d] = descriptor_analysis(d)
    o = S["B"][d]
    print(f"== {d}: LOO R2 " + "  ".join(f"{k}={v['r2']:.3f}" for k, v in o.items()
                                        if isinstance(v, dict) and "r2" in v)
          + f" | b_mean={o['coef']['mean']['b']:.3f} [{o['coef']['mean']['lo']:.3f},{o['coef']['mean']['hi']:.3f}]"
          + f" b_sd={o['coef']['sd']['b']:.3f} [{o['coef']['sd']['lo']:.3f},{o['coef']['sd']['hi']:.3f}]"
          + f" incR2={o['inc_r2']['v']:.4f} [{o['inc_r2']['lo']:.4f},{o['inc_r2']['hi']:.4f}] F={o['inc_r2']['F']:.1f}"
          + f" rmseLOO(mean)={o['mean']['rmse']:.3f}")

# ---- alternative mappings, canonical ---------------------------------
S["M"] = {}
for mname in ["hetero", "sat", "tail"]:
    gm = by(L(f"M_{mname}")["rows"], "key", "p_av")
    S["M"][mname] = {f"{k}|{p}": paired(gm, k, "R", p_av=p)
                     for k in CANON[1:] for p in [0.0, 0.3]}
    S["M"][mname].update({f"shape_{k}|{p}": paired(gm, k, "D_loc", p_av=p)
                          for k in ["D_mix", "D_tail"] for p in [0.0, 0.3]})
    print(f"== mapping {mname}: ", {k: fmt(v) for k, v in S["M"][mname].items()})

# ---- C: rho -----------------------------------------------------------
gc = by(L("C")["rows"], "key", "rho", "p_av")
S["C"] = {f"{k}|{r}|{p}": paired(gc, k, "R", rho=r, p_av=p)
          for k in ["D_loc", "D_mix", "D_tail"] for r in [0.0, 0.25, 0.5, 0.75] for p in [0.0, 0.3]}
ident = all(gc[("D_loc", 0.0, 0.0)][s]["events_vh"] == gc[("R", 0.0, 0.0)][s]["events_vh"]
            for s in range(40))
S["rho0_identical"] = bool(ident)
print("== C rho:", {k: fmt(v) for k, v in S["C"].items() if "|0.0" in k[-4:] or True and k.startswith("D_loc")})
print("   rho=0 runs bitwise identical across populations:", ident)

# ---- D, E -------------------------------------------------------------
gd = by(L("D")["rows"], "key", "margin", "p_av")
S["D"] = {f"{k}|{mg}|{p}": paired(gd, k, "R", margin=mg, p_av=p)
          for k in ["D_loc", "D_mix"] for mg in [0.2, 0.4] for p in [0.0, 0.3]}
S["D_coll"] = {f"{k}|{mg}": float(np.mean([r["coll_vh"] for r in gd[(k, mg, 0.0)].values()]))
               for k in ["R", "D_loc", "D_mix"] for mg in [0.2, 0.4]}
ge = by(L("E")["rows"], "key", "n", "p_av")
S["E"] = {f"{k}|{n}|{p}": paired(ge, k, "R", n=n, p_av=p)
          for k in ["D_loc", "D_mix"] for n in [48, 72] for p in [0.0, 0.3]}
print("== D margin:", {k: fmt(v) for k, v in S["D"].items()})
print("== E density:", {k: fmt(v) for k, v in S["E"].items()})

# ---- F: attribution ------------------------------------------------------
S["F"] = {}
F_rows = L("F")["rows"]
for k in ["R", "D_mix", "D_tail"]:
    rs = [r for r in F_rows if r["key"] == k]
    th = np.concatenate([r["veh_theta"] for r in rs]); te = np.concatenate([r["veh_tet"] for r in rs])
    qs = np.quantile(th, [0.2, 0.4, 0.6, 0.8]); qi = np.digitize(th, qs)
    d = dict(quint=[float(te[qi == i].sum() / te.sum()) for i in range(5)],
             spearman=float(stats.spearmanr(th, te).correlation))
    if k != "R":
        low = np.concatenate([r["veh_low"] for r in rs]).astype(bool)
        # bootstrap over seeds for the exposure share of the low component
        per = [(np.asarray(r["veh_tet"])[np.asarray(r["veh_low"], bool)].sum(), np.sum(r["veh_tet"]))
               for r in rs]
        per = np.array(per); ii = rng.integers(0, len(per), (NB, len(per)))
        bsh = per[ii, 0].sum(1) / per[ii, 1].sum(1)
        d.update(low_pop=float(low.mean()), low_exp=float(te[low].sum() / te.sum()),
                 low_exp_lo=float(np.percentile(bsh, 2.5)), low_exp_hi=float(np.percentile(bsh, 97.5)))
    S["F"][k] = d
    print(f"== F {k}:", {kk: (np.round(v, 3).tolist() if isinstance(v, list) else round(v, 3)) for kk, v in d.items()})

json.dump(S, open("results/exp1_summary.json", "w"), indent=1)
print("saved results/exp1_summary.json")
