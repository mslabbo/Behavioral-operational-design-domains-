"""
Experiment 2 consolidated analysis.  Usage (from the repository root): python code/exp2_final.py <part> ...
Parts write results/exp2_<part>.json:
  core   canonical boundaries, risk inflation by speed, priors with paired
         bootstrap CIs, robustness grid, alpha and rho_true sensitivity
  env    behavioural envelope: normal-shape and worst-case-over-shapes, for
         the reference-calibrated system and for a two-moment B-ODD system
  lhs    descriptor sufficiency over the 100 Latin-hypercube populations
         (all four mappings) and REF vs two-moment violations (baseline)
  maps   canonical results under the alternative mappings
Risk inflation RIF(pi, g; v) is the true conflict probability at system pi's
decision boundary for approach speed v, divided by alpha. Two calibration
errors are reported:
  marginal    Delta_M = E_v[RIF] over g's own approach-speed distribution
              (calibration on average over the decisions the system makes)
  conditional Delta_C = max RIF over v in {12, 14, 16} m/s, the speed bins
              holding 94% of approaches in the reference population
              (calibration given an observed approach speed)
"""
import sys, json, os, time
import numpy as np
import exp2_merge as X
from bodd_core import POPS, Pop, MAPPINGS
from exp1_designs import lhs_pops

VS = [float(v) for v in X.VG[:-1]]
CANON = ["R", "D_loc", "D_disp", "D_mix", "D_tail"]
rng = np.random.default_rng(99)
NB = 4000


def psi_for(name):
    return np.load("tables/" + {"baseline": "psi_r50.npy", "r25": "psi_r25.npy", "r75": "psi_r75.npy"}.get(
        name, f"psi_{name}.npy"))


def mapping_for(name):
    return {"r25": 0.25, "r75": 0.75}.get(name, MAPPINGS.get(name, 0.5))


def boundaries(P):
    return {v: X.crit_lag(P, v) for v in VS}


def rif_at(Ptrue, bnd):
    return {v: float(X.interp_p(Ptrue, np.array([d]), np.array([v]))[0] / X.ALPHA)
            for v, d in bnd.items()}


from scipy.stats import norm as _norm
from bodd_core import MARGINALS as _MG, as_mapping as _asm
_EDGES = np.array([-np.inf, 11, 13, 15, 17, np.inf])


def speed_probs(g, mapping):
    """probability that an approaching driver's speed falls in each VS bin."""
    M = _asm(mapping); r = M.rho("v0"); sr = _MG["v0"]["sign"] * r
    zs = (_EDGES - _MG["v0"]["mu"]) / _MG["v0"]["sd"]
    z = np.array([M.zinv(np.clip(x, -8.9, 8.9), sr) if np.isfinite(x) else x for x in zs])
    mt = M.m(X.TF); sc = np.sqrt(1 - r ** 2)
    W = np.array([_norm.cdf((z[i + 1] - sr * mt) / sc) - _norm.cdf((z[i] - sr * mt) / sc)
                  for i in range(len(VS))])
    w = W @ g.pdf(X.TF)
    return w / w.sum()


def delta(Ptrue, bnd, g, mapping):
    r = rif_at(Ptrue, bnd); arr = np.array([r[v] for v in VS])
    return float(speed_probs(g, mapping) @ arr), float(max(r[12.0], r[14.0], r[16.0]))


def ci_rate(a):
    a = np.asarray(a, float); i = rng.integers(0, len(a), (NB, len(a)))
    bs = a[i].mean(1)
    return [float(a.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def ci_diff(a, b, ratio=False):
    a, b = np.asarray(a, float), np.asarray(b, float)
    i = rng.integers(0, len(a), (NB, len(a)))
    bs = a[i].mean(1) / b[i].mean(1) if ratio else a[i].mean(1) - b[i].mean(1)
    pt = a.mean() / b.mean() if ratio else a.mean() - b.mean()
    return [float(pt), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def priors_for(tp, mapping):
    pr = {"REF": (POPS["R"], mapping), "ORACLE": (tp, mapping),
          "MOMENT": (Pop("mom", tp.m, tp.sd), mapping)}
    return pr


def stream_block(tk, mapping, psi, extra_priors=None, rho_alt=None):
    tp = POPS[tk]
    st = X.Stream(tp, mapping, seed=11)
    Pt = X.p_table(tp, mapping, psi)
    pr = priors_for(tp, mapping)
    if extra_priors:
        pr.update(extra_priors)
    arr = {}
    for nm, (g, m) in pr.items():
        ps = psi if m is mapping else rho_alt[m]
        arr[nm] = st.evaluate(X.p_table(g, m, ps), Pt, arrays=True)
    out = {}
    for nm, a in arr.items():
        out[nm] = dict(viol=ci_rate(a["viol"]), conf=ci_rate(a["conf"]),
                       exp_conf=ci_rate(a["p_true"]), wait=ci_rate(a["wait"]),
                       coll=float(a["coll"].mean()),
                       low_share_conf=float((a["low"] & a["conf"]).sum() / max(1, a["conf"].sum())))
        if nm != "ORACLE":
            out[nm]["conf_vs_oracle"] = ci_diff(a["conf"], arr["ORACLE"]["conf"], ratio=True)
            out[nm]["exp_vs_oracle"] = ci_diff(a["p_true"], arr["ORACLE"]["p_true"], ratio=True)
            out[nm]["wait_vs_oracle"] = ci_diff(a["wait"], arr["ORACLE"]["wait"])
    return out


def part_core():
    out = {}
    PSI = {0.25: psi_for("r25"), 0.5: psi_for("baseline"), 0.75: psi_for("r75")}
    P = {k: X.p_table(POPS[k], 0.5, PSI[0.5]) for k in CANON}
    bR = boundaries(P["R"])
    out["dstar"] = {k: {str(v): float(d) for v, d in boundaries(P[k]).items()} for k in CANON}
    out["rif_ref"] = {k: {str(v): r for v, r in rif_at(P[k], bR).items()} for k in CANON}
    out["delta_ref"] = {k: delta(P[k], bR, POPS[k], 0.5) for k in CANON}
    out["speed_probs_R"] = speed_probs(POPS["R"], 0.5).tolist()
    iv = list(X.VG).index(14.0)
    out["psi_v14"] = {str(d): PSI[0.5][iv, :, int(np.argmin(abs(X.DG - d)))].tolist() for d in [35.0, 45.0]}
    out["TH"] = X.TH.tolist(); out["DG"] = X.DG.tolist()
    out["Pcurve_v14"] = {k: P[k][iv].tolist() for k in CANON}
    # priors, with location/loading errors for the main truth
    out["stream"] = {}
    for tk in ["R", "D_loc", "D_mix", "D_tail"]:
        tp = POPS[tk]
        extra = {"RHO_LOW": (tp, 0.25), "RHO_HIGH": (tp, 0.75)}
        if tk == "D_mix":
            for dm in [-0.5, -0.25, 0.25, 0.5]:
                extra[f"DM{dm:+.2f}"] = (tp.shifted(dm), 0.5)
        out["stream"][tk] = stream_block(tk, 0.5, PSI[0.5], extra, rho_alt=PSI)
        print(f"  stream {tk} done", flush=True)
    # robustness grid (truth D_mix, rho_true 0.5)
    tp = POPS["D_mix"]; st = X.Stream(tp, 0.5, seed=11); Pt = P["D_mix"]
    grid = []
    for r in [0.25, 0.5, 0.75]:
        for dm in [-0.5, -0.25, 0.0, 0.25, 0.5]:
            a = st.evaluate(X.p_table(tp.shifted(dm), r, PSI[r]), Pt, arrays=True)
            grid.append(dict(rho_av=r, dm=dm, viol=ci_rate(a["viol"]),
                             conf=float(a["conf"].mean()), wait=float(a["wait"].mean())))
    out["robust"] = grid
    # alpha sensitivity and rho_true sensitivity (Delta over speeds and RIF at 14 m/s)
    out["alpha_sens"] = []
    for al in [0.01, 0.05, 0.10]:
        b = {v: X.crit_lag(P["R"], v, alpha=al) for v in VS}
        out["alpha_sens"].append(dict(alpha=al, **{k: float(speed_probs(POPS[k], 0.5) @ np.array(
                                                          [float(X.interp_p(P[k], np.array([b[v]]), np.array([v]))[0]) / al for v in VS]))
                                                          for k in CANON},
                                      **{f"{k}_v14": float(X.interp_p(P[k], np.array([b[14.0]]), np.array([14.0]))[0]) / al
                                         for k in CANON}))
    out["rho_sens"] = []
    for rt, ps in PSI.items():
        Pr = {k: X.p_table(POPS[k], rt, ps) for k in CANON}
        b = boundaries(Pr["R"])
        out["rho_sens"].append(dict(rho=rt, **{k: delta(Pr[k], b, POPS[k], rt)[0] for k in CANON},
                                    **{f"{k}_v14": rif_at(Pr[k], b)[14.0] for k in CANON}))
    return out


def shapes(m, s):
    """two-component mixtures with overall mean m and SD s (feasible shapes)."""
    out = [Pop("n", m, s)]
    for w in [0.05, 0.10, 0.20, 0.30, 0.40, 0.50]:
        dmax = min(3.0, 0.97 / np.sqrt(w * (1 - w)))
        for d in np.linspace(0.5, dmax, 6):
            out.append(Pop("x", m, s, w, d))
    return out


def part_env():
    psi = psi_for("baseline")
    PR = X.p_table(POPS["R"], 0.5, psi); bR = boundaries(PR)
    MS = np.linspace(-1.25, 0.5, 15); SS = np.linspace(0.6, 1.6, 11)
    G = {f"{sys_}_{kind}_{agg}": np.zeros((len(SS), len(MS)))
         for sys_ in ["ref", "mom"] for kind in ["norm", "wc"] for agg in ["M", "C"]}
    for i, s in enumerate(SS):
        for j, m in enumerate(MS):
            gn = Pop("n", m, s); Pn = X.p_table(gn, 0.5, psi)
            bM = boundaries(Pn)                     # two-moment B-ODD system
            dr, dm_ = [], []
            for g in shapes(m, s):
                Pg = X.p_table(g, 0.5, psi)
                dr.append(delta(Pg, bR, g, 0.5)); dm_.append(delta(Pg, bM, g, 0.5))
            for sys_, d in (("ref", np.array(dr)), ("mom", np.array(dm_))):
                for a, agg in enumerate(["M", "C"]):
                    G[f"{sys_}_norm_{agg}"][i, j] = d[0, a]
                    G[f"{sys_}_wc_{agg}"][i, j] = d[:, a].max()
    return dict(MS=MS.tolist(), SS=SS.tolist(), **{k: v.tolist() for k, v in G.items()})


def part_lhs():
    pops = lhs_pops()
    out = {"meta": [dict(key=p.name, **p.descriptors()) for p in pops]}
    for mname in ["baseline", "hetero", "sat", "tail"]:
        try:
            psi = psi_for(mname)
        except FileNotFoundError:
            print("  missing table", mname); continue
        mp = mapping_for(mname)
        bR = boundaries(X.p_table(POPS["R"], mp, psi))
        d = [delta(X.p_table(p, mp, psi), bR, p, mp) for p in pops]
        out[mname + "_M"] = [x[0] for x in d]; out[mname + "_C"] = [x[1] for x in d]
        dm = []
        for p in pops:
            bM = boundaries(X.p_table(Pop("m", p.m, p.sd), mp, psi))
            dm.append(delta(X.p_table(p, mp, psi), bM, p, mp))
        out[mname + "_mom_M"] = [x[0] for x in dm]; out[mname + "_mom_C"] = [x[1] for x in dm]
        print(f"  lhs RIF {mname} done", flush=True)
    psi = psi_for("baseline")
    PR = X.p_table(POPS["R"], 0.5, psi)
    vr, vm, vo, cr, cm = [], [], [], [], []
    for p in pops:
        st = X.Stream(p, 0.5, n_veh=150000, n_av=15000, seed=11)
        Pt = X.p_table(p, 0.5, psi)
        a = st.evaluate(PR, Pt); b = st.evaluate(X.p_table(Pop("m", p.m, p.sd), 0.5, psi), Pt)
        vr.append(float(a["viol"])); vm.append(float(b["viol"]))
        cr.append(float(a["exp_true"])); cm.append(float(b["exp_true"]))
    out["viol_ref"], out["viol_mom"], out["exp_ref"], out["exp_mom"] = vr, vm, cr, cm
    return out


def part_maps():
    out = {}
    for mname in ["hetero", "sat", "tail"]:
        psi = psi_for(mname); mp = mapping_for(mname)
        P = {k: X.p_table(POPS[k], mp, psi) for k in CANON}
        bR = boundaries(P["R"])
        o = {"rif_v14": {k: rif_at(P[k], bR)[14.0] for k in CANON},
             "delta": {k: delta(P[k], bR, POPS[k], mp) for k in CANON}, "stream": {}}
        for tk in ["D_loc", "D_mix", "D_tail"]:
            o["stream"][tk] = stream_block(tk, mp, psi)
        out[mname] = o
        print(f"  maps {mname} done", flush=True)
    return out


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    for part in sys.argv[1:]:
        t0 = time.time()
        # reseed per part: results do not depend on which parts share a process
        rng = np.random.default_rng(99)
        res = {"core": part_core, "env": part_env, "lhs": part_lhs, "maps": part_maps}[part]()
        json.dump(res, open(f"results/exp2_{part}.json", "w"))
        print(f"{part}: {time.time()-t0:.0f}s", flush=True)
