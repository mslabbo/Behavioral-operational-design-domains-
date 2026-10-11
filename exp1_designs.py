"""
Experiment 1 designs. Usage (from the repository root):  python code/exp1_designs.py <design> ...
Each design writes results/exp1_<design>.json.  All runs use per-run random
streams, so any run is reproducible from its seed alone.

  A            penetration sweep, 5 canonical populations x 7 penetrations x 40 seeds
  B0, B3       100 Latin-hypercube populations x 20 seeds at 0% / 30% AV
  C            trait-behaviour loading rho in {0, 0.25, 0.5, 0.75}
  D            behavioural-consistency margin 0.2 / 0.4 s
  E            density 48 / 72 vehicles on the 1800 m ring
  F            vehicle-level attribution (per-driver exposure), 60 seeds
  M_<map>      canonical populations under an alternative mapping (hetero|sat|tail)
  L_<map>_<p>  Latin-hypercube populations under an alternative mapping, p in {0,3}
"""
import sys, os, json, time
import numpy as np
from scipy.stats import qmc
import exp1_ring as E
from bodd_core import Pop, POPS, MAPPINGS, Mapping

CANON = ["R", "D_loc", "D_disp", "D_mix", "D_tail"]
S40, S20 = range(40), range(20)


def lhs_pops():
    U = qmc.LatinHypercube(d=4, seed=7).random(100)
    pops = []
    for i, u in enumerate(U):
        m = -1.0 + 1.3 * u[0]
        sd = 0.7 + 0.7 * u[1]
        w = 0.05 + 0.45 * u[2]
        dmax = min(3.0, 0.95 / np.sqrt(w * (1 - w)))
        pops.append(Pop(f"L{i:03d}", m, sd, w, dmax * u[3]))
    return pops


def meta(pops):
    return [dict(key=p.name, w=p.w, delta=p.delta, **p.descriptors()) for p in pops]


def run(design):
    E.TGAP_MARGIN = 0.3
    out = {}
    if design == "A":
        specs = [dict(pop=POPS[k], key=k, p_av=p, seed=s) for k in CANON
                 for p in [0.0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8] for s in S40]
        out["rows"] = E.simulate(specs)
    elif design in ("B0", "B3"):
        pav = 0.0 if design == "B0" else 0.3
        pops = lhs_pops()
        out["rows"] = E.simulate([dict(pop=p, key=p.name, p_av=pav, seed=s)
                                  for p in pops for s in S20])
        out["meta"] = meta(pops)
    elif design == "C":
        specs = [dict(pop=POPS[k], key=k, p_av=p, seed=s, rho=r)
                 for k in ["R", "D_loc", "D_mix", "D_tail"]
                 for r in [0.0, 0.25, 0.5, 0.75] for p in [0.0, 0.3] for s in S40]
        out["rows"] = E.simulate(specs)
    elif design == "D":
        rows = []
        for mg in [0.2, 0.4]:
            E.TGAP_MARGIN = mg
            res = E.simulate([dict(pop=POPS[k], key=k, p_av=p, seed=s)
                              for k in ["R", "D_loc", "D_mix"] for p in [0.0, 0.3] for s in S40])
            for r in res:
                r["margin"] = mg
            rows += res
        E.TGAP_MARGIN = 0.3
        out["rows"] = rows
    elif design == "E":
        rows = []
        for n in [48, 72]:
            res = E.simulate([dict(pop=POPS[k], key=k, p_av=p, seed=s)
                              for k in ["R", "D_loc", "D_mix"] for p in [0.0, 0.3] for s in S40], n=n)
            for r in res:
                r["n"] = n
            rows += res
        out["rows"] = rows
    elif design == "F":
        out["rows"] = E.simulate([dict(pop=POPS[k], key=k, p_av=0.0, seed=s)
                                  for k in ["R", "D_mix", "D_tail"] for s in range(60)],
                                 keep_vehicle=True)
    elif design.startswith("M_"):
        M = MAPPINGS[design[2:]]
        out["rows"] = E.simulate([dict(pop=POPS[k], key=k, p_av=p, seed=s, rho=M)
                                  for k in CANON for p in [0.0, 0.3] for s in S40])
    elif design.startswith("L_"):
        _, mname, pp = design.split("_")
        M, pav = MAPPINGS[mname], 0.3 if pp == "3" else 0.0
        pops = lhs_pops()
        out["rows"] = E.simulate([dict(pop=p, key=p.name, p_av=pav, seed=s, rho=M)
                                  for p in pops for s in S20])
        out["meta"] = meta(pops)
    elif design == "G":           # capability compensation sweep
        rows = []
        for lam in [0.25, 0.5, 0.75, 1.0]:
            E.ADAPT = lam
            res = E.simulate([dict(pop=POPS[k], key=k, p_av=p, seed=s)
                              for k in ["R", "D_loc", "D_mix"] for p in [0.0, 0.3] for s in S40])
            for r in res:
                r["adapt"] = lam
            rows += res
        E.ADAPT = 0.0
        out["rows"] = rows
    elif design == "H":           # channel decomposition: trait reaches one channel only
        base = {k: 0.0 for k in ["T", "v0", "a", "b", "tau_cf", "sig", "tau_pr", "lapse"]}
        chans = {"choice_T": ["T"], "speed_v0a": ["v0", "a"], "braking_b": ["b"],
                 "capability_tau_sig": ["tau_cf", "sig"]}
        rows = []
        for cname, ks in chans.items():
            M = Mapping({**base, **{k: 0.5 for k in ks}}, "linear", cname)
            res = E.simulate([dict(pop=POPS[k], key=k, p_av=p, seed=s, rho=M)
                              for k in ["R", "D_loc"] for p in [0.0, 0.3] for s in S40])
            for r in res:
                r["channel"] = cname
            rows += res
        out["rows"] = rows
    else:
        raise SystemExit("unknown design " + design)
    os.makedirs("results", exist_ok=True)
    json.dump(out, open(f"results/exp1_{design}.json", "w"))


if __name__ == "__main__":
    for d in sys.argv[1:]:
        t0 = time.time()
        run(d)
        print(f"{d}: {time.time() - t0:.0f} s", flush=True)
