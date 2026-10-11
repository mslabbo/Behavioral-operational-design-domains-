"""Mechanism statistics for Experiment 1 (capability compensation sweep,
channel decomposition, Gipps model) and auxiliary Experiment 2 statistics.
Writes results/mech_summary.json."""
import json
import numpy as np

rng = np.random.default_rng(7)
NB = 10000
L = lambda f: json.load(open(f"results/{f}.json"))


def rc(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    i = rng.integers(0, len(a), (NB, len(a)))
    bs = a[i].mean(1) / b[i].mean(1)
    return [float(a.mean() / b.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def pair(rows, k, kr="R", f="events_vh", **kw):
    sel = lambda kk: {r["seed"]: r[f] for r in rows if r["key"] == kk and all(r[a] == b for a, b in kw.items())}
    A, B = sel(k), sel(kr); s = sorted(set(A) & set(B))
    return rc([A[x] for x in s], [B[x] for x in s])


out = {}
# compensation sweep: lambda = 0 from design A
A = L("exp1_A")["rows"]; G = L("exp1_G")["rows"]
out["adapt"] = {}
for p in [0.0, 0.3]:
    for k in ["D_loc", "D_mix"]:
        out["adapt"][f"{k}|{p}"] = {"0.0": pair(A, k, p_av=p)}
        for lam in [0.25, 0.5, 0.75, 1.0]:
            out["adapt"][f"{k}|{p}"][str(lam)] = pair(G, k, adapt=lam, p_av=p)
out["adapt_R_ev"] = {str(lam): float(np.mean([r["events_vh"] for r in G if r["key"] == "R" and r["adapt"] == lam and r["p_av"] == 0.0]))
                     for lam in [0.25, 0.5, 0.75, 1.0]}
# channels
H = L("exp1_H")["rows"]
out["channels"] = {f"{c}|{p}": pair(H, "D_loc", channel=c, p_av=p)
                   for c in ["choice_T", "speed_v0a", "braking_b", "capability_tau_sig"] for p in [0.0, 0.3]}
# Gipps
GP = L("exp1_gipps")["gipps1981"]
out["gipps"] = {f"{k}|{p}": pair(GP, k, p_av=p) for k in ["D_loc", "D_disp", "D_mix", "D_tail"] for p in [0.0, 0.3]}
out["gipps_R_ev"] = {str(p): float(np.mean([r["events_vh"] for r in GP if r["key"] == "R" and r["p_av"] == p])) for p in [0.0, 0.3]}
out["gipps_coll"] = float(np.mean([r["coll_vh"] for r in GP]))
# Experiment 2 auxiliaries
LH = L("exp2_lhs"); meta = LH["meta"]
skew = np.array([m["skew"] for m in meta]); mean = np.array([m["mean"] for m in meta])
vm = np.array(LH["viol_mom"]); vr = np.array(LH["viol_ref"])
out["e2_lhs"] = dict(
    viol_ref=[float(np.median(vr)), float(np.percentile(vr, 10)), float(np.percentile(vr, 90)), float(vr.max())],
    viol_mom=[float(np.median(vm)), float(np.percentile(vm, 10)), float(np.percentile(vm, 90)), float(vm.max())],
    corr_vmom_skew=float(np.corrcoef(vm, skew)[0, 1]),
    corr_vmom_mean=float(np.corrcoef(vm, mean)[0, 1]),
    corr_momC_skew=float(np.corrcoef(np.array(LH["baseline_mom_C"]), skew)[0, 1]),
    frac_vmom_below_05=float((vm < 0.005).mean()),
    viol_mom_skew_lt_m05=float(np.median(vm[skew < -0.5])), n_skew_lt_m05=int((skew < -0.5).sum()),
    viol_mom_skew_ge_m05=float(np.median(vm[skew >= -0.5])),
    exp_ratio_median=float(np.median(np.array(LH["exp_mom"]) / np.array(LH["exp_ref"]))))
for m in ["baseline", "hetero", "sat", "tail"]:
    for agg in ["M", "C"]:
        x = np.array(LH[f"{m}_mom_{agg}"])
        out["e2_lhs"][f"{m}_mom_{agg}"] = [float(np.median(x)), float(np.percentile(x, 90)), float(x.max())]
json.dump(out, open("results/mech_summary.json", "w"), indent=1)
for k, v in out.items():
    print(k, v if not isinstance(v, dict) else {kk: (np.round(vv, 3).tolist() if isinstance(vv, list) else vv) for kk, vv in v.items()})
