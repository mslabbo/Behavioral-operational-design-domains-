"""
Alternative car-following model for Experiment 1: Gipps (1981), with
heterogeneous reaction times implemented as true delays (the speed applied at
t+dt is computed from the state observed tau earlier). Gipps has no desired
time-gap parameter: headway emerges from reaction time and braking, and the
model is collision-free when the follower's estimate of leader braking is not
exceeded. No behavioural-consistency constraint is imposed.

Mapping: tau = tau_cf (reaction time), a = a, desired maximum deceleration
b_G = 1.5 b (median 3.0 m/s^2), V = v0, b_hat = max(3.0, (b_G + 3)/2) as in
Gipps (1981). Main model "gipps1981": the trait reaches capability parameters
only. Two variants, reported in Appendix A: "bhat_trait", in which the trait
sets the assumed leader braking b_hat; and "margin", in which the safe-speed rule is planned over
tau + theta_s, where theta_s is a trait-mapped chosen safety margin (more
self-regulated drivers choose larger margins). The margin variant is the
Gipps analogue of the IDM desired time gap and preserves crash-freeness. Effective vehicle size 6.5 m.
AVs: tau = 0.5 s (no additional delay), a = 1.2, b_G = 3.0, b_hat = 4.0.
Humans carry the same trait-mapped Ornstein-Uhlenbeck acceleration noise as
in exp1_ring. Disturbances and conflict measures are identical.
"""
import numpy as np
from bodd_core import map_params
from exp1_ring import TTC_STAR, DRAC_STAR

LV, S_EFF = 4.5, 6.5


def simulate_gipps(specs, n=60, L=1800.0, dt=0.1, warm=120.0, dur=600.0, bhat_mode="gipps1981"):
    R = len(specs)
    names = ["v0", "a", "b", "tau_cf", "sig", "theta_s"]
    P = {k: np.zeros((R, n)) for k in names}
    isav = np.zeros((R, n), bool)
    period, hold = int(40.0 / dt), int(3.0 / dt)
    steps, wsteps = int((warm + dur) / dt), int(warm / dt)
    n_ev = steps // period + 1
    pert = np.zeros((R, n_ev), int)
    for r, sp in enumerate(specs):
        rng = np.random.default_rng(sp["seed"])
        u, z = rng.random(n), rng.standard_normal(n)
        eps = {k: rng.standard_normal(n) for k in ["T", "v0", "a", "b", "tau_cf", "sig", "theta_s"]}
        perm = rng.permutation(n)
        pert[r] = rng.integers(0, n, n_ev)
        th, _ = sp["pop"].sample(u, z)
        q = map_params(th, eps, sp.get("rho", 0.5), names)
        for k in names:
            P[k][r] = q[k]
        # variant-only parameter from a separate stream, so that the main
        # model's draws (parameters, AV placement, disturbances) are unchanged
        vr = np.random.default_rng([sp["seed"], 99])
        P.setdefault("bhat", np.zeros((R, n)))
        P["bhat"][r] = map_params(th, {"bhat": vr.standard_normal(n)}, sp.get("rho", 0.5), ["bhat"])["bhat"]
        isav[r, perm[:int(round(sp["p_av"] * n))]] = True

    tau = np.where(isav, 0.5, P["tau_cf"])
    amax = np.where(isav, 1.2, P["a"])
    bG = np.where(isav, 3.0, 1.5 * P["b"])
    bhat = np.where(isav, 4.0, np.maximum(3.0, (bG + 3.0) / 2))
    tplan = tau
    if bhat_mode == "margin":        # variant: trait also sets a chosen safety margin
        tplan = np.where(isav, tau, tau + P["theta_s"])
    elif bhat_mode == "bhat_trait":  # variant: trait sets the assumed leader braking
        bhat = np.where(isav, 4.0, P["bhat"])
    V = np.where(isav, 14.5, P["v0"])
    sig = np.where(isav, 0.0, P["sig"])
    phi = np.exp(-dt / 1.0); eta = np.zeros((R, n)); BLK = 200; noise = None
    dsteps = np.where(isav, 1, np.maximum(1, np.round(tau / dt).astype(int)))
    K = int(dsteps.max()) + 1

    nrng = [np.random.default_rng([sp["seed"], 777]) for sp in specs]
    lead = np.roll(np.arange(n), -1)
    wrap = np.zeros(n); wrap[-1] = L
    x = np.tile(np.arange(n) * (L / n), (R, 1))
    v = 10.0 + 0.3 * np.stack([g.standard_normal(n) for g in nrng])
    g_hist = np.zeros((K, R, n)); v_hist = np.zeros((K, R, n)); vl_hist = np.zeros((K, R, n))
    rows = np.arange(R)
    events = np.zeros(R); tet = np.zeros(R); coll = np.zeros(R); spd = np.zeros(R)
    drac_t = np.zeros(R)
    E_prev = np.zeros((R, n), bool); C_prev = np.zeros((R, n), bool)

    for k in range(steps):
        gap = x[:, lead] + wrap - x - (S_EFF - LV) - LV     # gap minus standstill margin
        slot = k % K
        g_hist[slot] = gap; v_hist[slot] = v; vl_hist[slot] = v[:, lead]
        dd = np.minimum(dsteps - 1, k)
        idx = ((k - dd) % K)[None]
        gd = np.take_along_axis(g_hist, idx, 0)[0]
        vd = np.take_along_axis(v_hist, idx, 0)[0]
        vld = np.take_along_axis(vl_hist, idx, 0)[0]
        v_acc = vd + 2.5 * amax * tau * (1 - vd / V) * np.sqrt(np.clip(0.025 + vd / V, 0, None))
        disc = (bG * tplan) ** 2 + bG * (2 * gd - vd * tplan + vld ** 2 / bhat)
        v_dec = -bG * tplan + np.sqrt(np.clip(disc, 0, None))
        v_tgt = np.maximum(np.minimum(v_acc, v_dec), 0.0)
        acc = (v_tgt - v) / dt
        if k % BLK == 0:
            noise = np.stack([g.standard_normal((BLK, n)) for g in nrng], axis=1)
        eta = eta * phi + sig * np.sqrt(1 - phi ** 2) * noise[k % BLK]
        acc = acc + eta
        if (k % period) < hold:
            pi = pert[:, k // period]
            forced = np.where(v[rows, pi] > 0.5, -4.0, 0.0)
            acc[rows, pi] = np.minimum(acc[rows, pi], forced)
        acc = np.clip(acc, -8.0, 3.0)
        vn = np.maximum(v + acc * dt, 0.0)
        x = x + 0.5 * (v + vn) * dt
        v = vn
        gp = x[:, lead] + wrap - x - LV
        C = gp <= 0.0
        if C.any():
            x = np.where(C, x[:, lead] + wrap - LV - 0.05, x)
            v = np.where(C, np.minimum(v, v[:, lead]), v)
            gp = x[:, lead] + wrap - x - LV
        if k >= wsteps:
            dvn = v - v[:, lead]
            closing = dvn > 0.01
            ttc = np.where(closing, gp / np.maximum(dvn, 1e-6), np.inf)
            E = ttc < TTC_STAR
            tet += E.sum(1) * dt
            events += (E & ~E_prev).sum(1); E_prev = E
            drac = np.where(closing, dvn ** 2 / (2 * np.maximum(gp, 0.01)), 0.0)
            drac_t += (drac > DRAC_STAR).sum(1) * dt
            coll += (C & ~C_prev).sum(1); C_prev = C
            spd += v.mean(1)
    vh = n * dur / 3600.0
    return [dict(key=sp.get("key"), p_av=sp["p_av"], seed=sp["seed"],
                 events_vh=events[r] / vh, tet_pct=100 * tet[r] / (n * dur),
                 drac_pct=100 * drac_t[r] / (n * dur), coll_vh=coll[r] / vh,
                 speed=spd[r] / (steps - wsteps)) for r, sp in enumerate(specs)]


if __name__ == "__main__":
    import json, os, time, sys
    from bodd_core import POPS
    t0 = time.time()
    seeds = range(int(sys.argv[1]) if len(sys.argv) > 1 else 40)
    specs = [dict(pop=POPS[k], key=k, p_av=p, seed=s)
             for k in ["R", "D_loc", "D_disp", "D_mix", "D_tail"] for p in [0.0, 0.3] for s in seeds]
    out = {"gipps1981": simulate_gipps(specs, bhat_mode="gipps1981")}
    vspecs = [dict(pop=POPS[k], key=k, p_av=0.0, seed=s) for k in ["R", "D_loc"] for s in range(8)]
    out["variants"] = {m: simulate_gipps(vspecs, bhat_mode=m) for m in ["margin", "bhat_trait"]}
    os.makedirs("results", exist_ok=True)
    json.dump(out, open("results/exp1_gipps.json", "w"))
    print(f"gipps: {len(specs)} runs {time.time()-t0:.0f}s")
