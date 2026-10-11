"""
Experiment 1: population behavioural composition and mixed-traffic
conflict exposure on a single-lane ring road.

Humans: IDM with perception delay tau_cf (constant-velocity gap
anticipation, after Treiber et al. 2006), Ornstein-Uhlenbeck acceleration
noise, parameters from the copula mapping layer.
AVs: deterministic IDM, no delay, no noise.
Disturbances: every 40 s a randomly chosen vehicle brakes at 4 m/s^2 for
up to 3 s (stopping-vehicle analogue).
All runs in a batch are simulated in parallel (vectorised over runs).
"""
import numpy as np
import time, json, sys
from bodd_core import Pop, POPS, map_params

S0, LV = 2.0, 4.5
TTC_STAR, DRAC_STAR = 1.5, 3.35
AV = dict(T=1.6, v0=14.5, a=1.2, b=2.0)
TGAP_MARGIN = 0.3
ADAPT = 0.0   # capability compensation: 0 = headway chosen independently of
              # control delay, 1 = headway fully set by delay (Gipps-like)


def simulate(specs, n=60, L=1800.0, dt=0.1, warm=120.0, dur=600.0,
             batch_seed=12345, keep_vehicle=False):
    R = len(specs)
    names = ["T", "v0", "a", "b", "tau_cf", "sig"]
    P = {k: np.zeros((R, n)) for k in names}
    theta = np.zeros((R, n)); low = np.zeros((R, n), bool)
    isav = np.zeros((R, n), bool)
    period, hold = int(40.0 / dt), int(3.0 / dt)
    steps, wsteps = int((warm + dur) / dt), int(warm / dt)
    n_ev = steps // period + 1
    pert = np.zeros((R, n_ev), int)

    for r, sp in enumerate(specs):
        rng = np.random.default_rng(sp["seed"])
        u, z = rng.random(n), rng.standard_normal(n)
        eps = {k: rng.standard_normal(n) for k in names}
        perm = rng.permutation(n)
        pert[r] = rng.integers(0, n, n_ev)
        th, lo = sp["pop"].sample(u, z)
        q = map_params(th, eps, sp.get("rho", 0.5), names)
        for k in names:
            P[k][r] = q[k]
        theta[r], low[r] = th, lo
        nav = int(round(sp["p_av"] * n))
        isav[r, perm[:nav]] = True

    # behavioural consistency (task-capability interface): humans do not
    # adopt time gaps shorter than their effective control delay + margin
    if ADAPT > 0:
        T_cap = 1.40 * P["tau_cf"] / 0.80        # headway proportional to own delay
        P["T"] = np.exp((1 - ADAPT) * np.log(P["T"]) + ADAPT * np.log(T_cap))
    P["T"] = np.maximum(P["T"], P["tau_cf"] + TGAP_MARGIN)
    for k, val in AV.items():
        P[k][isav] = val
    P["sig"][isav] = 0.0
    dsteps = np.where(isav, 0, np.round(P["tau_cf"] / dt).astype(int))
    K = int(dsteps.max()) + 1

    # per-run random streams: a run's noise depends only on its own seed, so
    # results are independent of batch composition and populations sharing a
    # seed share common random numbers in full
    nrng = [np.random.default_rng([sp["seed"], 777]) for sp in specs]
    BLK = 200
    noise = None
    lead = np.roll(np.arange(n), -1)
    wrap = np.zeros(n); wrap[-1] = L
    x = np.tile(np.arange(n) * (L / n), (R, 1))
    v = 10.0 + 0.3 * np.stack([g.standard_normal(n) for g in nrng])
    eta = np.zeros((R, n))
    phi = np.exp(-dt / 1.0)
    s_hist = np.zeros((K, R, n)); dv_hist = np.zeros((K, R, n))
    rows = np.arange(R)
    T, v0, amax, bb, sig = P["T"], P["v0"], P["a"], P["b"], P["sig"]
    sqab = 2.0 * np.sqrt(amax * bb)

    tet = np.zeros(R); tit = np.zeros(R); drac_t = np.zeros(R)
    events = np.zeros(R); coll = np.zeros(R); spd = np.zeros(R)
    tet_hh = np.zeros(R); tet_ha = np.zeros(R); tet_av = np.zeros(R)
    tet_veh = np.zeros((R, n)); coll_veh = np.zeros((R, n))
    E_prev = np.zeros((R, n), bool); C_prev = np.zeros((R, n), bool)
    hum = ~isav; av_lead = isav[:, lead]

    for k in range(steps):
        gap = x[:, lead] + wrap - x - LV
        dv = v - v[:, lead]
        slot = k % K
        s_hist[slot] = gap; dv_hist[slot] = dv
        dd = np.minimum(dsteps, k)
        idx = ((k - dd) % K)[None]
        s_del = np.take_along_axis(s_hist, idx, 0)[0]
        dv_del = np.take_along_axis(dv_hist, idx, 0)[0]
        s_est = np.maximum(s_del - dd * dt * dv_del, 0.1)
        s_star = S0 + np.maximum(0.0, v * T + v * dv_del / sqab)
        acc = amax * (1.0 - (v / v0) ** 4 - (s_star / s_est) ** 2)
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

        gap = x[:, lead] + wrap - x - LV
        C = gap <= 0.0
        if C.any():
            xl = (x[:, lead] + wrap)
            x = np.where(C, xl - LV - 0.05, x)
            v = np.where(C, np.minimum(v, v[:, lead]), v)
            gap = x[:, lead] + wrap - x - LV

        if k >= wsteps:
            dvn = v - v[:, lead]
            closing = dvn > 0.01
            ttc = np.where(closing, gap / np.maximum(dvn, 1e-6), np.inf)
            E = ttc < TTC_STAR
            tet += E.sum(1) * dt
            tit += np.where(E, TTC_STAR - ttc, 0.0).sum(1) * dt
            events += (E & ~E_prev).sum(1)
            E_prev = E
            drac = np.where(closing, dvn ** 2 / (2 * np.maximum(gap, 0.01)), 0.0)
            drac_t += (drac > DRAC_STAR).sum(1) * dt
            coll += (C & ~C_prev).sum(1)
            if keep_vehicle:
                coll_veh += (C & ~C_prev)
            C_prev = C
            tet_hh += (E & hum & ~av_lead).sum(1) * dt
            tet_ha += (E & hum & av_lead).sum(1) * dt
            tet_av += (E & isav).sum(1) * dt
            if keep_vehicle:
                tet_veh += E * dt
            spd += v.mean(1)

    vt = n * dur                     # vehicle-seconds measured
    vh = vt / 3600.0
    out = []
    for r, sp in enumerate(specs):
        d = dict(pop=sp["pop"].name, key=sp.get("key"), p_av=sp["p_av"],
                 seed=sp["seed"], rho=getattr(sp.get("rho", 0.5), "name", sp.get("rho", 0.5)),
                 tet_pct=100 * tet[r] / vt, tit_vh=tit[r] / vh,
                 events_vh=events[r] / vh, drac_pct=100 * drac_t[r] / vt,
                 coll_vh=coll[r] / vh, speed=spd[r] / (steps - wsteps),
                 sh_hh=tet_hh[r] / tet[r] if tet[r] > 0 else np.nan,
                 sh_ha=tet_ha[r] / tet[r] if tet[r] > 0 else np.nan,
                 sh_av=tet_av[r] / tet[r] if tet[r] > 0 else np.nan,
                 frac_av_lead_h=float((hum[r] & av_lead[r]).sum() / max(1, hum[r].sum())))
        if keep_vehicle:
            d["veh_theta"] = theta[r][hum[r]].tolist()
            d["veh_tet"] = tet_veh[r][hum[r]].tolist()
            d["veh_low"] = low[r][hum[r]].tolist()
            d["veh_T"] = P["T"][r][hum[r]].tolist()
            d["veh_tau"] = P["tau_cf"][r][hum[r]].tolist()
            d["veh_coll"] = coll_veh[r][hum[r]].tolist()
        out.append(d)
    return out


if __name__ == "__main__":
    # quick benchmark
    specs = [dict(pop=POPS[k], p_av=p, seed=s, key=k)
             for k in ["R", "D_mix"] for p in [0.0, 0.3] for s in range(20)]
    t0 = time.time()
    res = simulate(specs)
    print(f"{len(specs)} runs in {time.time()-t0:.1f} s")
    import collections
    agg = collections.defaultdict(list)
    for r in res:
        agg[(r["key"], r["p_av"])].append(r)
    for kk, rs in sorted(agg.items()):
        m = lambda f: np.nanmean([q[f] for q in rs])
        print(kk, f"ev/vh={m('events_vh'):.2f} tet%={m('tet_pct'):.3f} "
              f"tit={m('tit_vh'):.2f} drac%={m('drac_pct'):.4f} "
              f"coll/vh={m('coll_vh'):.3f} v={m('speed'):.2f}")
