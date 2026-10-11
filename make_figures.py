import os
import json, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from scipy.stats import norm
from bodd_core import POPS, MARGINALS, from_z, MAPPINGS

plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"],
                     "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
                     "legend.fontsize": 6.6, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                     "axes.linewidth": 0.7, "lines.linewidth": 1.2, "figure.dpi": 300})
CL = {"R": "#222222", "D_loc": "#C1442E", "D_disp": "#2E6FC1", "D_mix": "#1B7A52", "D_tail": "#8A5AB5"}
SH = {"R": "Reference", "D_loc": "Location", "D_disp": "Dispersion",
      "D_mix": "Two-phenotype", "D_tail": "Extreme subgroup"}
KEYS = ["R", "D_loc", "D_disp", "D_mix", "D_tail"]
J = lambda f: json.load(open(f"results/{f}.json"))
S1, MECH = J("exp1_summary"), J("mech_summary")
C2, ENV, LH = J("exp2_core"), J("exp2_env"), J("exp2_lhs")


def save(fig, name):
    path = os.path.join("figures", name)
    fig.savefig(path, bbox_inches="tight", dpi=300); plt.close(fig); print("wrote", path)


def fig1():
    fig, ax = plt.subplots(figsize=(7.0, 3.7))
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)
    ax.set_xlim(-0.12, 10.12); ax.set_ylim(0, 5.4); ax.axis("off")
    renderer = fig.canvas.get_renderer()
    inv = ax.transData.inverted()
    fit_report = []

    def fitted_text(xc, yc, s, size, maxw, **kw):
        """Place centred text, shrinking the font until it fits within maxw (data units)."""
        while True:
            t = ax.text(xc, yc, s, ha="center", va="center", fontsize=size, **kw)
            bb = t.get_window_extent(renderer=renderer)
            (x0, _), (x1, _) = inv.transform([(bb.x0, bb.y0), (bb.x1, bb.y1)])
            if x1 - x0 <= maxw or size <= 6.0:
                fit_report.append((s.split("\n")[0][:32], size, round(x1 - x0, 2), round(maxw, 2)))
                return
            t.remove(); size -= 0.1

    def width(s, size, **kw):
        t = ax.text(0, 0, s, fontsize=size, **kw)
        bb = t.get_window_extent(renderer=renderer); t.remove()
        (x0, _), (x1, _) = inv.transform([(bb.x0, bb.y0), (bb.x1, bb.y1)])
        return x1 - x0

    def box(x, y, w, h, t, b, fc, tsize=8.2):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06", fc=fc, ec="#333333", lw=0.8))
        fitted_text(x + w / 2, y + h - 0.26, t, tsize, w - 0.24, fontweight="bold")
        fitted_text(x + w / 2, y + h / 2 - 0.22, b, 6.9, w - 0.24)

    def arr(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=9, lw=0.9, color="#333333"))

    top = [("1. Measurement layer",
            "instrument + latent-trait model\ninvariance testing and linking\n\u2192 g(\u03b8) on the reference scale", "#EAF0F7"),
           ("2. Behavioural mapping layer",
            "any admissible H(q | \u03b8)\nhere: one-factor copula with\nexact reference marginals", "#EFF5EC"),
           ("3. Assessment layer",
            "risk functional R(\u03c0, g)\ncollective: simulation\nagent-level: linear in g", "#FBF0E6")]
    W, GAP, X0 = 3.1, 0.35, 0.0
    tsize = 8.2                       # one shared title size: largest at which all three fit
    while max(width(t, tsize, fontweight="bold") for t, _, _ in top) > W - 0.24:
        tsize -= 0.1
    for i, (t, b, fc) in enumerate(top):
        box(X0 + i * (W + GAP), 3.45, W, 1.65, t, b, fc, tsize=tsize)
    for i in range(2):
        xa = X0 + (i + 1) * W + i * GAP
        arr(xa + 0.02, 4.27, xa + GAP - 0.02, 4.27)
    box(0.0, 0.8, 4.2, 1.7, "B-ODD descriptor of a region",
        "d = D(g) = (\u03bc, \u03c3 [, shape bound])\nwith its linking uncertainty\n\u2014 what is measured of a population", "#F2F2F2")
    box(5.8, 0.8, 4.2, 1.7, "Behavioural envelope of an ADS",
        "E\u1d30(\u03b5) = { d : sup over g with D(g) = d\n of \u0394(\u03c0, g) \u2264 \u03b5 }\n\u2014 what the system was built to assume", "#F2F2F2")
    arr(X0 + W / 2, 3.43, X0 + W / 2, 2.56); arr(X0 + 2 * (W + GAP) + W / 2, 3.43, X0 + 2 * (W + GAP) + W / 2, 2.56)
    ax.annotate("", xy=(5.75, 1.6), xytext=(4.25, 1.6), arrowprops=dict(arrowstyle="<|-|>", lw=1.1, color="#B03A2E"))
    ax.text(5.0, 2.82, "behaviourally admissible  iff  D(g) ∈ Eᴰ(ε)", ha="center",
            fontsize=7.4, color="#B03A2E", fontweight="bold")
    ax.text(5.0, 0.3, "Conventional ODDs fix roadway, environment and traffic composition; the behavioural "
            "axis of the human population is unspecified.", fontsize=6.9, style="italic", ha="center")
    for r in fit_report:
        print("  fit:", r)
    print("  title size:", round(tsize, 1))
    save(fig, "fig1_framework.png")


def fig2():
    fig, axes = plt.subplots(1, 4, figsize=(7.6, 2.1))
    fig.subplots_adjust(wspace=0.62)
    th = np.linspace(-4.2, 3.2, 600)
    ax = axes[0]
    for k in KEYS:
        ax.plot(th, POPS[k].pdf(th), color=CL[k], label=SH[k], ls="--" if k == "D_disp" else "-")
    ax.set_xlabel("latent trait θ"); ax.set_ylabel("density"); ax.set_title("(a) Populations")
    ax.set_ylim(0, 0.95); ax.legend(frameon=False, loc="upper left", fontsize=4.6, handlelength=1.0, labelspacing=0.2)
    for ax, nm, lab, tt in ((axes[1], "T", "desired time gap T (s)", "(b) Time gap"),
                            (axes[2], "tau_pr", "perception–response (s)", "(c) Response time")):
        sg = MARGINALS[nm]["sign"]
        q = lambda p: from_z(nm, sg * 0.5 * th + np.sqrt(0.75) * norm.ppf(p))
        ax.fill_between(th, q(0.1), q(0.9), color="#999999", alpha=0.2)
        ax.plot(th, q(0.5), color="#333333"); ax.plot(th, q(0.1), ":", color="#333333", lw=0.8)
        ax.plot(th, q(0.9), ":", color="#333333", lw=0.8)
        ax.set_xlabel("latent trait θ"); ax.set_ylabel(lab); ax.set_title(tt)
    ax = axes[3]
    for mn, col, ls in (("baseline", "#222222", "-"), ("hetero", "#2E6FC1", "--"),
                        ("sat", "#1B7A52", "-."), ("tail", "#C1442E", ":")):
        M = MAPPINGS[mn]; r = M.rho("T")
        ax.plot(th, from_z("T", M.zstar(r * M.m(th), r)), color=col, ls=ls,
                label={"baseline": "baseline", "hetero": "heterogeneous ρ",
                       "sat": "saturating", "tail": "tail-accelerating"}[mn])
    ax.set_xlabel("latent trait θ"); ax.set_ylabel("median T (s)"); ax.set_title("(d) Mapping variants")
    ax.legend(frameon=False, fontsize=5.3, loc="upper left")
    save(fig, "fig2_populations.png")


def fig3():
    ps = [0.0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8]; x = np.array(ps) * 100
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6)); fig.subplots_adjust(wspace=0.3)
    for k in KEYS:
        m = np.array([S1["sweep"][f"{k}|{p}"]["mean"] for p in ps])
        se = np.array([S1["sweep"][f"{k}|{p}"]["se"] for p in ps])
        axes[0].errorbar(x, m, yerr=1.96 * se, color=CL[k], marker="o", ms=3, capsize=2, label=SH[k])
        if k != "R":
            r = np.array([S1["sweep"][f"{k}|{p}"]["ratio"]["r"] for p in ps])
            lo = np.array([S1["sweep"][f"{k}|{p}"]["ratio"]["lo"] for p in ps])
            hi = np.array([S1["sweep"][f"{k}|{p}"]["ratio"]["hi"] for p in ps])
            axes[1].plot(x, r, color=CL[k], marker="o", ms=3, label=SH[k])
            axes[1].fill_between(x, lo, hi, color=CL[k], alpha=0.10)
    axes[0].set_yscale("log"); axes[0].set_xlabel("automated-vehicle penetration (%)")
    axes[0].set_ylabel("conflict events per vehicle-hour"); axes[0].set_title("(a) Conflict exposure")
    axes[0].legend(frameon=False, fontsize=6.2)
    axes[1].axhline(1, color="#888888", lw=0.7, ls="--")
    axes[1].set_xlabel("automated-vehicle penetration (%)"); axes[1].set_ylabel("ratio to reference (95% CI)")
    axes[1].set_title("(b) Population gap under automation")
    save(fig, "fig3_penetration.png")


def fig4():
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.45)); fig.subplots_adjust(wspace=0.85)
    for ax, d, tt in ((axes[0], "B0", "(a) 0% AV"), (axes[1], "B3", "(b) 30% AV")):
        P = S1["B"][d]["points"]; mu, sd, y = map(np.array, (P["mean"], P["sd"], P["y"]))
        sc = ax.scatter(mu, y, c=sd, cmap="viridis", s=10, lw=0)
        b = np.polyfit(mu, y, 1); xs = np.linspace(mu.min(), mu.max(), 20)
        ax.plot(xs, np.polyval(b, xs), color="#C1442E", lw=1.0)
        ax.set_xlabel("population mean of θ"); ax.set_title(tt)
        if d == "B0":
            ax.set_ylabel("log conflict events / veh-h")
        cb = fig.colorbar(sc, ax=ax, pad=0.02); cb.set_label("population SD", fontsize=6.5); cb.ax.tick_params(labelsize=6)
    maps = [("baseline", "B0", "B3"), ("hetero", "L_hetero_0", "L_hetero_3"),
            ("sat", "L_sat_0", "L_sat_3"), ("tail", "L_tail_0", "L_tail_3")]
    xx = np.arange(len(maps)); w = 0.2
    for j, (lab, key, col, hat) in enumerate([("mean, 0% AV", 1, "#444444", None), ("mean+SD+skew, 0% AV", 1, "#9A9A9A", None),
                                              ("mean, 30% AV", 2, "#3A6EA5", None), ("mean+SD+skew, 30% AV", 2, "#9BB7D4", None)]):
        vals = [S1["B"][m[key]]["mean" if "skew" not in lab else "mean+sd+skew"]["r2"] for m in maps]
        axes[2].bar(xx + (j - 1.5) * w, vals, w, color=col, label=lab)
    axes[2].set_xticks(xx); axes[2].set_xticklabels(["base", "hetero", "satur.", "tail"], fontsize=6.5)
    axes[2].set_ylim(0.8, 1.0); axes[2].set_ylabel("leave-one-out $R^2$")
    axes[2].set_title("(c) Across mappings")
    axes[2].legend(frameon=False, fontsize=5.4, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2)
    save(fig, "fig4_descriptors.png")


def fig5():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6)); fig.subplots_adjust(wspace=0.75)
    lams = ["0.0", "0.25", "0.5", "0.75", "1.0"]; x = np.array([float(l) for l in lams])
    for k in ["D_loc", "D_mix"]:
        for p, ls in (("0.0", "-"), ("0.3", "--")):
            d = MECH["adapt"][f"{k}|{p}"]
            r = np.array([d[l][0] for l in lams]); lo = np.array([d[l][1] for l in lams]); hi = np.array([d[l][2] for l in lams])
            axes[0].plot(x, r, color=CL[k], ls=ls, marker="o", ms=3,
                         label=f"{SH[k]}, {'0' if p == '0.0' else '30'}% AV")
            axes[0].fill_between(x, lo, hi, color=CL[k], alpha=0.08)
    for k, mk in (("D_loc", "s"), ("D_mix", "D")):
        g = MECH["gipps"][f"{k}|0.0"]
        axes[0].errorbar(1.12, g[0], yerr=[[g[0] - g[1]], [g[2] - g[0]]], fmt=mk, color=CL[k], ms=4, capsize=2)
    axes[0].text(1.12, 0.55, "Gipps", ha="center", fontsize=6.5)
    axes[0].axhline(1, color="#888888", lw=0.7, ls="--"); axes[0].set_ylim(0.5, 2.5); axes[0].set_xlim(-0.05, 1.22)
    axes[0].set_xlabel("capability compensation λ"); axes[0].set_ylabel("conflict-exposure ratio to reference")
    axes[0].set_title("(a) Compensation sweep"); axes[0].legend(frameon=False, fontsize=5.4, loc="upper right")
    chans = [("choice_T", "headway choice T"), ("capability_tau_sig", "perception delay + noise"),
             ("speed_v0a", "desired speed + acceleration"), ("braking_b", "comfortable braking")]
    for j, (p, col) in enumerate((("0.0", "#444444"), ("0.3", "#9BB7D4"))):
        r = np.array([MECH["channels"][f"{c}|{p}"] for c, _ in chans])
        axes[1].barh(np.arange(len(chans)) + (j - 0.5) * 0.36, r[:, 0] - 1, 0.36, left=1, color=col,
                     xerr=[r[:, 0] - r[:, 1], r[:, 2] - r[:, 0]], capsize=2, error_kw=dict(lw=0.7),
                     label=f"{'0' if p == '0.0' else '30'}% AV")
    axes[1].set_yticks(range(len(chans))); axes[1].set_yticklabels([l for _, l in chans], fontsize=6.5)
    axes[1].invert_yaxis(); axes[1].axvline(1, color="#888888", lw=0.7)
    axes[1].set_xlabel("location-shift ratio, one channel"); axes[1].set_title("(b) Channel decomposition")
    axes[1].legend(frameon=False, fontsize=6)
    save(fig, "fig5_mechanism.png")


def fig6():
    TH, DG = np.array(C2["TH"]), np.array(C2["DG"])
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.4)); fig.subplots_adjust(wspace=0.45)
    for d, ls in (("35.0", "-"), ("45.0", "--")):
        axes[0].semilogy(TH, np.maximum(C2["psi_v14"][d], 1e-5), color="#222222", ls=ls, label=f"lag {d[:2]} m")
    axes[0].axhline(0.05, color="#C1442E", lw=0.8, ls=":"); axes[0].text(-4.7, 0.06, "α = 0.05", color="#C1442E", fontsize=6.5)
    axes[0].set_xlabel("follower trait θ"); axes[0].set_ylabel("conditional conflict probability")
    axes[0].set_title("(a) Conditional (v = 14 m/s)"); axes[0].legend(frameon=False)
    dR = C2["dstar"]["R"]["14.0"]
    for k in KEYS:
        axes[1].semilogy(DG, np.maximum(C2["Pcurve_v14"][k], 1e-5), color=CL[k], label=SH[k])
    axes[1].axhline(0.05, color="#C1442E", lw=0.8, ls=":"); axes[1].axvline(dR, color="#888888", lw=0.7, ls="--")
    axes[1].text(dR + 1.2, 0.15, f"reference AV\nboundary {dR:.0f} m", fontsize=6.2)
    axes[1].set_xlim(22, 70); axes[1].set_ylim(1e-4, 1)
    axes[1].set_xlabel("lag distance d (m)"); axes[1].set_ylabel("population conflict probability")
    axes[1].set_title("(b) Population"); axes[1].legend(frameon=False, fontsize=5.4, loc="lower left")
    m = (DG >= 26) & (DG <= 58); PR = np.array(C2["Pcurve_v14"]["R"])
    for k in KEYS[1:]:
        axes[2].plot(DG[m], np.array(C2["Pcurve_v14"][k])[m] / np.maximum(PR[m], 1e-12), color=CL[k], label=SH[k])
    axes[2].axhline(1, color="#888888", lw=0.7, ls="--"); axes[2].axvline(dR, color="#888888", lw=0.7, ls=":")
    axes[2].set_xlabel("lag distance d (m)"); axes[2].set_ylabel("true / reference-predicted")
    axes[2].set_title("(c) Miscalibration factor"); axes[2].legend(frameon=False, fontsize=5.8)
    save(fig, "fig6_riskcurves.png")


NICE = {"REF": "reference prior", "ORACLE": "B-ODD, correct", "MOMENT": "B-ODD, mean + SD",
        "RHO_LOW": "B-ODD, ρ under-set", "RHO_HIGH": "B-ODD, ρ over-set",
        "DM-0.50": "B-ODD, location −0.50", "DM-0.25": "B-ODD, location −0.25",
        "DM+0.25": "B-ODD, location +0.25", "DM+0.50": "B-ODD, location +0.50"}
ORDER = ["REF", "MOMENT", "RHO_LOW", "RHO_HIGH", "DM-0.50", "DM-0.25", "DM+0.25", "DM+0.50", "ORACLE"]


def fig7():
    s = C2["stream"]["D_mix"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9)); fig.subplots_adjust(wspace=0.8)
    for nm in ORDER:
        d = s[nm]; c = "#C1442E" if nm == "REF" else ("#1B7A52" if nm == "ORACLE" else "#3A6EA5")
        e = 1000 * np.array(d["exp_conf"])
        axes[0].errorbar(d["wait"][0], e[0], yerr=[[e[0] - e[1]], [e[2] - e[0]]], fmt="o", color=c, ms=4, capsize=2)
    lab = {"REF": "reference prior;\nlocation +0.50", "ORACLE": "correct; mean + SD",
           "RHO_LOW": "\u03c1 under-set", "DM+0.25": "location +0.25", "RHO_HIGH": "\u03c1 over-set",
           "DM-0.25": "location \u22120.25", "DM-0.50": "location \u22120.50"}
    off = {"REF": (6, -4), "ORACLE": (7, -3), "RHO_LOW": (7, 2), "DM+0.25": (7, -6),
           "RHO_HIGH": (-12, 8), "DM-0.25": (-58, -2), "DM-0.50": (-62, 2)}
    for nm, t in lab.items():
        d = s[nm]; axes[0].annotate(t, (d["wait"][0], 1000 * d["exp_conf"][0]), textcoords="offset points",
                                    xytext=off[nm], fontsize=5.9)
    axes[0].set_xlabel("mean waiting time at the stop line (s)")
    axes[0].set_ylabel("expected conflicts per 1000 merges")
    axes[0].set_title("(a) Safety–efficiency")
    v = np.array([s[n]["viol"] for n in ORDER]) * 100
    cols = ["#C1442E" if n == "REF" else ("#1B7A52" if n == "ORACLE" else "#3A6EA5") for n in ORDER]
    axes[1].barh(range(len(ORDER)), v[:, 0], color=cols, xerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]],
                 capsize=2, error_kw=dict(lw=0.7))
    axes[1].set_yticks(range(len(ORDER))); axes[1].set_yticklabels([NICE[n] for n in ORDER], fontsize=6.3)
    axes[1].invert_yaxis(); axes[1].set_xlabel("exceedances of 5% constraint (%)")
    axes[1].set_title("(b) Chance-constraint exceedances")
    save(fig, "fig7_priors.png")


def fig8():
    g = C2["robust"]; rhos = sorted({r["rho_av"] for r in g}); dms = sorted({r["dm"] for r in g})
    M = np.zeros((len(rhos), len(dms)))
    for r in g:
        M[rhos.index(r["rho_av"]), dms.index(r["dm"])] = 100 * r["viol"][0]
    ref = 100 * C2["stream"]["D_mix"]["REF"]["viol"][0]
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    im = ax.imshow(M, cmap="RdYlBu_r", aspect="auto", origin="lower", vmin=0, vmax=max(ref, M.max()))
    ax.set_xticks(range(len(dms))); ax.set_xticklabels([f"{d:+.2f}" for d in dms])
    ax.set_yticks(range(len(rhos))); ax.set_yticklabels([f"{r:.2f}" for r in rhos])
    for i in range(len(rhos)):
        for j in range(len(dms)):
            ax.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center", fontsize=6.4,
                    color="white" if M[i, j] > 0.6 * max(ref, M.max()) else "black")
    ax.set_xlabel("error in assumed population location (θ units)")
    ax.set_ylabel("assumed loading ρ"); ax.set_title(f"Exceedances (%) — reference prior = {ref:.1f}%")
    cb = fig.colorbar(im, ax=ax, pad=0.02); cb.set_label("exceedances (%)", fontsize=7)
    save(fig, "fig8_robustness.png")


def fig9():
    MS, SS = np.array(ENV["MS"]), np.array(ENV["SS"])
    fig, axes = plt.subplots(2, 2, figsize=(6.6, 5.0), sharex=True, sharey=True)
    fig.subplots_adjust(wspace=0.12, hspace=0.32, right=0.86)
    panels = [("ref_wc_M", "(a) Reference prior: marginal, worst shape"), ("ref_wc_C", "(b) Reference prior: conditional, worst shape"),
              ("mom_wc_M", "(c) Two-moment prior: marginal, worst shape"), ("mom_wc_C", "(d) Two-moment prior: conditional, worst shape")]
    for ax, (k, tt) in zip(axes.flat, panels):
        A = np.array(ENV[k])
        im = ax.pcolormesh(MS, SS, A, cmap="RdYlBu_r", shading="auto", vmin=0.5, vmax=3.0)
        cs = ax.contour(MS, SS, A, levels=[1.1, 1.25, 1.5, 2.0], colors="k", linewidths=0.6)
        ax.clabel(cs, fmt="%.2f", fontsize=5.5)
        for pk, mk in (("R", "o"), ("D_loc", "s"), ("D_disp", "^")):
            p = POPS[pk]; ax.plot(p.m, p.sd, mk, color="k", ms=4, mfc="white", mew=0.7)
        ax.set_title(tt, fontsize=7.2)
    for ax in axes[1]:
        ax.set_xlabel("population mean of θ")
    for ax in axes[:, 0]:
        ax.set_ylabel("population SD of θ")
    cax = fig.add_axes([0.88, 0.15, 0.02, 0.7]); cb = fig.colorbar(im, cax=cax)
    cb.set_label("calibration error Δ (true risk / α)", fontsize=7)
    save(fig, "fig9_envelope.png")


def fig10():
    meta = LH["meta"]; mean = np.array([m["mean"] for m in meta]); skew = np.array([m["skew"] for m in meta])
    vr, vm = 100 * np.array(LH["viol_ref"]), 100 * np.array(LH["viol_mom"])
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5)); fig.subplots_adjust(wspace=0.3)
    axes[0].scatter(mean, vr, s=10, color="#C1442E", label="reference prior", lw=0)
    axes[0].scatter(mean, vm, s=10, color="#3A6EA5", label="B-ODD, mean + SD", lw=0)
    axes[0].set_xlabel("population mean of θ"); axes[0].set_ylabel("exceedances (%)")
    axes[0].set_title("(a) 100 populations"); axes[0].legend(frameon=False)
    sc = axes[1].scatter(skew, vm, s=12, c=mean, cmap="viridis", lw=0)
    axes[1].set_xlabel("population skewness"); axes[1].set_ylabel("exceedances, two-moment prior (%)")
    axes[1].set_title(f"(b) Residual vs shape (r = {MECH['e2_lhs']['corr_vmom_skew']:.2f})")
    cb = fig.colorbar(sc, ax=axes[1], pad=0.02); cb.set_label("population mean", fontsize=6.5)
    save(fig, "fig10_lhs_calibration.png")


for f in (fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8, fig9, fig10):
    try:
        f()
    except Exception as e:
        import traceback; traceback.print_exc(); print("FAILED", f.__name__)
