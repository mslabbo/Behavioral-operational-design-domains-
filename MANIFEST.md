# Manifest: manuscript item → producing script → result file

Scripts are in `code/`, result files in `results/`, risk tables in `tables/`, figures in `figures/`.
Run scripts from the repository root, e.g. `python code/exp1_summary.py`.

| Item | Content | Script (`code/`) | Result file (`results/`) |
|---|---|---|---|
| Table 1 | Reference marginals | `bodd_core.py` (`MARGINALS`) | — |
| Table 2 | Canonical populations and descriptors | `bodd_core.py` (`POPS`, `Pop.descriptors`) | — |
| Table 3 | Exp 1 canonical exposure, ratios with CIs | `exp1_designs.py A` → `exp1_summary.py` | `exp1_A.json`, `exp1_summary.json` (`table3`) |
| Table 4 | Exp 1 descriptor sufficiency | `exp1_designs.py B0 B3 L_*` → `exp1_summary.py` | `exp1_summary.json` (`B`) |
| Table 5 | Exp 2 priors, two-phenotype population | `exp2_final.py core` | `exp2_core.json` (`stream.D_mix`) |
| Risk tables | Conditional conflict probability, Exp 2 | `psi_build.py`, `psi_build_map.py` | `tables/psi_*.npy` |
| Table 6 | Sensitivity | `exp1_summary.py`, `mech_summary.py`, `exp2_final.py core maps` | `exp1_summary.json` (`C`,`D`,`E`,`M`), `mech_summary.json`, `exp2_core.json` (`alpha_sens`,`rho_sens`), `exp2_maps.json` |
| Fig. 1 | Framework schematic | `make_figures.py` (`fig1`) | — |
| Fig. 2 | Populations and mapping variants | `make_figures.py` (`fig2`) | — |
| Fig. 3 | Penetration sweep | `make_figures.py` (`fig3`) | `exp1_summary.json` (`sweep`) |
| Fig. 4 | Descriptor sufficiency, Exp 1 | `make_figures.py` (`fig4`) | `exp1_summary.json` (`B`) |
| Fig. 5 | Compensation sweep, Gipps, channels | `exp1_designs.py G H`, `gipps_ring.py` → `mech_summary.py` | `exp1_G.json`, `exp1_H.json`, `exp1_gipps.json`, `mech_summary.json` |
| Fig. 6 | Conditional and population risk | `exp2_final.py core` | `exp2_core.json` |
| Fig. 7 | Priors: safety–efficiency, exceedances | `exp2_final.py core` | `exp2_core.json` (`stream`) |
| Fig. 8 | Loading × location robustness grid | `exp2_final.py core` | `exp2_core.json` (`robust`) |
| Fig. 9 | Behavioural envelope | `exp2_final.py env` | `exp2_env.json` |
| Fig. 10 | Exp 2 across 100 populations | `exp2_final.py lhs` → `mech_summary.py` | `exp2_lhs.json`, `mech_summary.json` (`e2_lhs`) |
| Section 4.2 | Unconstrained-model contact diagnostic | `diag_unconstrained.py` | `diag_unconstrained.json` |
| Section 4.4 | Verification checks | `verify.py` | `verify_output.txt` |
| Appendix A | Gipps variants | `gipps_ring.py` | `exp1_gipps.json` (`variants`) |
| Appendix A | Supplementary results | as above | `exp1_summary.json` (`F`), `exp2_core.json`, `exp2_maps.json` |

Random seeds: Experiment 1 run *s* uses `numpy.random.default_rng(s)` for parameters and
disturbances and `default_rng([s, 777])` for noise; canonical designs use seeds 0–39 (Gipps variant parameters use `default_rng([s, 99])`),
Latin-hypercube designs 0–19, attribution 0–59; the population design uses
`scipy.stats.qmc.LatinHypercube(seed=7)`. Experiment 2: idiosyncratic draws
`eps_draws(seed=2024)`, traffic streams `Stream(seed=11)` (canonical) and `seed=11`
with 150 000 vehicles (Latin hypercube); bootstrap generators are seeded in each script.

Environment: Python 3.11.15, NumPy 2.4.4, SciPy 1.17.1, Matplotlib 3.10.9 (`requirements.txt`).
