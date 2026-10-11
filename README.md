# Behavioural operational design domains: reproduction code

Code, saved results and precomputed tables for the manuscript

> **Behavioural operational design domains for automated driving systems: modelling population shift and safety calibration**
> Submitted to *Transportation Research Part C: Emerging Technologies*.

The study is computational: no empirical data are used. Two simulation experiments test how the behavioural composition of a human driver population affects (1) conflict exposure in mixed traffic and (2) the calibration of an automated vehicle's chance-constrained gap-acceptance decisions.

## Quick start

Requires Python 3.11 and three libraries (versions pinned in `requirements.txt`). Run every command from the repository root.

```bash
pip install -r requirements.txt

python reproduce.py figures      # regenerate all 10 figures from the saved results   (~10 s)
python reproduce.py verify       # run the 9 verification checks of Section 4.4        (~1 min)
python reproduce.py full         # rerun every simulation and analysis                  (~50 min)
python reproduce.py full --tables   # also rebuild the precomputed risk tables          (~90 min)
```

Times are for a 2-core machine. Every number in the manuscript can be checked against the files in `results/` without running anything.

## Repository layout

| Path | Contents |
|---|---|
| `code/` | All simulation, analysis, figure and verification scripts |
| `results/` | Saved outputs of every experiment (JSON) and the verification log |
| `tables/` | Precomputed conditional-risk tables for Experiment 2 (NumPy `.npy`) |
| `figures/` | The 10 manuscript figures, as produced by `code/make_figures.py` |
| `MANIFEST.md` | Each manuscript table, figure and section mapped to its script and result file |
| `reproduce.py` | One-command runner for the three levels above |

## Scripts

| Script | Role |
|---|---|
| `bodd_core.py` | Population models and the behavioural mapping layer (one-factor copula and its variants) |
| `exp1_ring.py` | Experiment 1 simulator: mixed traffic on a ring road (intelligent driver model with human-factor extensions) |
| `gipps_ring.py` | Experiment 1 with the Gipps (1981) car-following model, and the two variants in Appendix A |
| `exp1_designs.py` | Runs the Experiment 1 designs (`python code/exp1_designs.py A B0 ...`; see the file header) |
| `diag_unconstrained.py` | Contact diagnostic for the model without the admissibility constraint (Section 4.2) |
| `exp1_summary.py`, `mech_summary.py` | Effect sizes with bootstrap confidence intervals, regressions and mechanism statistics |
| `exp2_merge.py` | Experiment 2 model: automated gap acceptance with an explicit prior over the human response |
| `psi_build.py`, `psi_build_map.py` | Build the conditional-risk tables in `tables/` |
| `exp2_final.py` | Experiment 2 analyses: priors, robustness, behavioural envelope, descriptor sufficiency |
| `make_figures.py` | All figures |
| `verify.py` | The verification checks reported in Section 4.4 |

To run a single step, call its script from the root, for example `python code/exp2_final.py env`.

## Reproducibility

- Every random quantity is generated from fixed seeds; seeds are listed in `MANIFEST.md`. Each Experiment 1 run depends only on its own seed, so results do not change with how runs are batched.
- With the pinned library versions, rerunning a step reproduces its saved result file exactly.
- `results/verify_output.txt` is the expected output of `python reproduce.py verify`.

## Licence

Code: MIT (`LICENSE`). Results, tables and figures: CC BY 4.0 (`LICENSE-DATA`).

## Citation

See `CITATION.cff`. Please cite the article once published.
