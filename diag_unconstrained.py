"""Diagnostic reported in Section 4.2: contacts in the unconstrained model
(no admissibility constraint), reference population, no automation.
Run from the repository root:  python code/diag_unconstrained.py"""
import json
import numpy as np
import exp1_ring as E
from bodd_core import POPS


def main():
    E.TGAP_MARGIN = -1e9          # disables T >= tau + margin
    res = E.simulate([dict(pop=POPS["R"], key="R", p_av=0.0, seed=s) for s in range(40)],
                     keep_vehicle=True)
    coll_vh = np.mean([r["coll_vh"] for r in res])
    T = np.concatenate([r["veh_T"] for r in res])
    tau = np.concatenate([r["veh_tau"] for r in res])
    c = np.concatenate([r["veh_coll"] for r in res]) > 0
    out = dict(coll_vh=float(coll_vh), n_involved=int(c.sum()), n_drivers=int(len(c)),
               med_T_inv=float(np.median(T[c])), med_tau_inv=float(np.median(tau[c])),
               share_tau_gt_T=float(np.mean(tau[c] > T[c])),
               share_T_lt_tau_plus_03=float(np.mean(T[c] < tau[c] + 0.3)),
               pop_share_T_lt_tau_plus_03=float(np.mean(T < tau + 0.3)),
               pop_share_tau_gt_T=float(np.mean(tau > T)))
    json.dump(out, open("results/diag_unconstrained.json", "w"), indent=1)
    print(out)


if __name__ == "__main__":
    main()
