"""Reproduce the results of the B-ODD manuscript. Run from the repository root.

    python reproduce.py figures          regenerate all 10 figures from the saved results   (~10 s)
    python reproduce.py verify           run the 9 verification checks of Section 4.4        (~1 min)
    python reproduce.py full             rerun every simulation and analysis, then figures
                                         and verification                                    (~50 min)
    python reproduce.py full --tables    as full, and also rebuild the risk tables           (~90 min)

Times are for a 2-core machine. Every step writes its outputs to results/,
tables/ or figures/, so steps can also be run one at a time (see README.md).
"""
import subprocess
import sys
import time

PY = sys.executable

EXP1 = [["code/exp1_designs.py", "A", "B0", "B3", "C", "D", "E", "F", "G", "H"],
        ["code/exp1_designs.py", "M_hetero", "M_sat", "M_tail",
         "L_hetero_0", "L_sat_0", "L_tail_0", "L_hetero_3", "L_sat_3", "L_tail_3"],
        ["code/gipps_ring.py", "40"],
        ["code/diag_unconstrained.py"],
        ["code/exp1_summary.py"]]
TABLES = [["code/psi_build.py", "0.25"], ["code/psi_build.py", "0.5"], ["code/psi_build.py", "0.75"],
          ["code/psi_build_map.py", "hetero", "sat", "tail"]]
EXP2 = [["code/exp2_final.py", "core", "env", "lhs", "maps"],
        ["code/mech_summary.py"]]
FIGURES = [["code/make_figures.py"]]


def run(cmd):
    t0 = time.time()
    print(f"\n>>> python {' '.join(cmd)}", flush=True)
    subprocess.run([PY] + cmd, check=True)
    print(f"    done in {time.time() - t0:.0f} s", flush=True)


def verify():
    t0 = time.time()
    print("\n>>> python code/verify.py", flush=True)
    out = subprocess.run([PY, "code/verify.py"], check=False, capture_output=True, text=True)
    print(out.stdout, end="")
    if out.stderr:
        print(out.stderr, file=sys.stderr, end="")
    with open("results/verify_output.txt", "w", encoding="utf-8") as f:
        f.write(out.stdout)
    print(f"    done in {time.time() - t0:.0f} s (output saved to results/verify_output.txt)")
    if out.returncode != 0 or "ALL CHECKS PASSED" not in out.stdout:
        sys.exit("verification did not pass")


def main():
    args = sys.argv[1:]
    if not args or args[0] not in ("figures", "verify", "full"):
        sys.exit(__doc__)
    if args[0] == "figures":
        for c in FIGURES:
            run(c)
    elif args[0] == "verify":
        verify()
    else:
        steps = EXP1 + (TABLES if "--tables" in args else []) + EXP2 + FIGURES
        for c in steps:
            run(c)
        verify()


if __name__ == "__main__":
    main()
