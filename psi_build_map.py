import sys, time, numpy as np, exp2_merge as X
from bodd_core import MAPPINGS
eps, _ = X.eps_draws(seed=2024, M=512)
for nm in sys.argv[1:]:
    t0 = time.time()
    psi = X.psi_table(MAPPINGS[nm], eps, M=128, dt_ct=0.05)
    np.save(f"tables/psi_{nm}.npy", psi)
    print(f"{nm} done {time.time()-t0:.0f}s", flush=True)
