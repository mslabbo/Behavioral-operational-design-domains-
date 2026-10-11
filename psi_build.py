import sys, time, numpy as np, exp2_merge as X
rho=float(sys.argv[1]); eps,_=X.eps_draws(seed=2024,M=512)
t0=time.time(); psi=X.psi_table(rho,eps,M=128,dt_ct=0.05)
np.save(f"tables/psi_r{int(100*rho)}.npy",psi); print(f"rho={rho} done {time.time()-t0:.0f}s shape={psi.shape}")
