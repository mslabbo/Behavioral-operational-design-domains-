"""
B-ODD core: latent-trait population models and the Gaussian-copula
behavioural mapping layer.

Mapping (one-factor Gaussian copula):
    z_k = s_k * rho_k * theta + sqrt(1 - rho_k^2) * eps_k,   eps_k ~ N(0,1)
    q_k = F_k^{-1}( Phi(z_k) )
If theta ~ N(0,1) (the reference population), z_k ~ N(0,1) and q_k ~ F_k
exactly: the reference marginals are reproduced by construction.
"""
import numpy as np
from scipy.stats import norm

# Reference marginals. kind: ln = lognormal(median, log-sd); n = normal(mu, sd)
# sign: +1 means higher theta (more self-regulation) -> larger value.
MARGINALS = {
    "T":      dict(kind="ln", med=1.40, s=0.30, sign=+1, lo=0.50, hi=3.00),  # desired time gap (s)
    "v0":     dict(kind="n",  mu=14.5,  sd=1.50, sign=-1, lo=10.5, hi=20.0), # desired speed (m/s)
    "a":      dict(kind="ln", med=1.40, s=0.25, sign=-1, lo=0.60, hi=3.00),  # max acceleration
    "b":      dict(kind="ln", med=2.00, s=0.20, sign=-1, lo=1.00, hi=4.00),  # comfortable decel
    "tau_cf": dict(kind="ln", med=0.80, s=0.25, sign=-1, lo=0.30, hi=1.60),  # car-following delay (s)
    "sig":    dict(kind="ln", med=0.30, s=0.30, sign=-1, lo=0.05, hi=1.20),  # accel. noise SD
    "tau_pr": dict(kind="ln", med=1.10, s=0.25, sign=-1, lo=0.50, hi=4.00),  # perception-response (s)
    "lapse":  dict(kind="probit", p=0.10, sign=-1),                          # attention lapse
    "theta_s": dict(kind="ln", med=0.40, s=0.50, sign=+1, lo=0.05, hi=2.00), # Gipps variant: chosen safety margin (s)
    "bhat":    dict(kind="ln", med=3.00, s=0.20, sign=+1, lo=1.50, hi=6.00), # Gipps variant: assumed leader braking (m/s^2)
}
LAPSE_EXTRA = (1.2, 0.7, 0.2, 8.0)  # lapse delay: lognormal(median, log-sd), clipped


def latent_z(theta, eps, rho, sign):
    return sign * rho * theta + np.sqrt(1.0 - rho ** 2) * eps


def from_z(name, z):
    m = MARGINALS[name]
    if m["kind"] == "ln":
        return np.clip(m["med"] * np.exp(m["s"] * z), m["lo"], m["hi"])
    if m["kind"] == "n":
        return np.clip(m["mu"] + m["sd"] * z, m["lo"], m["hi"])
    return z > norm.ppf(1.0 - m["p"])          # probit threshold


# ---- generalised mapping layer -----------------------------------------
# z_k = s_k rho_k m(theta) + sqrt(1 - rho_k^2) eps_k,  z*_k = Phi^-1(G_k(z_k)),
# q_k = F_k^-1(Phi(z*_k)), where G_k is the CDF of z_k when theta ~ N(0,1).
# For m(theta) = theta, G_k = Phi and this is the Gaussian copula. For any
# monotone m, the renormalisation through G_k keeps the reference marginals
# exact (Proposition 1 holds for every mapping variant).
_QT = np.linspace(-8.0, 8.0, 4001)
_QW = norm.pdf(_QT); _QW /= _QW.sum()
_ZG = np.linspace(-9.0, 9.0, 3601)


class Mapping:
    def __init__(self, rho=0.5, form="linear", name=None, c_sat=1.5, kappa=0.3):
        self.rho_spec, self.form = rho, form
        self.c_sat, self.kappa = c_sat, kappa
        self.name = name or (f"{form}" + ("" if isinstance(rho, dict) else f"_{rho}"))
        raw = self._raw(_QT)
        self._mu = float(_QW @ raw)
        self._sd = float(np.sqrt(_QW @ (raw - self._mu) ** 2))
        self._G = {}

    def _raw(self, th):
        if self.form == "linear":
            return th
        if self.form == "sat":          # saturating: trait effect flattens in both tails
            return self.c_sat * np.tanh(th / self.c_sat)
        if self.form == "tail":         # tail-accelerating: effect steepens in lower tail
            return th - self.kappa * np.minimum(th, 0.0) ** 2
        raise ValueError(self.form)

    def m(self, th):
        if self.form == "linear":
            return th
        return (self._raw(th) - self._mu) / self._sd

    def rho(self, k):
        return self.rho_spec[k] if isinstance(self.rho_spec, dict) else self.rho_spec

    def _table(self, srho):
        key = round(float(srho), 8)
        if key not in self._G:
            loc = srho * self.m(_QT)
            sc = np.sqrt(1 - srho ** 2)
            G = np.array([_QW @ norm.cdf((z - loc) / sc) for z in _ZG])
            G = np.clip(np.maximum.accumulate(G), 1e-15, 1 - 1e-15)
            self._G[key] = norm.ppf(G)            # z -> z*
        return self._G[key]

    def zstar(self, z, srho):
        """normal score of z under the reference distribution of z."""
        if self.form == "linear":
            return z
        return np.interp(z, _ZG, self._table(srho))

    def zinv(self, zs, srho):
        """inverse of zstar: latent z with reference normal score zs."""
        if self.form == "linear":
            return zs
        return np.interp(zs, self._table(srho), _ZG)


def as_mapping(rho):
    return rho if isinstance(rho, Mapping) else Mapping(rho, "linear")


def map_params(theta, eps, rho, names):
    M = as_mapping(rho)
    mt = M.m(theta)
    out = {}
    for k in names:
        s, r = MARGINALS[k]["sign"], M.rho(k)
        z = s * r * mt + np.sqrt(1.0 - r ** 2) * eps[k]
        out[k] = from_z(k, M.zstar(z, s * r))
    return out


# heterogeneous loadings: violation-type tendencies (speed, gap, acceleration)
# coupled more strongly than error/lapse-type tendencies (Helman and Reed, 2015)
RHO_HET = {"T": 0.65, "v0": 0.65, "a": 0.65, "b": 0.50,
           "tau_cf": 0.35, "sig": 0.35, "tau_pr": 0.35, "lapse": 0.35}
MAPPINGS = {
    "baseline": Mapping(0.5, "linear", "baseline"),
    "hetero":   Mapping(RHO_HET, "linear", "hetero"),
    "sat":      Mapping(0.5, "sat", "sat"),
    "tail":     Mapping(0.5, "tail", "tail"),
}


class Pop:
    """Two-component normal mixture parameterised by overall mean m, overall
    SD sd, low-component weight w, and standardised separation delta
    (difference of component means in units of the overall SD).
    w = 0 or delta = 0 gives a normal population."""

    def __init__(self, name, m=0.0, sd=1.0, w=0.0, delta=0.0):
        self.name, self.m, self.sd = name, m, sd
        if w > 0 and delta > 0:
            k = w * (1 - w) * delta ** 2
            if k >= 1:
                raise ValueError("infeasible mixture")
            self.w, self.delta = w, delta
            self.s = sd * np.sqrt(1 - k)
            self.mu_hi = m + w * delta * sd
            self.mu_lo = m - (1 - w) * delta * sd
        else:
            self.w, self.delta = 0.0, 0.0
            self.s, self.mu_hi, self.mu_lo = sd, m, m

    def sample(self, u, z):
        low = u < self.w
        return np.where(low, self.mu_lo, self.mu_hi) + self.s * z, low

    def pdf(self, th):
        return (self.w * norm.pdf(th, self.mu_lo, self.s)
                + (1 - self.w) * norm.pdf(th, self.mu_hi, self.s))

    def cdf(self, th):
        return (self.w * norm.cdf(th, self.mu_lo, self.s)
                + (1 - self.w) * norm.cdf(th, self.mu_hi, self.s))

    def shifted(self, dm, name=None):
        return Pop(name or self.name + f"{dm:+.2f}", self.m + dm, self.sd,
                   self.w, self.delta)

    def lpm(self, c):
        """Lower partial moment E[(c - theta)^+]."""
        out = 0.0
        for wt, mu in ((self.w, self.mu_lo), (1 - self.w, self.mu_hi)):
            if wt == 0:
                continue
            zz = (c - mu) / self.s
            out += wt * ((c - mu) * norm.cdf(zz) + self.s * norm.pdf(zz))
        return out

    def descriptors(self):
        p, w = 1 - self.w, self.w
        d1, d2 = self.mu_hi - self.m, self.mu_lo - self.m
        skew = (p * d1 ** 3 + w * d2 ** 3) / self.sd ** 3
        return dict(mean=self.m, sd=self.sd, skew=skew,
                    tail15=float(self.cdf(-1.5)), tail20=float(self.cdf(-2.0)),
                    lpm1=float(self.lpm(-1.0)))


# ---- canonical population scenarios ----------------------------------
# Two-phenotype scenario: stylised weight and separation (not estimated from data)
W_MIX, DELTA_MIX = 0.40, 1.5

POPS = {
    "R":      Pop("Reference",            0.0, 1.0),
    "D_loc":  Pop("Location shift",      -0.5, 1.0),
    "D_disp": Pop("Dispersion",           0.0, 1.3),
    "D_mix":  Pop("Two-phenotype",       -0.5, 1.0, W_MIX, DELTA_MIX),
    "D_tail": Pop("Extreme subgroup",    -0.5, 1.0, 0.10, 2.5),
}
