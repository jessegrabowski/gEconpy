import sympy as sp

from sympy.abc import _clash1, _clash2

LOCAL_DICT = {letter: sp.Symbol(letter) for letter in (*_clash1, *_clash2)}

SPECIAL_BLOCK_NAMES = ["OPTIONS", "TRYREDUCE", "ASSUMPTIONS"]
EQUATION_TAGS = ["exclude", "minimize", "maximize"]
STEADY_STATE_NAMES = ["STEADY_STATE", "SS", "STEADYSTATE", "STEADY"]
BLOCK_COMPONENTS = [
    "DEFINITIONS",
    "CONTROLS",
    "OBJECTIVE",
    "CONSTRAINTS",
    "IDENTITIES",
    "SHOCKS",
    "CALIBRATION",
]

GCN_ASSUMPTIONS = [
    "positive",
    "negative",
    "nonpositive",
    "nonnegative",
    "real",
    "integer",
    "finite",
    "unit_interval",
]

DIST_TO_PARAM_NAMES = {
    "AsymmetricLaplace": ["kappa", "mu", "b", "q"],
    "Bernoulli": ["p", "logit_p"],
    "Beta": ["alpha", "beta", "mu", "sigma", "nu"],
    "BetaBinomial": ["alpha", "beta", "n"],
    "BetaScaled": ["alpha", "beta", "lower", "upper"],
    "Binomial": ["n", "p"],
    "Categorical": ["p", "logit_p"],
    "Cauchy": ["alpha", "beta"],
    "ChiSquared": ["nu"],
    "Dirichlet": ["alpha"],
    "DiscreteUniform": ["lower", "upper"],
    "DiscreteWeibull": ["q", "beta"],
    "ExGaussian": ["mu", "sigma", "nu"],
    "Exponential": ["lam", "beta"],
    "Gamma": ["alpha", "beta", "mu", "sigma"],
    "Geometric": ["p"],
    "Gumbel": ["mu", "beta"],
    "HalfCauchy": ["beta"],
    "HalfNormal": ["sigma", "tau"],
    "HalfStudentT": ["nu", "sigma", "lam"],
    "HyperGeometric": ["N", "k", "n"],
    "InverseGamma": ["alpha", "beta", "mu", "sigma"],
    "Kumaraswamy": ["a", "b"],
    "Laplace": ["mu", "b"],
    "LogLogistic": ["alpha", "beta"],
    "LogNormal": ["mu", "sigma"],
    "Logistic": ["mu", "s"],
    "LogitNormal": ["mu", "sigma", "tau"],
    "Moyal": ["mu", "sigma"],
    "MvNormal": ["mu", "cov", "tau"],
    "NegativeBinomial": ["mu", "alpha", "p", "n"],
    "Normal": ["mu", "sigma", "tau"],
    "Pareto": ["alpha", "m"],
    "Poisson": ["mu"],
    "Rice": ["nu", "sigma", "b"],
    "SkewNormal": ["mu", "sigma", "alpha", "tau"],
    "SkewStudentT": ["mu", "sigma", "a", "b", "lam"],
    "StudentT": ["nu", "mu", "sigma", "lam"],
    "Triangular": ["lower", "c", "upper"],
    "TruncatedNormal": ["mu", "sigma", "lower", "upper"],
    "Uniform": ["lower", "upper"],
    "VonMises": ["mu", "kappa"],
    "Wald": ["mu", "lam", "phi"],
    "Weibull": ["alpha", "beta"],
    "ZeroInflatedBinomial": ["psi", "n", "p"],
    "ZeroInflatedNegativeBinomial": ["psi", "mu", "alpha", "p", "n"],
    "ZeroInflatedPoisson": ["psi", "mu"],
}

WRAPPER_TO_PARAM_NAMES = {
    "maxent": ["lower", "upper", "mass"],
    "Censored": ["lower", "upper"],
    "Truncated": ["lower", "upper"],
    "Hurdle": ["psi"],
}

PRELIZ_DISTS = list(DIST_TO_PARAM_NAMES.keys())
PRELIZ_DIST_WRAPPERS = list(WRAPPER_TO_PARAM_NAMES.keys())

KNOWN_DISTRIBUTIONS = frozenset(PRELIZ_DISTS)
KNOWN_WRAPPERS = frozenset(PRELIZ_DIST_WRAPPERS)
KNOWN_COMPONENTS = frozenset(name.lower() for name in BLOCK_COMPONENTS)
KNOWN_SPECIAL_BLOCKS = frozenset(name.lower() for name in SPECIAL_BLOCK_NAMES)
KNOWN_ASSUMPTIONS = frozenset(GCN_ASSUMPTIONS)
