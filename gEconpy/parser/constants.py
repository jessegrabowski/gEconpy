import re

import sympy as sp

from sympy.abc import _clash1, _clash2

LOCAL_DICT = {}
for letter in _clash1:
    LOCAL_DICT[letter] = sp.Symbol(letter)
for letter in _clash2:
    LOCAL_DICT[letter] = sp.Symbol(letter)

OPERATORS = re.escape("+-*/^=();:")

BLOCK_START_TOKEN = "{"
BLOCK_END_TOKEN = "};"
LAG_TOKEN = "[-1]"
LEAD_TOKEN = "[1]"
SS_TOKEN = "[ss]"
EXPECTATION_TOKEN = "E[]"
CALIBRATING_EQ_TOKEN = "->"


SPECIAL_BLOCK_NAMES = ["OPTIONS", "TRYREDUCE", "ASSUMPTIONS"]
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

TIME_INDEX_DICT = {"ss": "ss", "t": 0, "tL1": -1, "t1": 1}

GCN_ASSUMPTIONS = [
    "positive",
    "negative",
    "nonpositive",
    "nonnegative",
    "real",
    "integer",
    "finite",
]

SYMPY_ASSUMPTIONS = [
    "finite",
    "infinite",
    "even",
    "odd",
    "prime",
    "composite",
    "positive",
    "negative",
    "zero",
    "nonzero",
    "nonpositive",
    "nonnegative",
    "integer",
    "rational",
    "irrational",
    "real",
    "extended real",
    "hermitian",
    "complex",
    "imaginary",
    "antihermitian",
    "algebraic",
    "transcendental",
]

DEFAULT_ASSUMPTIONS = {"real": True}

# Distribution parameter mappings for PreliZ
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
