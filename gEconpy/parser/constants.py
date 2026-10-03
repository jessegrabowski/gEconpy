import sympy as sp

from sympy.abc import _clash1, _clash2

LOCAL_DICT = {letter: sp.Symbol(letter) for letter in (*_clash1, *_clash2)}

EQUATION_TAGS = ["exclude", "minimize", "maximize"]

# Tags that take a quoted string rather than standing alone. Their values are captions, so they are prose.
# Equations take ``name``. A control takes ``foc_name``, which names the first-order condition the control
# produces rather than the control itself, since the derived equation has no source text of its own to tag.
VALUE_TAGS = ["name"]
CONTROL_TAGS = ["foc_name"]
STEADY_STATE_NAMES = ["STEADY_STATE", "SS", "STEADYSTATE", "STEADY"]
STEADY_STATE_BLOCK_KEYS = frozenset(name.replace("_", "") for name in STEADY_STATE_NAMES)
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
    "nonzero",
    "real",
    "integer",
    "finite",
    "unit_interval",
]

# Assumptions that constrain a symbol's support. Each maps to the interval it implies, which is checked against any
# bound declared alongside it. The remaining assumptions in GCN_ASSUMPTIONS say nothing about the support and are
# passed to sympy untouched.
SIGN_ASSUMPTION_SUPPORTS: dict[str, tuple[tuple[float | None, float | None], tuple[bool, bool]]] = {
    "positive": ((0.0, None), (False, False)),
    "nonnegative": ((0.0, None), (True, False)),
    "negative": ((None, 0.0), (False, False)),
    "nonpositive": ((None, 0.0), (False, True)),
    "unit_interval": ((0.0, 1.0), (False, False)),
}

SYMBOL_METADATA_FIELDS = ["name", "latex", "source"]
SYMBOL_FIELDS = [*SYMBOL_METADATA_FIELDS, "bounds", *GCN_ASSUMPTIONS]

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
KNOWN_ASSUMPTIONS = frozenset(GCN_ASSUMPTIONS)
KNOWN_SYMBOL_FIELDS = frozenset(SYMBOL_FIELDS)
