import pyparsing as pp

from gEconpy.exceptions import InvalidParameterException, RepeatedParameterException

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

WRAPPER_TO_ARGS = {
    "maxent": ["lower", "upper", "mass"],
    "Censored": ["lower", "upper"],
    "Truncated": ["lower", "upper"],
    "Hurdle": ["psi"],
}


PRELIZ_DISTS = list(DIST_TO_PARAM_NAMES.keys())
PRELIZ_DIST_WRAPPERS = list(WRAPPER_TO_ARGS.keys())


def evaluate_expression(parsed_expr):
    if isinstance(parsed_expr, int | float):
        return float(parsed_expr)
    elif not parsed_expr:
        return None
    elif isinstance(parsed_expr, pp.ParseResults):
        parsed_expr = parsed_expr.as_list()
        if len(parsed_expr) == 1 and isinstance(parsed_expr[0], list):
            parsed_expr = parsed_expr[0]
        expr_str = "".join(map(str, parsed_expr))
        return eval(expr_str, {"__builtins__": None}, {})
    return parsed_expr


def result_to_dict(parsed_tokens, dist_name, valid_params):
    if not parsed_tokens:
        return {}

    parsed_tokens = [x for x in parsed_tokens.as_list() if x is not None]
    if len(parsed_tokens) == 0:
        return {}

    res = {}

    for param_name, param_value in parsed_tokens:
        if param_name in res:
            raise RepeatedParameterException(dist_name, param_name)
        if param_name not in valid_params:
            raise InvalidParameterException(dist_name, param_name, valid_params)
        res[param_name] = param_value

    return res


def process_initial_value(initial_value):
    if isinstance(initial_value, float | int | None):
        return initial_value

    return initial_value.as_list()


WRAPPER_FUNCS = pp.MatchFirst([pp.Keyword(wrapper) for wrapper in PRELIZ_DIST_WRAPPERS])
DISTRIBUTION_ID = pp.MatchFirst([pp.Keyword(dist) for dist in PRELIZ_DISTS])

VARIABLE_ID = pp.Word(pp.alphas, pp.alphanums + "_")

EQUALS = pp.Literal("=").suppress()
LPAREN = pp.Literal("(").suppress()
RPAREN = pp.Literal(")").suppress()
COMMA = pp.Literal(",").suppress()

NUMBER = pp.pyparsing_common.number
NUMBER_EXPR = pp.infixNotation(
    NUMBER,
    [
        (pp.Literal("/"), 2, pp.opAssoc.LEFT),
        (pp.Literal("*"), 2, pp.opAssoc.LEFT),
        (pp.Literal("+"), 2, pp.opAssoc.LEFT),
        (pp.Literal("-"), 2, pp.opAssoc.LEFT),
    ],
)

VALUE = NUMBER_EXPR | VARIABLE_ID

ARG_NAME = pp.Word(pp.alphas, pp.alphanums + "_")
KEY_VALUE_PAIR = pp.Group(ARG_NAME + EQUALS + VALUE)

KWARG_LIST = pp.Optional(pp.delimitedList(KEY_VALUE_PAIR, delim=COMMA), default=None)


DIST = DISTRIBUTION_ID("dist_name") + LPAREN + KWARG_LIST("dist_kwargs") + RPAREN
INITIAL_VALUE = EQUALS + NUMBER_EXPR("initial_value")

wrapped_distribution = (
    WRAPPER_FUNCS("wrapper_name")
    + LPAREN
    + DIST
    + pp.Optional(COMMA + KWARG_LIST("wrapper_kwargs"))
    + RPAREN
)

dist_syntax = (
    (wrapped_distribution | DIST) + pp.Optional(INITIAL_VALUE) + pp.StringEnd()
)


def dist_parse_action(tokens):
    res = {
        "dist_name": tokens["dist_name"],
        "wrapper_name": tokens.get("wrapper_name", None),
        "initial_value": process_initial_value(tokens.get("initial_value", None)),
    }

    dist_name = res["dist_name"]
    wrapper_name = res["wrapper_name"]

    res["dist_kwargs"] = result_to_dict(
        tokens["dist_kwargs"], dist_name, DIST_TO_PARAM_NAMES[dist_name]
    )
    res["wrapper_kwargs"] = result_to_dict(
        tokens.get("wrapper_kwargs"),
        wrapper_name,
        WRAPPER_TO_ARGS.get(wrapper_name, []),
    )

    return res


dist_syntax.add_parse_action(dist_parse_action)


__all__ = ["dist_syntax", "evaluate_expression", "PRELIZ_DISTS", "PRELIZ_DIST_WRAPPERS"]
