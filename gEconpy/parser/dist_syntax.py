import pyparsing as pp


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


preliz_dists = [
    "AsymmetricLaplace",
    "Bernoulli",
    "Beta",
    "BetaBinomial",
    "BetaScaled",
    "Binomial",
    "Categorical",
    "Cauchy",
    "ChiSquared",
    "Dirichlet",
    "DiscreteUniform",
    "DiscreteWeibull",
    "ExGaussian",
    "Exponential",
    "Gamma",
    "Geometric",
    "Gumbel",
    "HalfCauchy",
    "HalfNormal",
    "HalfStudentT",
    "HyperGeometric",
    "InverseGamma",
    "Kumaraswamy",
    "Laplace",
    "LogLogistic",
    "LogNormal",
    "Logistic",
    "LogitNormal",
    "Moyal",
    "MvNormal",
    "NegativeBinomial",
    "Normal",
    "Pareto",
    "Poisson",
    "Rice",
    "SkewNormal",
    "SkewStudentT",
    "StudentT",
    "Triangular",
    "TruncatedNormal",
    "Uniform",
    "VonMises",
    "Wald",
    "Weibull",
    "ZeroInflatedBinomial",
    "ZeroInflatedNegativeBinomial",
    "ZeroInflatedPoisson",
]


wrapper_keywords = pp.MatchFirst(
    [pp.Keyword("maxent"), pp.Keyword("Truncated"), pp.Keyword("Censored")]
)("wrapper_func")

dist_name = pp.MatchFirst([pp.Keyword(dist) for dist in preliz_dists])
var_name = pp.Word(pp.alphas, pp.alphanums + "_")

equals = pp.Literal("=").suppress()

number = pp.pyparsing_common.number
numeric_expr = pp.infixNotation(
    number,
    [
        (pp.Literal("/"), 2, pp.opAssoc.LEFT),
        (pp.Literal("*"), 2, pp.opAssoc.LEFT),
        (pp.Literal("+"), 2, pp.opAssoc.LEFT),
        (pp.Literal("-"), 2, pp.opAssoc.LEFT),
    ],
)

value = numeric_expr | var_name

key = pp.Word(pp.alphas, pp.alphanums + "_")
key_value_pair = pp.Group(key + equals + value)

args = pp.Optional(
    pp.nestedExpr("(", ")", content=pp.delimitedList(key_value_pair)), default=None
)("dist_kwargs")
inner_dist_syntax = dist_name("dist_name") + args

wrapper_args = pp.Optional(
    pp.Suppress(",") + pp.delimitedList(key_value_pair), default=None
)("wrapper_kwargs")

dist_syntax = (
    pp.Optional(wrapper_keywords, default=None)("wrapper_func")
    + pp.Optional(pp.Suppress("("))
    + inner_dist_syntax("distribution")
    + wrapper_args
    + pp.Optional(pp.Suppress(")"))
    + pp.Optional(equals)
    + pp.Optional(numeric_expr, default=None)("initial_value")
    + pp.StringEnd()
)


__all__ = ["dist_syntax", "evaluate_expression"]
