import pyparsing as pp

from gEconpy.parser.ast import GCNDistribution, GCNEquation, Variable
from gEconpy.parser.constants import BLOCK_COMPONENTS
from gEconpy.parser.error_catalog import ErrorCode
from gEconpy.parser.errors import GCNParseFailure
from gEconpy.parser.grammar.statements import (
    DIST_EXPR,
    DISTRIBUTION,
    EQUATION,
    VARIABLE_LIST,
    VARIABLE_REF,
)
from gEconpy.parser.grammar.tokens import (
    COMMA,
    COMMENT,
    EQUALS,
    IDENTIFIER,
    KW_CALIBRATION,
    KW_CONSTRAINTS,
    KW_CONTROLS,
    KW_DEFINITIONS,
    KW_IDENTITIES,
    KW_OBJECTIVE,
    KW_SHOCKS,
    LBRACE,
    NUMBER,
    RBRACE,
    SEMI,
    TILDE,
)
from gEconpy.parser.suggestions import suggest_block_component

_KNOWN_COMPONENTS = frozenset(c.lower() for c in BLOCK_COMPONENTS)


def _make_equation_component(keyword: pp.ParserElement, name: str) -> pp.ParserElement:
    component = keyword.suppress() - LBRACE - pp.ZeroOrMore(EQUATION)("equations") - RBRACE - SEMI
    component.set_parse_action(lambda t: (name, list(t.equations)))
    return component


DEFINITIONS = _make_equation_component(KW_DEFINITIONS, "definitions")
OBJECTIVE = _make_equation_component(KW_OBJECTIVE, "objective")
CONSTRAINTS = _make_equation_component(KW_CONSTRAINTS, "constraints")
IDENTITIES = _make_equation_component(KW_IDENTITIES, "identities")

CONTROLS = (
    KW_CONTROLS.suppress() - LBRACE - pp.Optional(VARIABLE_LIST("variables") + SEMI) - RBRACE - SEMI
).set_parse_action(lambda t: ("controls", list(t.variables) if t.variables else []))

SHOCK_DISTRIBUTION = VARIABLE_REF("shock_var") + TILDE + DIST_EXPR + pp.Optional(EQUALS + NUMBER)("initial") + SEMI


def _build_shock_distribution(tokens) -> GCNDistribution:
    var = tokens.shock_var
    param_name = var.name

    dist_name = tokens.dist_name

    dist_kwargs = {}
    if hasattr(tokens, "dist_args") and tokens.dist_args:
        for arg in tokens.dist_args:
            dist_kwargs[arg.arg_name] = arg.arg_value

    wrapper_name = tokens.wrapper_name if hasattr(tokens, "wrapper_name") and tokens.wrapper_name else None
    wrapper_kwargs = {}
    if wrapper_name and hasattr(tokens, "wrapper_args") and tokens.wrapper_args:
        for arg in tokens.wrapper_args:
            wrapper_kwargs[arg.arg_name] = arg.arg_value

    initial_value = None
    if tokens.initial:
        initial_value = float(tokens.initial[0])

    return GCNDistribution(
        parameter_name=param_name,
        dist_name=dist_name,
        dist_kwargs=dist_kwargs,
        wrapper_name=wrapper_name,
        wrapper_kwargs=wrapper_kwargs,
        initial_value=initial_value,
    )


SHOCK_DISTRIBUTION.set_parse_action(_build_shock_distribution)

SHOCK_VAR = (VARIABLE_REF + SEMI).set_parse_action(lambda t: t[0])

SHOCK_VAR_LIST = (VARIABLE_REF + pp.Suppress(COMMA) + pp.DelimitedList(VARIABLE_REF) + SEMI).set_parse_action(list)

SHOCK_ITEM = SHOCK_DISTRIBUTION | SHOCK_VAR_LIST | SHOCK_VAR


def _build_shocks(tokens) -> tuple[str, tuple[list[Variable], list[GCNDistribution]]]:
    variables = []
    distributions = []

    for item in tokens.shock_items:
        if isinstance(item, GCNDistribution):
            distributions.append(item)
            variables.append(Variable(name=item.parameter_name))
        elif isinstance(item, Variable):
            variables.append(item)
        elif isinstance(item, list):
            variables.extend(item)

    return ("shocks", (variables, distributions))


SHOCKS = (KW_SHOCKS.suppress() - LBRACE - pp.ZeroOrMore(SHOCK_ITEM)("shock_items") - RBRACE - SEMI).set_parse_action(
    _build_shocks
)

CALIBRATION_ITEM = DISTRIBUTION | EQUATION


def _build_calibration(tokens) -> tuple[str, list[GCNEquation | GCNDistribution]]:
    return ("calibration", list(tokens.cal_items))


CALIBRATION = (
    KW_CALIBRATION.suppress() - LBRACE - pp.ZeroOrMore(CALIBRATION_ITEM)("cal_items") - RBRACE - SEMI
).set_parse_action(_build_calibration)

VALID_COMPONENT = DEFINITIONS | CONTROLS | OBJECTIVE | CONSTRAINTS | IDENTITIES | SHOCKS | CALIBRATION


def _unknown_component_fail(s: str, loc: int, toks) -> None:
    name = toks[0]
    actual_loc = s.find(name, loc)
    if actual_loc == -1:
        actual_loc = loc
    suggestions = suggest_block_component(name)
    raise GCNParseFailure(
        s,
        actual_loc,
        f"Unknown component '{name}'",
        code=ErrorCode.E013,
        found=name,
        suggestions=suggestions,
    )


UNKNOWN_COMPONENT = (pp.NotAny(VALID_COMPONENT) + IDENTIFIER("name") + pp.FollowedBy(LBRACE)).set_parse_action(
    _unknown_component_fail
)

COMPONENT = VALID_COMPONENT | UNKNOWN_COMPONENT
COMPONENT.ignore(COMMENT)


__all__ = [
    "CALIBRATION",
    "COMPONENT",
    "CONSTRAINTS",
    "CONTROLS",
    "DEFINITIONS",
    "IDENTITIES",
    "OBJECTIVE",
    "SHOCKS",
]
