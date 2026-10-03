import pyparsing as pp

from gEconpy.parser.ast import GCNDistribution, GCNEquation, Variable
from gEconpy.parser.error_catalog import ErrorCode
from gEconpy.parser.errors import GCNParseFailure
from gEconpy.parser.grammar.statements import (
    CONTROL_TAG,
    DIST_EXPR,
    DISTRIBUTION,
    EQUATION,
    MISSING_TILDE,
    VARIABLE_LIST,
    VARIABLE_REF,
    collect_kwargs,
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


def _make_equation_component(keyword: pp.ParserElement, name: str) -> pp.ParserElement:
    component = keyword.suppress() - LBRACE - pp.ZeroOrMore(EQUATION)("equations") - RBRACE - SEMI
    component.set_parse_action(lambda t: (name, list(t.equations)))
    return component


DEFINITIONS = _make_equation_component(KW_DEFINITIONS, "definitions")
OBJECTIVE = _make_equation_component(KW_OBJECTIVE, "objective")
CONSTRAINTS = _make_equation_component(KW_CONSTRAINTS, "constraints")
IDENTITIES = _make_equation_component(KW_IDENTITIES, "identities")


# Records where each entry starts, so an error about one entry points at that entry rather than at the component.
_ENTRY_START = pp.Empty().set_parse_action(lambda _s, loc, _toks: loc)


def _build_controls(s: str, _loc: int, tokens: pp.ParseResults) -> tuple[str, tuple[list[Variable], dict[str, str]]]:
    controls: list[Variable] = []
    foc_names: dict[str, str] = {}

    for entry in tokens.entries:
        variables = list(entry.variables)
        controls.extend(variables)

        if not entry.tag:
            continue

        tag_name, caption = entry.tag[0]

        # The tag names the first-order condition one control produces, and a list of controls produces several.
        if len(variables) > 1:
            raise GCNParseFailure(
                s,
                entry.start[0],
                f"'@{tag_name}' names one first-order condition, so it cannot tag {len(variables)} controls at once",
                code=ErrorCode.E024,
                found=variables[0].name,
            )
        foc_names[variables[0].name] = caption

    return "controls", (controls, foc_names)


# One optionally-tagged list per entry, so the untagged ``C[], L[], K[];`` spelling stays a single entry and
# parses exactly as it did before. Both the start offset and the tag are grouped, because the branches of
# CONTROL_TAG flatten into the entry differently and a group gives each one the same shape either way.
_CONTROL_ENTRY = pp.Group(
    pp.Group(_ENTRY_START)("start") + pp.Group(pp.Optional(CONTROL_TAG))("tag") + VARIABLE_LIST("variables") + SEMI
)

CONTROLS = (
    KW_CONTROLS.suppress() - LBRACE - pp.ZeroOrMore(_CONTROL_ENTRY)("entries") - RBRACE - SEMI
).set_parse_action(_build_controls)


def _build_shock_distribution(tokens: pp.ParseResults) -> GCNDistribution:
    wrapper_name = tokens.wrapper_name or None
    initial_value = float(tokens.initial[0]) if tokens.initial else None

    return GCNDistribution(
        parameter_name=tokens.shock_var.name,
        dist_name=tokens.dist_name,
        dist_kwargs=collect_kwargs(tokens.dist_args),
        wrapper_name=wrapper_name,
        wrapper_kwargs=collect_kwargs(tokens.wrapper_args) if wrapper_name else {},
        initial_value=initial_value,
        location=tokens.shock_var.location,
    )


SHOCK_DISTRIBUTION = (
    VARIABLE_REF("shock_var") + TILDE + DIST_EXPR + pp.Optional(EQUALS + NUMBER)("initial") + SEMI
).set_parse_action(_build_shock_distribution)

SHOCK_VAR = (VARIABLE_REF + SEMI).set_parse_action(lambda t: t[0])

SHOCK_VAR_LIST = (VARIABLE_REF + COMMA + pp.DelimitedList(VARIABLE_REF) + SEMI).set_parse_action(list)

SHOCK_ITEM = SHOCK_DISTRIBUTION | SHOCK_VAR_LIST | SHOCK_VAR


def _build_shocks(tokens: pp.ParseResults) -> tuple[str, tuple[list[Variable], list[GCNDistribution]]]:
    variables = []
    distributions = []

    for item in tokens.shock_items:
        if isinstance(item, GCNDistribution):
            distributions.append(item)
            variables.append(Variable(name=item.parameter_name))
        elif isinstance(item, Variable):
            variables.append(item)
        else:
            variables.extend(item)

    return ("shocks", (variables, distributions))


SHOCKS = (KW_SHOCKS.suppress() - LBRACE - pp.ZeroOrMore(SHOCK_ITEM)("shock_items") - RBRACE - SEMI).set_parse_action(
    _build_shocks
)

CALIBRATION_ITEM = DISTRIBUTION | MISSING_TILDE | EQUATION


def _build_calibration(tokens: pp.ParseResults) -> tuple[str, list[GCNEquation | GCNDistribution]]:
    return ("calibration", list(tokens.cal_items))


CALIBRATION = (
    KW_CALIBRATION.suppress() - LBRACE - pp.ZeroOrMore(CALIBRATION_ITEM)("cal_items") - RBRACE - SEMI
).set_parse_action(_build_calibration)

VALID_COMPONENT = DEFINITIONS | CONTROLS | OBJECTIVE | CONSTRAINTS | IDENTITIES | SHOCKS | CALIBRATION


def _unknown_component_fail(s: str, loc: int, toks: pp.ParseResults) -> None:
    name = toks[0]
    name_loc = s.find(name, loc)
    if name_loc == -1:
        name_loc = loc
    raise GCNParseFailure(
        s,
        name_loc,
        f"Unknown component '{name}'",
        code=ErrorCode.E013,
        found=name,
        suggestions=suggest_block_component(name),
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
