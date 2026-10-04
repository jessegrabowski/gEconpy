import sympy as sp

from gEconpy.classes.time_aware_symbol import TimeAwareSymbol, render_latex
from gEconpy.parser.ast import GCNModel, variable_key
from gEconpy.parser.ast.nodes import Expectation
from gEconpy.parser.transform.to_sympy import ASTToSympyConverter


class ConditionalExpectation(sp.Function):
    r"""
    A placeholder wrapper that renders as :math:`\\mathbb{E}_t[\\cdot]`.

    It exists only for printing. Model equations are expectation-free, because first-order perturbation is
    certainty equivalent and the parser drops the operator, so nothing here reaches the solver.
    """

    nargs = 1

    def _latex(self, printer) -> str:
        return rf"\mathbb{{E}}_t\left[{printer._print(self.args[0])}\right]"


def wrap_leads_in_expectations(expression: sp.Expr) -> sp.Expr:
    """
    Wrap each additive term carrying a lead variable in a conditional expectation.

    Every lead in a first-order perturbation model came from under an expectation the parser discarded, so the
    operator is recoverable rather than guessed. Factors that carry no lead stay outside, which is how the
    discount factor ends up in front of the operator rather than inside it.

    Parameters
    ----------
    expression : sympy expression
        An equation from the solved system.

    Returns
    -------
    wrapped : sympy expression
        The same expression with lead-carrying factors wrapped. Unchanged when it holds no leads.
    """
    leads = [
        symbol
        for symbol in expression.atoms(sp.Symbol)
        if isinstance(symbol, TimeAwareSymbol) and isinstance(symbol.time_index, int) and symbol.time_index > 0
    ]
    if not leads:
        return expression

    terms = []
    for term in sp.Add.make_args(expression):
        without_leads, with_leads = term.as_independent(*leads)
        terms.append(term if with_leads == 1 else without_leads * ConditionalExpectation(with_leads))

    return sp.Add(*terms)


class _PrintingConverter(ASTToSympyConverter):
    """
    Converts an authored equation while keeping its expectation operator.

    The solver path drops the operator, because first-order perturbation is certainty equivalent. A printed
    equation should show where the author put it.
    """

    def _convert_expectation(self, node: Expectation) -> sp.Basic:
        return ConditionalExpectation(self.convert(node.expr))


_AUTHORED_COMPONENTS = ("identities", "constraints", "objective")


def authored_sides(source_ast: GCNModel | None, equation_id: str) -> tuple[sp.Expr, sp.Expr] | None:
    """
    Recover the two sides of an equation as its author wrote them.

    ``Block.solve_optimization`` stores every equation as a residual equal to zero, which no paper tabulates. A
    derived first-order condition has no authored form, so it has nothing to recover.

    Parameters
    ----------
    source_ast : GCNModel, optional
        The parsed file, from ``Model._source_ast``. None for a model built without one.
    equation_id : str
        The id of the equation, built in ``Block.solve_optimization`` as ``{block}.{component}.{position}``.

    Returns
    -------
    sides : tuple of sympy expression, or None
        The left and right sides, or None for a derived equation or an id the file cannot match.
    """
    if source_ast is None:
        return None

    block_name, _, remainder = equation_id.partition(".")
    component, _, position = remainder.partition(".")
    if component not in _AUTHORED_COMPONENTS:
        return None

    block = source_ast.get_block(block_name)
    if block is None:
        return None

    # A block has one objective, so that id carries no position. Every other component is positional, and a
    # position that is not a plain number belongs to an id this module did not build. ``-1`` matters here: it
    # would otherwise index from the end and quietly return the wrong equation.
    if component == "objective":
        index = 0
    elif position.isdigit():
        index = int(position)
    else:
        return None

    equations = getattr(block, component)
    if index >= len(equations):
        return None

    equation = equations[index]
    converter = _PrintingConverter()
    return converter.convert_expr(equation.lhs), converter.convert_expr(equation.rhs)


def symbol_names_for(expression: sp.Expr, overrides: dict[str, str]) -> dict[sp.Symbol, str]:
    """
    Resolve declared LaTeX overrides against the symbols of one expression.

    Sympy caches a symbol on its name and assumptions, so the same variable is a different object once it is
    rebuilt without them. Matching on the storage name rather than on identity is what lets an override survive
    crossing from the parsed file to the solved system.

    Parameters
    ----------
    expression : sympy expression
        The expression about to be rendered.
    overrides : dict mapping str to str
        Declared LaTeX, keyed by the symbol's storage name.

    Returns
    -------
    symbol_names : dict mapping sympy.Symbol to str
        One entry per symbol of ``expression`` carrying an override, as :func:`sympy.latex` takes them.
    """
    if not overrides:
        return {}

    resolved = {}
    for symbol in expression.atoms(sp.Symbol):
        if isinstance(symbol, TimeAwareSymbol):
            override = overrides.get(variable_key(symbol.base_name))
            if override is not None:
                resolved[symbol] = render_latex(symbol, stem_override=override)
        elif symbol.name in overrides:
            resolved[symbol] = overrides[symbol.name]
    return resolved


def equation_sides_latex(
    source_ast: GCNModel | None,
    equation_id: str,
    expression: sp.Expr,
    overrides: dict[str, str] | None = None,
    expectations: bool = True,
) -> tuple[str, str]:
    """
    Render one equation as a left and a right side.

    Parameters
    ----------
    source_ast : GCNModel, optional
        The parsed file, used to recover an authored equation's own two sides. None for a model built without one.
    equation_id : str
        The id of the equation.
    expression : sympy expression
        The equation as the solved system holds it, used when there is no authored form.
    overrides : dict mapping str to str, optional
        Declared LaTeX, keyed by the symbol's storage name. Defaults to none.
    expectations : bool, optional
        Wrap lead-carrying terms in a conditional expectation. Defaults to True.

    Returns
    -------
    left : str
        The left side. A derived equation's residual.
    right : str
        The right side, which is ``0`` for a derived equation.
    """
    sides = authored_sides(source_ast, equation_id)
    left, right = (expression, sp.Integer(0)) if sides is None else sides

    if expectations:
        # A side the author already wrote an operator on is left alone, since they said where it belongs.
        left, right = _wrap_unless_authored(left), _wrap_unless_authored(right)

    # Resolved against the sides actually being rendered, which are rebuilt symbols for an authored equation.
    names = overrides or {}
    return (
        sp.latex(left, symbol_names=symbol_names_for(left, names)),
        sp.latex(right, symbol_names=symbol_names_for(right, names)),
    )


def _wrap_unless_authored(side: sp.Expr) -> sp.Expr:
    return side if side.has(ConditionalExpectation) else wrap_leads_in_expectations(side)


def definition_sides_latex(
    source_ast: GCNModel | None, overrides: dict[str, str] | None = None
) -> list[tuple[str, str]]:
    """
    Recover every ``definitions`` entry, in source order.

    ``Block.solve_optimization`` substitutes definitions into the equations that use them, so they never reach
    the solved system. An authored equation still names them, and without these rows the printed system has
    more unknowns than equations.

    Parameters
    ----------
    source_ast : GCNModel, optional
        The parsed file. None for a model built without one.
    overrides : dict mapping str to str, optional
        Declared LaTeX, keyed by the symbol's storage name. Defaults to none.

    Returns
    -------
    definitions : list of tuple of str
        The rendered left and right side of each definition.
    """
    if source_ast is None:
        return []

    converter = _PrintingConverter()
    names = overrides or {}
    return [
        (
            sp.latex(left := converter.convert_expr(equation.lhs), symbol_names=symbol_names_for(left, names)),
            sp.latex(right := converter.convert_expr(equation.rhs), symbol_names=symbol_names_for(right, names)),
        )
        for block in source_ast.blocks
        for equation in block.definitions
    ]
