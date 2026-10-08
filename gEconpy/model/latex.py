from collections.abc import Callable

import sympy as sp

from gEconpy.classes.time_aware_symbol import TimeAwareSymbol, render_latex
from gEconpy.parser.ast import GCNModel, variable_key
from gEconpy.parser.ast.nodes import Expectation, GCNEquation
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


class PrintingConverter(ASTToSympyConverter):
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
    converter = PrintingConverter()
    return converter.convert_expr(equation.lhs), converter.convert_expr(equation.rhs)


def symbol_names_for(
    expression: sp.Expr,
    overrides: dict[str, str],
    renderer: Callable[..., str] = render_latex,
) -> dict[sp.Symbol, str]:
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
        Declared markup, keyed by the symbol's storage name.
    renderer : callable, optional
        Composes a declared stem with the symbol's subscripts, in the target language. Defaults to
        :func:`~gEconpy.classes.time_aware_symbol.render_latex`.

    Returns
    -------
    symbol_names : dict mapping sympy.Symbol to str
        One entry per symbol of ``expression`` carrying an override, as ``sympy.latex`` takes them.
    """
    if not overrides:
        return {}

    resolved = {}
    for symbol in expression.atoms(sp.Symbol):
        if isinstance(symbol, TimeAwareSymbol):
            override = overrides.get(variable_key(symbol.base_name))
            if override is not None:
                resolved[symbol] = renderer(symbol, stem_override=override)
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
    left, right = equation_expressions(source_ast, equation_id, expression, expectations)
    return render_sides(left, right, overrides, printer=sp.latex, renderer=render_latex)


def equation_expressions(
    source_ast: GCNModel | None,
    equation_id: str,
    expression: sp.Expr,
    expectations: bool = True,
) -> tuple[sp.Expr, sp.Expr]:
    """
    Recover one equation's two sides as expressions, before any language renders them.

    Parameters
    ----------
    source_ast : GCNModel, optional
        The parsed file, used to recover an authored equation's own two sides. None for a model built without one.
    equation_id : str
        The id of the equation.
    expression : sympy expression
        The equation as the solved system holds it, used when there is no authored form.
    expectations : bool, optional
        Wrap lead-carrying terms in a conditional expectation. Defaults to True.

    Returns
    -------
    left : sympy expression
        The left side. A derived equation's residual.
    right : sympy expression
        The right side, which is zero for a derived equation.
    """
    sides = authored_sides(source_ast, equation_id)
    left, right = (expression, sp.Integer(0)) if sides is None else sides
    if not expectations:
        return left, right

    # A side the author already wrote an operator on is left alone, since they said where it belongs.
    return wrap_unless_authored(left), wrap_unless_authored(right)


def authored_expressions(equation: GCNEquation) -> tuple[sp.Expr, sp.Expr]:
    """Convert one authored equation's two sides to expressions, keeping the operators the author wrote."""
    converter = PrintingConverter()
    return converter.convert_expr(equation.lhs), converter.convert_expr(equation.rhs)


def render_sides(
    left: sp.Expr,
    right: sp.Expr,
    overrides: dict[str, str] | None,
    printer: Callable[..., str],
    renderer: Callable[..., str],
) -> tuple[str, str]:
    """
    Render two sides of an equation, resolving declared markup against the symbols each side actually holds.

    An authored equation is rebuilt from the file, so its symbols are different objects from the solved
    system's and an override has to be matched by name rather than by identity.

    Parameters
    ----------
    left, right : sympy expression
        The two sides.
    overrides : dict mapping str to str, optional
        Declared markup, keyed by the symbol's storage name.
    printer : callable
        Renders an expression, taking ``symbol_names`` as ``sympy.latex`` does.
    renderer : callable
        Composes a declared stem with a symbol's subscripts, in the same language.

    Returns
    -------
    left : str
        The rendered left side.
    right : str
        The rendered right side.
    """
    names = overrides or {}
    return (
        printer(left, symbol_names=symbol_names_for(left, names, renderer=renderer)),
        printer(right, symbol_names=symbol_names_for(right, names, renderer=renderer)),
    )


def wrap_unless_authored(side: sp.Expr) -> sp.Expr:
    return side if side.has(ConditionalExpectation) else wrap_leads_in_expectations(side)


def authored_equation_latex(equation: GCNEquation, overrides: dict[str, str] | None = None) -> tuple[str, str]:
    """
    Render one authored equation as a left and a right side, exactly as its author wrote it.

    Parameters
    ----------
    equation : GCNEquation
        The parsed equation.
    overrides : dict mapping str to str, optional
        Declared LaTeX, keyed by the symbol's storage name. Defaults to none.

    Returns
    -------
    left : str
        The rendered left side.
    right : str
        The rendered right side.
    """
    left, right = authored_expressions(equation)
    return render_sides(left, right, overrides, printer=sp.latex, renderer=render_latex)


def definition_rows(source_ast: GCNModel | None, overrides: dict[str, str] | None = None) -> list[tuple[str, str, str]]:
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
        The block name and the rendered left and right side of each definition.
    """
    if source_ast is None:
        return []

    return [
        (block.name, *authored_equation_latex(equation, overrides))
        for block in source_ast.blocks
        for equation in block.definitions
    ]


def block_heading(block_name: str) -> str:
    """
    Render a block's identifier as prose, for a section heading or a generated caption.

    An underscore is a word break. A name written in all caps carries no case information, so it is lowered and
    its first word capitalized: ``TECHNOLOGY_SHOCKS`` reads as "Technology shocks". Any other name is the
    author's own casing and is left alone, so ``Ricardian_Household`` reads as "Ricardian Household" and an
    acronym survives.

    Parameters
    ----------
    block_name : str
        The block identifier as the author wrote it.

    Returns
    -------
    heading : str
        The identifier as prose.
    """
    words = " ".join(block_name.split("_"))
    return words.lower().capitalize() if block_name.isupper() else words
