import sympy as sp

from gEconpy.classes.time_aware_symbol import TimeAwareSymbol
from gEconpy.parser.ast import GCNModel
from gEconpy.parser.transform.to_sympy import ast_to_sympy


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
    return ast_to_sympy(equation.lhs), ast_to_sympy(equation.rhs)
