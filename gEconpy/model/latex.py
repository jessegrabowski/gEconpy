import sympy as sp

from gEconpy.classes.time_aware_symbol import TimeAwareSymbol


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
