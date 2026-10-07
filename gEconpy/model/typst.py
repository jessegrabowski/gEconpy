from typing import Any

import sympy as sp

from sympy.printing.precedence import PRECEDENCE, precedence
from sympy.printing.printer import Printer

from gEconpy.classes.time_aware_symbol import TimeAwareSymbol, render_name_typst, render_typst
from gEconpy.exceptions import TypstPrintError
from gEconpy.model.latex import authored_expressions, equation_expressions, render_sides
from gEconpy.parser.ast import GCNModel


class TypstPrinter(Printer):
    """
    Render a sympy expression as Typst math.

    Typst is not one of sympy's printing targets, so this covers the node types a ``.gcn`` model can produce and
    raises on everything else. Emitting a plausible-looking string for an unhandled node would put a silent error
    in a paper, which is worse than a failed render.
    """

    printmethod = "_typst"
    _default_settings: dict[str, Any] = {"symbol_names": {}, "order": None}

    def emptyPrinter(self, expr: sp.Basic) -> str:
        raise TypstPrintError(expr)

    def _print_Symbol(self, expr: sp.Symbol) -> str:
        override = self._settings["symbol_names"].get(expr)
        return override if override is not None else render_name_typst(expr.name)

    def _print_TimeAwareSymbol(self, expr: TimeAwareSymbol) -> str:
        override = self._settings["symbol_names"].get(expr)
        return override if override is not None else render_typst(expr)

    def _print_Integer(self, expr: sp.Integer) -> str:
        return str(expr.p)

    def _print_Float(self, expr: sp.Float) -> str:
        rendered = sp.printing.str.sstr(expr, full_prec=False)
        if "e" not in rendered:
            return rendered

        # Typst reads the e of Python's exponent notation as a variable, so 1.0e-7 typesets as 1.0e minus 7.
        mantissa, _, exponent = rendered.partition("e")
        return f"{mantissa} times 10^({int(exponent)})"

    def _print_Rational(self, expr: sp.Rational) -> str:
        return f"frac({expr.p}, {expr.q})"

    def _print_Add(self, expr: sp.Add) -> str:
        terms = expr.as_ordered_terms(order=self.order)
        rendered = self._print(terms[0])
        for term in terms[1:]:
            text = self._print(term)
            rendered += f" - {text[1:].lstrip()}" if text.startswith("-") else f" + {text}"
        return rendered

    def _print_Mul(self, expr: sp.Mul) -> str:
        sign, positive = ("-", -expr) if expr.could_extract_minus_sign() else ("", expr)
        numerator, denominator = sp.fraction(positive)

        if denominator is sp.S.One:
            return sign + self._render_product(positive)
        # frac() delimits both arguments itself, so neither needs parentheses of its own.
        return f"{sign}frac({self._print(numerator)}, {self._print(denominator)})"

    def _print_Pow(self, expr: sp.Pow) -> str:
        if expr.exp.is_Rational and expr.exp.is_negative:
            return f"frac(1, {self._print(sp.S.One / expr)})"
        return f"{self.parenthesize(expr.base, PRECEDENCE['Func'])}^({self._print(expr.exp)})"

    def _print_ConditionalExpectation(self, expr: sp.Expr) -> str:
        return f"EE_t [{self._print(expr.args[0])}]"

    def _print_log(self, expr: sp.Expr) -> str:
        return f"log({self._print(expr.args[0])})"

    def _print_exp(self, expr: sp.Expr) -> str:
        return f"e^({self._print(expr.args[0])})"

    def _render_product(self, expr: sp.Expr) -> str:
        if not expr.is_Mul:
            return self.parenthesize(expr, PRECEDENCE["Mul"])
        return " ".join(self.parenthesize(factor, PRECEDENCE["Mul"]) for factor in expr.as_ordered_factors())

    def parenthesize(self, expr: sp.Expr, level: int) -> str:
        rendered = self._print(expr)
        return f"({rendered})" if precedence(expr) < level else rendered

    @property
    def order(self) -> str | None:
        return self._settings["order"]


def typst(expression: sp.Expr, symbol_names: dict[sp.Symbol, str] | None = None) -> str:
    """
    Render a sympy expression as Typst math.

    Parameters
    ----------
    expression : sympy expression
        The expression to render.
    symbol_names : dict mapping sympy.Symbol to str, optional
        Declared Typst for individual symbols, replacing what would be inferred, as ``sympy.latex`` takes them.
        Defaults to inferring every symbol.

    Returns
    -------
    typst : str
        The rendered expression, without surrounding math delimiters.

    Raises
    ------
    TypstPrintError
        If the expression contains a node type the printer has no rendering for.
    """
    return TypstPrinter({"symbol_names": symbol_names or {}}).doprint(expression)


def equation_sides_typst(
    source_ast: GCNModel | None,
    equation_id: str,
    expression: sp.Expr,
    overrides: dict[str, str] | None = None,
    expectations: bool = True,
) -> tuple[str, str]:
    """
    Render one equation as a left and a right side of Typst math.

    The Typst twin of :func:`~gEconpy.model.latex.equation_sides_latex`, sharing its recovery of an authored
    equation's own two sides so the two languages print one equation rather than two different expressions.

    Parameters
    ----------
    source_ast : GCNModel, optional
        The parsed file, used to recover an authored equation's own two sides. None for a model built without one.
    equation_id : str
        The id of the equation.
    expression : sympy expression
        The equation as the solved system holds it, used when there is no authored form.
    overrides : dict mapping str to str, optional
        Declared Typst, keyed by the symbol's storage name. Defaults to none.
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
    return render_sides(left, right, overrides, printer=typst, renderer=render_typst)


def definition_rows_typst(
    source_ast: GCNModel | None,
    overrides: dict[str, str] | None = None,
) -> list[tuple[str, str, str]]:
    """
    Recover every ``definitions`` entry as Typst, in source order.

    Parameters
    ----------
    source_ast : GCNModel, optional
        The parsed file. None for a model built without one.
    overrides : dict mapping str to str, optional
        Declared Typst, keyed by the symbol's storage name. Defaults to none.

    Returns
    -------
    definitions : list of tuple of str
        The block name and the rendered left and right side of each definition.
    """
    if source_ast is None:
        return []

    return [
        (block.name, *render_sides(*authored_expressions(equation), overrides, printer=typst, renderer=render_typst))
        for block in source_ast.blocks
        for equation in block.definitions
    ]
