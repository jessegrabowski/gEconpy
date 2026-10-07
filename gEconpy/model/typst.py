from typing import Any

import sympy as sp

from sympy.printing.precedence import PRECEDENCE, precedence
from sympy.printing.printer import Printer

from gEconpy.classes.time_aware_symbol import TimeAwareSymbol, render_name_typst, render_typst
from gEconpy.exceptions import TypstPrintError


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
