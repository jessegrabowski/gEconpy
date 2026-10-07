from collections.abc import Callable

import pytest
import sympy as sp

from gEconpy import model_from_gcn
from gEconpy.classes.time_aware_symbol import TimeAwareSymbol
from gEconpy.data import get_example_gcn
from gEconpy.exceptions import TypstPrintError
from gEconpy.model.latex import wrap_leads_in_expectations
from gEconpy.model.typst import typst

alpha = sp.Symbol("alpha")
A, C, K, mc = (TimeAwareSymbol(name, 0) for name in ("A", "C", "K", "mc"))


class TestNodes:
    @pytest.mark.parametrize(
        "expression, expected",
        [
            (C, "C_(t)"),
            (alpha, "alpha"),
            (mc, 'upright("mc")_(t)'),
            (sp.Integer(3), "3"),
            (sp.Float(0.99), "0.99"),
            (sp.Rational(1, 2), "frac(1, 2)"),
            (C + alpha, "alpha + C_(t)"),
            (C * alpha, "alpha C_(t)"),
            (C**alpha, "C_(t)^(alpha)"),
            (C / K, "frac(C_(t), K_(t))"),
            (C ** sp.Integer(-1), "frac(1, C_(t))"),
            (sp.log(A), "log(A_(t))"),
            (sp.exp(-C), "e^(-C_(t))"),
            ((C + alpha) * K, "K_(t) (alpha + C_(t))"),
            (
                TimeAwareSymbol("K", -1) ** alpha * TimeAwareSymbol("L", 0) ** (1 - alpha),
                "K_(t-1)^(alpha) L_(t)^(1 - alpha)",
            ),
        ],
        ids=[
            "time_aware_symbol",
            "parameter",
            "multi_letter",
            "integer",
            "float",
            "rational",
            "add",
            "mul",
            "pow",
            "fraction",
            "negative_power",
            "log",
            "exp",
            "parenthesized_sum",
            "cobb_douglas",
        ],
    )
    def test_each_node_renders(self, expression, expected):
        assert typst(expression) == expected

    @pytest.mark.parametrize(
        "value, expected",
        [(1e-7, "1.0 times 10^(-7)"), (2.5e12, "2500000000000.0"), (0.99, "0.99")],
        ids=["scientific", "large_but_plain", "plain"],
    )
    def test_a_float_never_prints_pythons_exponent_notation(self, value, expected):
        """Typst reads the e as a variable, so 1.0e-7 typesets as 1.0e minus 7 rather than a small number."""
        assert typst(sp.Float(value)) == expected

    def test_an_expectation_uses_typsts_own_operator(self):
        """Typst has a double-struck E built in, so the operator needs no package the way LaTeX's does."""
        wrapped = wrap_leads_in_expectations(sp.Symbol("beta") * TimeAwareSymbol("U", 1))

        assert typst(wrapped) == "beta EE_t [U_(t+1)]"

    def test_a_node_with_no_rendering_raises(self):
        """A plausible-looking string for an unhandled node would put a silent error in a paper."""
        with pytest.raises(TypstPrintError, match=r"No Typst rendering for sin.*_print_sin method"):
            typst(sp.sin(C))

    def test_a_declared_name_replaces_the_inferred_one(self):
        assert typst(C + K, symbol_names={C: "cal(C)_(t)"}) == "cal(C)_(t) + K_(t)"


class TestAgreementWithLatex:
    """The two printers render the same expressions, so a structural divergence prints two different models."""

    def test_additive_terms_keep_the_order_the_latex_printer_gives_them(self):
        """Sympy's LaTeX printer reorders additive terms, and a printer that does not drifts from it."""
        assert sp.latex(mc - 1) == r"\text{mc}_{t} - 1"
        assert typst(mc - 1) == 'upright("mc")_(t) - 1'

    @pytest.mark.parametrize(
        "model_name", ["RBC", "RBC_two_household", "New_Keynesian"], ids=["rbc", "two_household", "nk"]
    )
    def test_every_sum_in_a_packaged_model_keeps_the_same_term_order(self, model_name):
        """
        Compare the order of each sum's terms, which is the divergence that matters.

        Both printers read one expression and sympy's LaTeX printer reorders what it is given. How either lays a
        term out is deliberately not compared: the LaTeX printer leaves a negative power as a leading factor
        where this one builds a ``frac``, which moves a variable within its own term and says the same thing.
        """
        model = model_from_gcn(get_example_gcn(model_name), verbose=False)
        sums = {
            addition for equation in model.equations for addition in equation.atoms(sp.Add) if len(addition.args) > 1
        }
        assert sums, "a model with no sums cannot discriminate"

        for addition in sums:
            assert _term_order(addition, sp.latex) == _term_order(addition, typst), addition


def _term_order(addition: sp.Add, printer: Callable[[sp.Expr], str]) -> list[sp.Expr]:
    """Return the sum's terms, ordered by where the printer places each one in the rendered sum."""
    rendered = printer(addition)
    positions = {}
    for term in addition.args:
        # A negative term carries its sign into the sum as an operator, so the sign is not part of what to find.
        text = printer(term).lstrip("-").strip()
        assert text in rendered, f"{term} rendered as {text!r}, which is not in {rendered!r}"
        positions[term] = rendered.index(text)
    return sorted(positions, key=positions.__getitem__)
