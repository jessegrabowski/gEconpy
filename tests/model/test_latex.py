import re

from pathlib import Path

import pytest
import sympy as sp

from gEconpy import model_from_gcn
from gEconpy.classes.time_aware_symbol import TimeAwareSymbol
from gEconpy.data import get_example_gcn
from gEconpy.model.latex import authored_sides, wrap_leads_in_expectations
from tests.conftest import TEST_GCNS

GOLDEN = Path(__file__).parent.parent / "_resources" / "RBC_equations.tex"


@pytest.fixture(scope="module")
def rbc():
    """Built once: every test here reads the same rendering, and building the model dominates their runtime."""
    return model_from_gcn(get_example_gcn("RBC"), verbose=False)


class TestExpectationWrapper:
    def test_a_lead_carrying_term_is_wrapped_and_its_coefficient_stays_outside(self):
        """The discount factor belongs in front of the operator, which is what as_independent buys."""
        beta, delta = sp.symbols("beta delta")
        lam_lead, r_lead, lam = TimeAwareSymbol("lambda", 1), TimeAwareSymbol("r", 1), TimeAwareSymbol("lambda", 0)

        wrapped = wrap_leads_in_expectations(beta * (lam_lead * r_lead - lam_lead * (delta - 1)) - lam)

        assert sp.latex(wrapped) == (
            r"\beta \mathbb{E}_t\left[\lambda_{t+1} r_{t+1} "
            r"- \lambda_{t+1} \left(\delta - 1\right)\right] - \lambda_{t}"
        )

    @pytest.mark.parametrize("time_index", [0, -1, "ss"], ids=["contemporaneous", "lag", "steady_state"])
    def test_an_expression_with_no_lead_is_returned_unchanged(self, time_index):
        expression = 2 * TimeAwareSymbol("K", time_index) + 1

        assert wrap_leads_in_expectations(expression) is expression

    def test_only_the_term_carrying_the_lead_is_wrapped(self):
        lead, now = TimeAwareSymbol("C", 1), TimeAwareSymbol("C", 0)

        rendered = sp.latex(wrap_leads_in_expectations(lead + now))

        assert r"\mathbb{E}_t\left[C_{t+1}\right]" in rendered
        assert r"\mathbb{E}_t\left[C_{t}\right]" not in rendered


class TestAuthoredForm:
    @pytest.mark.parametrize(
        "equation_id, expected",
        [
            ("HOUSEHOLD.constraints.0", r"C_{t} + I_{t} = K_{t-1} r_{t} + L_{t} w_{t}"),
            ("FIRM.constraints.0", r"Y_{t} = A_{t} K_{t-1}^{\alpha} L_{t}^{1 - \alpha}"),
        ],
        ids=["budget_constraint", "production_function"],
    )
    def test_an_authored_equation_keeps_the_sides_its_author_wrote(self, rbc, equation_id, expected):
        left, right = authored_sides(rbc._source_ast, equation_id)

        assert f"{sp.latex(left)} = {sp.latex(right)}" == expected

    @pytest.mark.parametrize("equation_id", ["HOUSEHOLD.foc.C", "FIRM.foc.L"], ids=["consumption", "labor"])
    def test_a_derived_condition_has_no_authored_form(self, rbc, equation_id):
        """It exists only as a first-order condition, so there is nothing in the file to recover."""
        assert authored_sides(rbc._source_ast, equation_id) is None

    def test_an_objective_resolves_without_a_position(self, rbc):
        """A block has one objective, so its id carries no position the way a constraint's does."""
        assert authored_sides(rbc._source_ast, "FIRM.objective") is not None

    @pytest.mark.parametrize(
        "equation_id",
        [
            "TECHNOLOGY_SHOCKS.identities.-1",
            "TECHNOLOGY_SHOCKS.identities.abc",
            "TECHNOLOGY_SHOCKS.identities.",
            "TECHNOLOGY_SHOCKS.identities.99",
            "NO_SUCH_BLOCK.identities.0",
            "nonsense",
        ],
        ids=["negative", "not_a_number", "empty", "past_the_end", "unknown_block", "not_an_id"],
    )
    def test_an_id_this_module_did_not_build_recovers_nothing(self, rbc, equation_id):
        """``-1`` is the one that matters: indexing from the end would quietly return the wrong equation."""
        assert authored_sides(rbc._source_ast, equation_id) is None

    def test_a_model_built_without_a_file_recovers_nothing(self):
        assert authored_sides(None, "HOUSEHOLD.constraints.0") is None


class TestOverrides:
    @pytest.fixture(scope="class")
    def overridden(self, tmp_path_factory):
        source = """
        symbols
        {
            mc[] { latex = "\\mathcal{M}"; bounds = (0, None); };
            Y[]  { bounds = (0, None); };
            alpha { latex = "\\alpha^{\\star}"; bounds = (0, 1); };
        };

        block H
        {
            identities { mc[] = alpha * Y[]; Y[] = 1; };
            calibration { alpha = 0.3; };
        };
        """
        path = tmp_path_factory.mktemp("latex") / "overridden.gcn"
        path.write_text(source)
        return model_from_gcn(path, verbose=False)

    def test_an_override_reaches_an_authored_equation(self, overridden):
        """
        An authored equation is rebuilt from the AST without the declared assumptions.

        Sympy caches on name and assumptions, so those symbols are different objects from the solved system's.
        Resolving overrides by identity drops every one of them on any model that declares bounds, which is
        every shipped example.
        """
        rendered = overridden.to_latex()

        assert r"{\mathcal{M}}_{t}" in rendered
        assert r"\alpha^{\star}" in rendered

    def test_a_symbol_without_an_override_still_infers_its_own(self, overridden):
        assert "Y_{t}" in overridden.to_latex()


class TestToLatex:
    def test_a_caption_with_tex_specials_is_escaped(self, tmp_path):
        """An unescaped % comments out the row terminator and silently swallows the next equation."""
        path = tmp_path / "specials.gcn"
        path.write_text(
            "symbols { Y[] { }; A[] { }; };\n"
            'block H { identities { @name = "C_t & 100% of firms" Y[] = A[]; A[] = 1; }; };\n'
        )

        rendered = model_from_gcn(path, verbose=False).to_latex()

        assert r"\tag{\text{C\_t \& 100\% of firms}}" in rendered

    def test_an_authored_objective_renders_with_its_own_sides(self):
        """RBC reduces its objective away, so the one authored component to survive elsewhere needs its own model."""
        model = model_from_gcn(TEST_GCNS / "rbc_with_excluded.gcn", verbose=False)

        left, right = authored_sides(model._source_ast, "HOUSEHOLD.objective")

        assert sp.latex(left) == "U_{t}"
        assert r"\mathbb{E}_t" in sp.latex(wrap_leads_in_expectations(right))

    def test_an_authored_expectation_is_kept_where_the_author_put_it(self):
        """
        ``Three_Equation_NK.gcn`` writes ``E[][pi[1]]`` around one lead.

        Discarding that and re-deriving the placement at Add-term granularity produced an operator enclosing
        ``i[]`` and ``rn[]`` as well, which is harmless arithmetically and not what any paper writes.
        """
        model = model_from_gcn(get_example_gcn("Three_Equation_NK"), verbose=False)

        euler = next(row for row in model.to_latex().splitlines() if row.startswith("x_{t}"))
        enclosed = re.findall(r"\\mathbb\{E\}_t\\left\[(.*?)\\right\]", euler)

        assert enclosed == [r"x_{t+1}", r"\pi_{t+1}"]

    def test_the_rendered_system_matches_the_golden_file(self, rbc):
        """
        One small model, checked whole, so a regression in any rule shows as a diff.

        The per-rule tests in tests/classes/test_latex.py are the real coverage. Read the diff before
        regenerating this.
        """
        assert rbc.to_latex() + "\n" == GOLDEN.read_text()

    def test_turning_expectations_off_drops_the_operator(self, rbc):
        assert r"\mathbb{E}_t" in rbc.to_latex()
        assert r"\mathbb{E}_t" not in rbc.to_latex(expectations=False)

    def test_every_equation_becomes_one_row(self, rbc):
        rows = rbc.to_latex().splitlines()[1:-1]

        assert len(rows) == len(rbc._equation_ids)
        assert all("&=" in row for row in rows)
