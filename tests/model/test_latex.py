import pytest
import sympy as sp

from gEconpy.classes.time_aware_symbol import TimeAwareSymbol
from gEconpy.model.latex import wrap_leads_in_expectations


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
