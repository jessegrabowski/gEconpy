import pickle

import pytest
import sympy as sp

from gEconpy.classes.time_aware_symbol import TimeAwareSymbol
from gEconpy.utilities import diff_through_time, step_equation_backward, step_equation_forward

x_t = TimeAwareSymbol("x", 0)
x_tp1 = TimeAwareSymbol("x", 1)
x_tm1 = TimeAwareSymbol("x", -1)


class TestTimeAwareSymbol:
    @pytest.mark.parametrize(
        ("time_index", "name"),
        [(-2, "x_t-2"), (-1, "x_t-1"), (0, "x_t"), (1, "x_t+1"), (2, "x_t+2"), ("ss", "x_ss")],
    )
    def test_name_from_time_index(self, time_index, name):
        symbol = TimeAwareSymbol("x", time_index)
        assert symbol.name == name
        assert symbol.base_name == "x"
        assert symbol.time_index == time_index

    def test_safe_name(self):
        assert x_tp1.safe_name == "x_tp1"
        assert x_tm1.safe_name == "x_tm1"

    def test_stepping(self):
        assert x_t.step_forward() == x_tp1
        assert x_t.step_backward() == x_tm1
        assert x_tp1.to_ss() == x_tm1.to_ss()
        assert x_t.to_ss().exit_ss() == x_t
        assert x_tm1.set_t(1) == x_tp1

    def test_set_t_rejects_unknown_string(self):
        with pytest.raises(ValueError, match="integer or 'ss'"):
            x_t.set_t("foo")

    def test_stepping_keeps_assumptions(self):
        positive_x = TimeAwareSymbol("x", 0, positive=True)
        assert positive_x.step_forward().assumptions0 == positive_x.assumptions0
        assert positive_x.step_forward() != x_tp1

    def test_pickle_round_trip(self):
        symbol = TimeAwareSymbol("y_z", -3, positive=True)
        restored = pickle.loads(pickle.dumps(symbol))
        assert restored == symbol
        assert restored.name == "y_z_t-3"
        assert restored.assumptions0 == symbol.assumptions0


class TestEquationStepping:
    def test_step_equation_backward(self):
        eq = x_t + x_tp1 + x_tm1
        assert step_equation_backward(eq) == x_tm1 + x_t + TimeAwareSymbol("x", -2)

    def test_step_equation_forward(self):
        eq = x_t + x_tp1 + x_tm1
        assert step_equation_forward(eq) == x_tp1 + TimeAwareSymbol("x", 2) + x_t


class TestDiffThroughTime:
    def test_geometric_sum(self):
        beta = sp.Symbol("beta")
        X = sum(TimeAwareSymbol("x", t) for t in range(-10, 10))
        assert diff_through_time(X, x_t, beta) == sum(beta**t for t in range(11))

    def test_gap_in_time_indices(self):
        # x_t + a * x_{t-2} contributes at shift 0 and shift 2 but not at shift 1, so the summation must keep
        # walking past a zero derivative.
        a, beta = sp.symbols("a beta")
        eq = x_t + a * TimeAwareSymbol("x", -2)
        result = diff_through_time(eq, x_t, beta)
        assert sp.simplify(result - (1 + a * beta**2)) == 0

    def test_no_appearance(self):
        eq = TimeAwareSymbol("y", 0) + sp.Symbol("c")
        assert diff_through_time(eq, x_t, sp.Symbol("beta")) == 0

    def test_only_forward_appearances(self):
        eq = x_t + sp.Symbol("a") * TimeAwareSymbol("x", 3)
        assert diff_through_time(eq, x_t, sp.Symbol("beta")) == 1
