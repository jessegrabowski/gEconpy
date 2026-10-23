import pytest
import sympy as sp
import sympy.core.parameters

from gEconpy.classes.containers import SymbolDictionary
from gEconpy.classes.time_aware_symbol import TimeAwareSymbol
from gEconpy.model.steady_state.sympy_engine import (
    propagate_steady_state_through_identities,
    resubstitute_steady_state,
    system_to_steady_state,
)
from tests._resources.cache_compiled_models import load_and_cache_model


class TestPropagateSteadyStateThroughIdentities:
    def test_propagates_through_lag_chain(self):
        """Chained identities x__lag1 = x[-1], x__lag2 = x__lag1[-1] all resolve to x_ss."""
        x = TimeAwareSymbol("x", 0)
        x_lag1 = TimeAwareSymbol("x__lag1", 0)
        x_lag2 = TimeAwareSymbol("x__lag2", 0)

        result = propagate_steady_state_through_identities(
            ss_solution_dict=SymbolDictionary({x.to_ss(): sp.Float(0.0)}),
            steady_state_equations=[x_lag1.to_ss() - x.to_ss(), x_lag2.to_ss() - x_lag1.to_ss()],
            variables=[x, x_lag1, x_lag2],
        )

        assert len(result) == 3
        assert all(float(result[v.to_ss()]) == 0.0 for v in [x, x_lag1, x_lag2])

    def test_propagates_affine_relationships(self):
        """Affine equation y = 2*x + 3 is solved when x is known."""
        x = TimeAwareSymbol("x", 0)
        y = TimeAwareSymbol("y", 0)

        result = propagate_steady_state_through_identities(
            ss_solution_dict=SymbolDictionary({x.to_ss(): sp.Float(1.0)}),
            steady_state_equations=[y.to_ss() - 2 * x.to_ss() - 3],
            variables=[x, y],
        )

        assert float(result[y.to_ss()]) == 5.0

    def test_propagates_log_exp_bijections(self):
        """Bijection log(y) = x is inverted to y = exp(x)."""
        x = TimeAwareSymbol("x", 0)
        y = TimeAwareSymbol("y", 0)

        result = propagate_steady_state_through_identities(
            ss_solution_dict=SymbolDictionary({x.to_ss(): sp.Float(0.0)}),
            steady_state_equations=[sp.log(y.to_ss()) - x.to_ss()],
            variables=[x, y],
        )

        assert float(result[y.to_ss()]) == 1.0

    def test_rejects_multi_solution_equations(self):
        """Equation x^2 = 4 has two solutions, so neither is chosen."""
        x = TimeAwareSymbol("x", 0)
        y = TimeAwareSymbol("y", 0)

        result = propagate_steady_state_through_identities(
            ss_solution_dict=SymbolDictionary({y.to_ss(): sp.Float(4.0)}),
            steady_state_equations=[x.to_ss() ** 2 - y.to_ss()],
            variables=[x, y],
        )

        assert x.to_ss() not in result

    def test_rejects_underdetermined_equations(self):
        """Equation x + y = z with only z known cannot determine x or y."""
        x = TimeAwareSymbol("x", 0)
        y = TimeAwareSymbol("y", 0)
        z = TimeAwareSymbol("z", 0)

        result = propagate_steady_state_through_identities(
            ss_solution_dict=SymbolDictionary({z.to_ss(): sp.Float(5.0)}),
            steady_state_equations=[x.to_ss() + y.to_ss() - z.to_ss()],
            variables=[x, y, z],
        )

        assert x.to_ss() not in result
        assert y.to_ss() not in result

    def test_rejects_complex_ces_production_function(self):
        """CES production function inversion is too complex to attempt."""
        Y = TimeAwareSymbol("Y", 0)
        A = TimeAwareSymbol("A", 0)
        x1 = TimeAwareSymbol("x1", 0)
        x2 = TimeAwareSymbol("x2", 0)
        alpha, psi = sp.Symbol("alpha"), sp.Symbol("psi")

        ces_aggregator = (
            alpha ** (1 / psi) * x1.to_ss() ** ((psi - 1) / psi)
            + (1 - alpha) ** (1 / psi) * x2.to_ss() ** ((psi - 1) / psi)
        ) ** (psi / (psi - 1))

        result = propagate_steady_state_through_identities(
            ss_solution_dict=SymbolDictionary(
                {
                    Y.to_ss(): sp.Float(1.0),
                    A.to_ss(): sp.Float(1.0),
                    x1.to_ss(): sp.Float(0.5),
                }
            ),
            steady_state_equations=[Y.to_ss() - A.to_ss() * ces_aggregator],
            variables=[Y, A, x1, x2],
        )

        assert x2.to_ss() not in result

    def test_empty_input_with_underdetermined_system(self):
        """Empty input with multiple unknowns per equation returns empty."""
        x = TimeAwareSymbol("x", 0)
        y = TimeAwareSymbol("y", 0)

        result = propagate_steady_state_through_identities(SymbolDictionary(), [x.to_ss() + y.to_ss()], [x, y])
        assert len(result) == 0

    def test_solves_ar1_log_process(self):
        """AR(1) in logs: log(A) = rho*log(A[-1]) + epsilon solves to A_ss = 1 when epsilon_ss = 0."""
        A = TimeAwareSymbol("A", 0)
        rho = sp.Symbol("rho")

        steady_state_eq = sp.log(A.to_ss()) - rho * sp.log(A.to_ss())

        result = propagate_steady_state_through_identities(
            ss_solution_dict=SymbolDictionary(),
            steady_state_equations=[steady_state_eq],
            variables=[A],
        )

        assert A.to_ss() in result
        assert float(result[A.to_ss()]) == 1.0


def test_resubstitution_names_other_steady_states_without_changing_values():
    """Every relationship must still evaluate to the solved steady state once the names are folded back in."""
    model = load_and_cache_model("rbc_2_block_ss.gcn")
    relationships = model.steady_state_relationships
    values = {**model.parameters().to_sympy(), **model.steady_state(verbose=False, progressbar=False).to_sympy()}

    resubstituted = resubstitute_steady_state(relationships)

    assert [eq.lhs for eq in resubstituted] == [eq.lhs for eq in relationships]
    assert sum(sp.count_ops(eq.rhs) for eq in resubstituted) < sum(sp.count_ops(eq.rhs) for eq in relationships)
    for eq in resubstituted:
        assert float(eq.rhs.subs(values)) == pytest.approx(float(values[eq.lhs]))


def test_resubstitution_rebuilds_a_piecewise_relationship():
    """A Piecewise's condition pairs take no evaluate keyword, which once made the rebuild raise."""
    alpha, beta = sp.symbols("alpha beta", positive=True)
    a_ss, b_ss = sp.symbols("A_ss B_ss")
    relationships = [
        sp.Eq(a_ss, alpha + beta, evaluate=False),
        sp.Eq(b_ss, sp.Piecewise((alpha + beta, alpha > 1), (beta, True)), evaluate=False),
    ]

    resubstituted = resubstitute_steady_state(relationships)

    assert resubstituted[1].rhs == sp.Piecewise((a_ss, alpha > 1), (beta, True))


def test_resubstitution_keeps_the_sympy_cache(monkeypatch):
    """Switching sympy's global evaluate flag clears its cache, so the rebuild must not touch the flag."""
    clears = []
    monkeypatch.setattr(sympy.core.parameters, "clear_cache", lambda: clears.append(None))
    alpha, beta = sp.symbols("alpha beta", positive=True)
    a_ss, b_ss = sp.symbols("A_ss B_ss")
    relationships = [
        sp.Eq(a_ss, alpha + beta, evaluate=False),
        sp.Eq(b_ss, (alpha + beta) ** 2, evaluate=False),
    ]

    resubstituted = resubstitute_steady_state(relationships)

    assert resubstituted[1].rhs == sp.Pow(a_ss, 2, evaluate=False)
    assert clears == []


def test_system_to_steady_state_collapses_time_indices_and_zeros_shocks():
    x, eps = TimeAwareSymbol("x", 0), TimeAwareSymbol("eps", 0)
    rho = sp.Symbol("rho")

    (steady_state_eq,) = system_to_steady_state([x - rho * x.set_t(-1) - eps + x.set_t(1) ** 2], [eps])

    assert sp.expand(steady_state_eq - (x.to_ss() ** 2 + x.to_ss() * (1 - rho))) == 0
