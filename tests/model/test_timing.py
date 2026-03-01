import pytest
import sympy as sp

from gEconpy.classes.time_aware_symbol import TimeAwareSymbol
from gEconpy.model.timing import (
    classify_variables_by_timing,
    collect_time_aware_atoms,
    make_all_variable_time_combinations,
    natural_sort_key,
)
from tests._resources.cache_compiled_models import load_and_cache_model


def _names(symbols):
    return [s.base_name for s in symbols]


def _times(symbols):
    return [s.time_index for s in symbols]


class TestNaturalSortKey:
    @pytest.mark.parametrize(
        "names, expected",
        [
            (["c", "a", "b"], ["a", "b", "c"]),
            (["x10", "x2", "x1"], ["x1", "x2", "x10"]),
            (["beta", "alpha2", "alpha10", "alpha1"], ["alpha1", "alpha2", "alpha10", "beta"]),
        ],
        ids=["alphabetic", "numeric_suffix", "mixed"],
    )
    def test_ordering(self, names, expected):
        result = sorted([TimeAwareSymbol(n, 0) for n in names], key=natural_sort_key)
        assert _names(result) == expected


class TestCollectTimeAwareAtoms:
    def test_returns_all_time_aware_symbols(self):
        x_t, x_tm1, y_tp1, alpha = (
            TimeAwareSymbol("x", 0),
            TimeAwareSymbol("x", -1),
            TimeAwareSymbol("y", 1),
            sp.Symbol("alpha"),
        )
        atoms = collect_time_aware_atoms([x_t - alpha * x_tm1, y_tp1])
        assert atoms == {x_t, x_tm1, y_tp1}

    def test_empty_input(self):
        assert collect_time_aware_atoms([]) == set()


class TestClassifyVariablesByTiming:
    def test_separates_by_time_and_type(self):
        x_t, x_tm1, y_t, y_tp1, eps = (
            TimeAwareSymbol("x", 0),
            TimeAwareSymbol("x", -1),
            TimeAwareSymbol("y", 0),
            TimeAwareSymbol("y", 1),
            TimeAwareSymbol("eps", 0),
        )
        tm1, t, tp1, shocks = classify_variables_by_timing([x_t - x_tm1 + eps, y_tp1 - y_t], ["eps"])

        assert _names(tm1) == ["x"] and _times(tm1) == [-1]
        assert _names(t) == ["x", "y"] and _times(t) == [0, 0]
        assert _names(tp1) == ["y"] and _times(tp1) == [1]
        assert _names(shocks) == ["eps"]

    def test_absent_times_are_empty(self):
        """Variables only at t produce empty tm1 and tp1 lists."""
        x_t, y_t = TimeAwareSymbol("x", 0), TimeAwareSymbol("y", 0)
        tm1, _, tp1, _ = classify_variables_by_timing([x_t + y_t], [])
        assert tm1 == [] and tp1 == []

    def test_shock_with_no_endogenous_at_same_time(self):
        eps = TimeAwareSymbol("eps", 0)
        tm1, t, tp1, shocks = classify_variables_by_timing([eps], ["eps"])
        assert tm1 == [] and t == [] and tp1 == []
        assert _names(shocks) == ["eps"]

    def test_uses_natural_sort_order(self):
        syms = [TimeAwareSymbol("x10", 0), TimeAwareSymbol("x2", 0), TimeAwareSymbol("x1", 0)]
        _, t, _, _ = classify_variables_by_timing([sum(syms)], [])
        assert _names(t) == ["x1", "x2", "x10"]

    def test_multiple_shocks_sorted(self):
        e1, e2 = TimeAwareSymbol("eps_z", 0), TimeAwareSymbol("eps_a", 0)
        _, _, _, shocks = classify_variables_by_timing([e1 + e2 + TimeAwareSymbol("x", 0)], ["eps_z", "eps_a"])
        assert _names(shocks) == ["eps_a", "eps_z"]

    def test_rejects_unexpected_time_indices(self):
        x_t2 = TimeAwareSymbol("x", 2)
        with pytest.raises(ValueError, match="unexpected time indices"):
            classify_variables_by_timing([x_t2], [])

    def test_on_compiled_model(self):
        mod = load_and_cache_model("one_block_1.gcn")
        shock_names = [s.base_name for s in mod.shocks]
        tm1, t, tp1, _shocks = classify_variables_by_timing(mod.equations, shock_names)

        model_var_names = {v.base_name for v in mod.variables}
        for group, expected_t in [(tm1, -1), (t, 0), (tp1, 1)]:
            assert all(v.base_name in model_var_names for v in group)
            assert all(v.time_index == expected_t for v in group)


class TestMakeAllVariableTimeCombinations:
    def test_produces_three_equal_length_lists(self):
        variables = [TimeAwareSymbol("x", 0), TimeAwareSymbol("y", 0)]
        lags, now, leads = make_all_variable_time_combinations(variables)
        assert len(lags) == len(now) == len(leads) == 2
        assert _times(lags) == [-1, -1]
        assert _times(now) == [0, 0]
        assert _times(leads) == [1, 1]

    def test_normalizes_to_time_zero(self):
        variables = [TimeAwareSymbol("x", -1), TimeAwareSymbol("y", 1)]
        _, now, _ = make_all_variable_time_combinations(variables)
        assert _names(now) == ["x", "y"]
        assert _times(now) == [0, 0]

    def test_deduplicates_same_base_name(self):
        variables = [TimeAwareSymbol("x", 0), TimeAwareSymbol("x", -1), TimeAwareSymbol("x", 1)]
        _, now, _ = make_all_variable_time_combinations(variables)
        assert len(now) == 1

    def test_preserves_caller_ordering(self):
        variables = [TimeAwareSymbol("c", 0), TimeAwareSymbol("a", 0), TimeAwareSymbol("b", 0)]
        lags, now, leads = make_all_variable_time_combinations(variables)
        assert _names(now) == ["c", "a", "b"]
        assert _names(lags) == ["c", "a", "b"]
        assert _names(leads) == ["c", "a", "b"]

    def test_on_compiled_model(self):
        mod = load_and_cache_model("one_block_1.gcn")
        _lags, now, _leads = make_all_variable_time_combinations(mod.variables)
        assert len(now) == len(mod.variables)
        assert set(now) == {v.set_t(0) for v in mod.variables}
