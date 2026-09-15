import pickle

import pytest
import sympy as sp


# Since sympy 1.14.0, complex values are mpmath mpc numbers.
from mpmath.ctx_mp_python import _mpc

from gEconpy.classes.containers import SteadyStateResults, SymbolDictionary
from gEconpy.classes.time_aware_symbol import TimeAwareSymbol

C = TimeAwareSymbol("C", 0, positive=True)
A = TimeAwareSymbol("A", 1, negative=True)
r = TimeAwareSymbol("r", -1, imaginary=True)
alpha = sp.Symbol("alpha", real=True)


@pytest.fixture
def symbol_dict():
    return SymbolDictionary({C: 1, A: -1, r: 2j, alpha: 0.3})


class TestSymbolDictErrors:
    def test_raises_on_invalid_keys(self):
        with pytest.raises(KeyError):
            SymbolDictionary({1: 3})

    @pytest.mark.parametrize("keys", [{"A": 3, sp.Symbol("B"): 2}, {sp.Symbol("B"): 2, "A": 3}])
    def test_raises_on_mixed_keys(self, keys):
        with pytest.raises(KeyError, match="Cannot mix"):
            SymbolDictionary(keys)

    @pytest.mark.parametrize(
        ("initial", "new_key"), [({sp.Symbol("A"): 3}, "B"), ({"A": 3}, sp.Symbol("B"))], ids=["sympy", "string"]
    )
    def test_setitem_raises_on_wrong_key_type(self, initial, new_key):
        d = SymbolDictionary(initial)
        with pytest.raises(KeyError):
            d[new_key] = 4

    def test_pipe_merge_errors_with_non_dict_other(self):
        d = SymbolDictionary({"A": 4})
        with pytest.raises(TypeError, match="__or__ not defined on non-dictionary objects"):
            d | {1, 2, 3}

    def test_pipe_merge_errors_on_mixed_modes(self, symbol_dict):
        with pytest.raises(ValueError, match="Cannot merge"):
            SymbolDictionary({"A": 3, "B": 4}) | symbol_dict


class TestSymbolDictionary:
    def test_is_variable(self, symbol_dict):
        assert symbol_dict._is_variable == {"C": True, "A": True, "r": True, "alpha": False}

    def test_convert_to_string(self, symbol_dict):
        d = symbol_dict.to_string()
        assert list(d.keys()) == ["C_t", "A_tp1", "r_tm1", "alpha"]
        assert not d.is_sympy

        symbol_dict.to_string(inplace=True)
        assert list(symbol_dict.keys()) == ["C_t", "A_tp1", "r_tm1", "alpha"]
        assert not symbol_dict.is_sympy

    def test_convert_to_sympy(self):
        d = SymbolDictionary({"a": 2, "b": 3}).to_sympy()
        assert list(d.keys()) == [sp.Symbol("a"), sp.Symbol("b")]
        assert d.is_sympy

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("alpha", sp.Symbol("alpha")),
            ("x_y_t", TimeAwareSymbol("x_y", 0)),
            ("K_ss", TimeAwareSymbol("K", "ss")),
            ("beta_tp1", TimeAwareSymbol("beta", 1)),
            ("beta_tm1", TimeAwareSymbol("beta", -1)),
        ],
    )
    def test_to_sympy_parses_time_suffix(self, name, expected):
        d = SymbolDictionary({name: 1.0}).to_sympy()
        (key,) = d.keys()
        assert key == expected
        assert type(key) is type(expected)

    def test_to_sympy_overrides_keep_parameters_plain(self):
        d = SymbolDictionary({"K_ss": 1.0, "alpha": 0.3}).to_sympy(
            new_is_variable={"K_ss": False}, new_assumptions={"alpha": {"positive": True}}
        )
        K_ss, alpha = d.keys()

        assert type(K_ss) is sp.Symbol
        assert K_ss.name == "K_ss"
        assert alpha.is_positive

    def test_string_key_with_time_suffix_becomes_time_aware(self, symbol_dict):
        d = symbol_dict.to_string()
        d["F_ss"] = 3

        d.to_sympy(inplace=True)
        assert TimeAwareSymbol("F", "ss") in d

    def test_plain_symbol_with_time_suffix_survives_round_trip(self, symbol_dict):
        F_ss = sp.Symbol("F_ss")
        d = symbol_dict.copy()
        d[F_ss] = 3

        d.to_string(inplace=True)
        assert "F_ss" in d
        d.to_sympy(inplace=True)
        assert F_ss in d

    def test_copy_is_new_object(self, symbol_dict):
        assert symbol_dict.copy() is not symbol_dict

    @pytest.mark.parametrize("method", ["copy", "values_to_float", "float_to_values"])
    def test_copy_keeps_subclass_and_attributes(self, method):
        results = SteadyStateResults({C: 1.0})
        results.success = True

        copied = getattr(results, method)()
        assert isinstance(copied, SteadyStateResults)
        assert copied.success

    def test_step_forward_leaves_steady_state_keys(self):
        d = SymbolDictionary({C.to_ss(): 1.0, C: 2.0})
        assert d.step_forward() == {C.to_ss(): 1.0, C.step_forward(): 2.0}

    @pytest.mark.parametrize("method", ["to_string", "to_sympy", "values_to_float"])
    def test_assumptions_preserved(self, symbol_dict, method):
        assumptions = symbol_dict._assumptions.copy()
        assert getattr(symbol_dict, method)()._assumptions == assumptions

    def test_join_with_pipe(self, symbol_dict):
        F = TimeAwareSymbol("F", "ss")
        other = SymbolDictionary({F: 3})

        merged = symbol_dict | other
        merged.sort_keys(inplace=True)

        assert list(merged.keys()) == [A, C, F, alpha, r]
        assert merged._assumptions == symbol_dict._assumptions | other._assumptions
        assert merged._is_variable == symbol_dict._is_variable | other._is_variable

    @pytest.mark.parametrize(
        ("method", "expected_keys"),
        [
            ("step_forward", ["C_tp1", "A_tp2", "r_t", "alpha"]),
            ("step_backward", ["C_tm1", "A_t", "r_tm2", "alpha"]),
            ("to_ss", ["C_ss", "A_ss", "r_ss", "alpha"]),
        ],
    )
    def test_time_shifts(self, symbol_dict, method, expected_keys):
        shifted = getattr(symbol_dict, method)()
        assert list(shifted.to_string().keys()) == expected_keys

        getattr(symbol_dict, method)(inplace=True)
        assert list(symbol_dict.to_string().keys()) == expected_keys

    @pytest.mark.parametrize(
        ("convert", "expected"),
        [(lambda d: d, [A, C, alpha, r]), (lambda d: d.to_string(), ["A_tp1", "C_t", "alpha", "r_tm1"])],
        ids=["sympy_mode", "string_mode"],
    )
    def test_sort_dictionary(self, symbol_dict, convert, expected):
        d = convert(symbol_dict)

        assert list(d.sort_keys().keys()) == expected

        d.sort_keys(inplace=True)
        assert list(d.keys()) == expected

    def test_sequential_pipe_from_empty_rejects_mixed_modes(self, symbol_dict):
        string_dict = SymbolDictionary({"A": 3, "B": 4})
        merged = SymbolDictionary() | string_dict
        with pytest.raises(ValueError, match="Cannot merge"):
            merged | symbol_dict

    def test_convert_values(self, symbol_dict):
        d_sympy = symbol_dict.float_to_values()
        assert all(isinstance(value, sp.core.Number | _mpc) for value in d_sympy.values())

        d_float = d_sympy.values_to_float()
        assert all(isinstance(value, int | float | _mpc) for value in d_float.values())

    def test_convert_values_inplace(self, symbol_dict):
        symbol_dict.float_to_values(inplace=True)
        assert all(isinstance(value, sp.core.Number | _mpc) for value in symbol_dict.values())

        symbol_dict.values_to_float(inplace=True)
        assert all(isinstance(value, int | float | _mpc) for value in symbol_dict.values())

    def test_not_inplace_conversion_leaves_original_alone(self, symbol_dict):
        symbol_dict.to_string()
        assert all(isinstance(key, sp.Symbol) for key in symbol_dict)
        assert symbol_dict.is_sympy

    def test_update_preserves_assumptions(self):
        X = TimeAwareSymbol("X", 0, positive=True)
        Y = TimeAwareSymbol("Y", "ss", real=True)

        d1 = SymbolDictionary({C: 1, A: 2})
        d1.update(SymbolDictionary({X: 3, Y: 4}))

        assert len(d1) == 4
        assert d1[X] == 3
        assert d1[Y] == 4
        assert d1._assumptions["X"]["positive"]
        assert d1._assumptions["Y"]["real"]
        assert d1._is_variable["X"]
        assert d1._is_variable["Y"]

    def test_update_with_string_mode_preserves_assumptions(self, symbol_dict):
        d1 = symbol_dict.to_string()
        d2 = SymbolDictionary({TimeAwareSymbol("X", 0, positive=True): 3}).to_string()

        original_assumptions = d1._assumptions.copy()
        d1.update(d2)

        assert set(d1._assumptions) == set(original_assumptions) | {"X"}

    def test_update_with_plain_dict(self):
        d1 = SymbolDictionary({"A": 1, "B": 2})
        d1.update({"C": 3})
        assert d1 == {"A": 1, "B": 2, "C": 3}

    def test_update_kwargs_respect_mode(self):
        d1 = SymbolDictionary({"A": 1})
        d1.update(B=2)
        assert d1 == {"A": 1, "B": 2}

        with pytest.raises(KeyError, match="Cannot add string key"):
            SymbolDictionary({C: 1}).update(x=2)

    def test_pipe_with_plain_dict(self):
        merged = SymbolDictionary({C: 1}) | {A: 2}

        assert isinstance(merged, SymbolDictionary)
        assert merged == {C: 1, A: 2}
        assert merged.is_sympy
        assert merged._assumptions["A"]["negative"]

    @pytest.mark.parametrize(
        "merge", [lambda d: SymbolDictionary() | d, lambda d: d | SymbolDictionary()], ids=["empty_left", "empty_right"]
    )
    def test_pipe_with_empty_keeps_other_side(self, symbol_dict, merge):
        merged = merge(symbol_dict)

        assert merged == symbol_dict
        assert merged.is_sympy
        assert merged._assumptions == symbol_dict._assumptions

    def test_pickle_round_trip(self, symbol_dict):
        restored = pickle.loads(pickle.dumps(symbol_dict))
        assert restored == symbol_dict
        assert restored.is_sympy
        assert restored._assumptions == symbol_dict._assumptions
        assert restored._is_variable == symbol_dict._is_variable

    def test_pickle_round_trip_keeps_subclass_attributes(self):
        results = SteadyStateResults({C: 1.0})
        results.success = True
        restored = pickle.loads(pickle.dumps(results))
        assert isinstance(restored, SteadyStateResults)
        assert restored.success
