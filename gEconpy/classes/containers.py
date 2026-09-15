from collections import defaultdict
from typing import Any, cast

import sympy as sp

from sympy.polys.domains.mpelements import ComplexElement

from gEconpy.classes.time_aware_symbol import TimeAwareSymbol

SAFE_STRING_TO_INDEX_DICT = {"ss": "ss", "tp1": 1, "tm1": -1, "t": 0}


def safe_string_to_sympy(s, assumptions=None):
    """
    Convert a string to a symbol, using a trailing time suffix to decide which kind of symbol to make.

    A name ending in ``_t``, ``_tp1``, ``_tm1`` or ``_ss`` becomes a
    :class:`~gEconpy.classes.time_aware_symbol.TimeAwareSymbol` with the matching time index. Any other name
    becomes a plain :class:`~sympy.core.symbol.Symbol`.

    Parameters
    ----------
    s : str or sp.Symbol
        Name to convert. A symbol is returned unchanged.
    assumptions : dict mapping str to dict, optional
        Sympy assumptions to attach, keyed by symbol name. Defaults to no assumptions.

    Returns
    -------
    symbol : sp.Symbol
        The converted symbol.
    """
    if isinstance(s, sp.Symbol):
        return s

    assumptions = assumptions or defaultdict(dict)

    *name, time_index_str = s.split("_")
    if time_index_str not in [str(x) for x in SAFE_STRING_TO_INDEX_DICT]:
        name.append(time_index_str)
        name = "_".join(name)
        return sp.Symbol(name, **assumptions[name])

    name = "_".join(name)
    time_index = SAFE_STRING_TO_INDEX_DICT[time_index_str]
    return TimeAwareSymbol(name, time_index, **assumptions.get(name, {}))


def symbol_to_string(symbol: str | sp.Symbol):
    """
    Convert a symbol to its name, returning a string unchanged.

    Parameters
    ----------
    symbol : str or sp.Symbol
        Symbol to convert.

    Returns
    -------
    name : str
        The symbol name. Time aware symbols return their safe name, with ``+`` and ``-`` replaced by ``p`` and ``m``.
    """
    if isinstance(symbol, str):
        return symbol
    return symbol.safe_name if isinstance(symbol, TimeAwareSymbol) else symbol.name


def string_keys_to_sympy(d, assumptions=None, is_variable=None):
    """
    Convert the string keys of a dictionary to symbols, leaving the values alone.

    Parameters
    ----------
    d : dict
        Dictionary with string or symbol keys.
    assumptions : dict mapping str to dict, optional
        Sympy assumptions to attach, keyed by symbol name. Defaults to no assumptions.
    is_variable : dict mapping str to bool, optional
        Marks which names are model variables, and so eligible to become time aware symbols. Names absent from
        this mapping are treated as variables.

    Returns
    -------
    result : dict
        Dictionary with symbol keys.
    """

    def has_time_suffix(s):
        suffixes = ["_t", "_tp1", "_tm1", "_ss"]
        return any(s.endswith(suffix) for suffix in suffixes)

    result = {}
    assumptions = assumptions if assumptions is not None else defaultdict(dict)
    is_variable = is_variable if is_variable is not None else defaultdict(bool)

    for key, value in d.items():
        if isinstance(key, sp.Symbol):
            result[key] = value
            continue

        if not is_variable.get(key, True) or not has_time_suffix(key):
            result[sp.Symbol(key, **assumptions.get(key, {}))] = value
            continue

        new_key = safe_string_to_sympy(key, assumptions)
        result[new_key] = value

    return result


def sympy_keys_to_strings(d):
    """
    Convert the symbol keys of a dictionary to strings, leaving the values alone.

    Parameters
    ----------
    d : dict
        Dictionary with symbol keys.

    Returns
    -------
    result : dict mapping str to object
        Dictionary with string keys.
    """
    result = {}
    for key in d:
        result[symbol_to_string(key)] = d[key]

    return result


def sympy_number_values_to_floats(d: dict[sp.Symbol, Any]):
    """
    Replace sympy numeric values with Python floats or complex numbers, in place.

    Parameters
    ----------
    d : dict
        Dictionary whose values may be sympy numbers.

    Returns
    -------
    d : dict
        The same dictionary, with numeric values converted.
    """
    for var, value in d.items():
        if isinstance(value, sp.core.Number):
            d[var] = float(value)
        elif isinstance(value, ComplexElement):
            d[var] = complex(value)
    return d


def float_values_to_sympy_float(d: dict[sp.Symbol, Any]):
    """
    Replace Python numeric values with sympy numbers, in place.

    Parameters
    ----------
    d : dict
        Dictionary whose values may be Python numbers.

    Returns
    -------
    d : dict
        The same dictionary, with numeric values converted.
    """
    for var, value in d.items():
        if isinstance(value, float | int):
            d[var] = sp.Float(value)
        elif isinstance(value, complex):
            d[var] = sp.CC(value)

    return d


def sort_dictionary(d):
    """
    Return a new dictionary with the same items, ordered by sorted key.

    Parameters
    ----------
    d : dict
        Dictionary to sort. Keys must be mutually comparable.

    Returns
    -------
    result : dict
        The sorted dictionary.
    """
    result = {}
    sorted_keys = sorted(d.keys())
    for key in sorted_keys:
        result[key] = d[key]

    return result


def _unpickle_symbol_dictionary(cls, items, is_sympy, assumptions, is_variable, extra_attrs):
    """Reconstruct a SymbolDictionary (or subclass) from pickled state."""
    d = cls.__new__(cls)
    # Initialize internal state before any __setitem__ calls
    d.is_sympy = is_sympy
    d._assumptions = assumptions
    d._is_variable = is_variable
    for k, v in extra_attrs.items():
        setattr(d, k, v)
    dict.update(d, items)
    return d


class SymbolDictionary(dict):
    """
    Dictionary whose keys are either all strings or all sympy symbols, and which converts between the two.

    The dictionary remembers the assumptions of every sympy key it stores, so converting to string keys with
    :meth:`to_string` and back with :meth:`to_sympy` returns symbols carrying the same assumptions. String keys and
    sympy keys cannot be mixed.

    Parameters
    ----------
    *args
        Positional arguments forwarded to :class:`dict`.
    **kwargs
        Keyword arguments forwarded to :class:`dict`.

    Raises
    ------
    KeyError
        If the keys are not all strings or all sympy symbols.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.is_sympy: bool = False
        self._assumptions: dict = {}
        self._is_variable: dict = {}

        keys = list(self.keys())
        if any(not isinstance(x, sp.Symbol | str) for x in keys):
            raise KeyError("All keys should be either string or Sympy symbols")

        if len(keys) > 0:
            self.is_sympy = not isinstance(keys[0], str)

            if self.is_sympy and any(isinstance(x, str) for x in keys):
                raise KeyError("Cannot mix sympy and string keys")
            if not self.is_sympy and any(isinstance(x, sp.Symbol) for x in keys):
                raise KeyError("Cannot mix sympy and string keys")

        self._save_assumptions(keys)
        self._save_is_variable(keys)

    def __reduce__(self):
        # Default dict pickling calls __setitem__ before __init__, which crashes
        # because _assumptions doesn't exist yet. Serialize the raw dict contents
        # and metadata, then reconstruct via __new__ + dict.update (bypassing __setitem__).
        base_attrs = {"is_sympy", "_assumptions", "_is_variable"}
        extra_attrs = {k: v for k, v in self.__dict__.items() if k not in base_attrs}
        return (
            _unpickle_symbol_dictionary,
            (type(self), dict(self), self.is_sympy, self._assumptions, self._is_variable, extra_attrs),
        )

    def __or__(self, other: dict):
        if not isinstance(other, dict):
            raise TypeError("__or__ not defined on non-dictionary objects")
        if not isinstance(other, SymbolDictionary):
            other = SymbolDictionary(other)

        d_copy = cast(SymbolDictionary, self.copy())

        # If one dict or the other is empty, only merge assumptions
        if len(d_copy.keys()) == 0:
            other_copy = cast(SymbolDictionary, other.copy())
            other_copy._assumptions.update(self._assumptions)
            other_copy._is_variable.update(self._is_variable)
            return other_copy

        if len(other.keys()) == 0:
            d_copy._assumptions.update(other._assumptions)
            d_copy._is_variable.update(self._is_variable)
            return d_copy

        # If both are populated but of different types, raise an error
        if other.is_sympy != self.is_sympy:
            raise ValueError("Cannot merge string-mode SymbolDictionary with sympy-mode SymbolDictionary")

        # Full merge
        other_assumptions = getattr(other, "_assumptions", {})
        other_is_variable = getattr(other, "_is_variable", {})

        d_copy.update(other)
        d_copy._assumptions.update(other_assumptions)
        d_copy._is_variable.update(other_is_variable)

        return d_copy

    def copy(self) -> "SymbolDictionary":
        """Return a shallow copy that keeps the sympy flag and the assumptions."""
        new_d = SymbolDictionary(super().copy())
        new_d.is_sympy = self.is_sympy
        new_d._assumptions = self._assumptions
        new_d._is_variable = self._is_variable

        return new_d

    def _save_assumptions(self, keys):
        if not self.is_sympy:
            return
        if not isinstance(keys, list):
            keys = [keys]
        for key in keys:
            if isinstance(key, TimeAwareSymbol):
                self._assumptions[key.base_name] = key.assumptions0
            else:
                self._assumptions[key.name] = key.assumptions0

    def _save_is_variable(self, keys):
        if not self.is_sympy:
            return
        if not isinstance(keys, list):
            keys = [keys]
        for key in keys:
            if isinstance(key, TimeAwareSymbol):
                self._is_variable[key.base_name] = True
            else:
                self._is_variable[key.name] = False

    def __setitem__(self, key, value):
        if len(self.keys()) == 0:
            self.is_sympy = isinstance(key, sp.Symbol)
        elif self.is_sympy and not isinstance(key, sp.Symbol):
            raise KeyError("Cannot add string key to dictionary in sympy mode")
        elif not self.is_sympy and isinstance(key, sp.Symbol):
            raise KeyError("Cannot add sympy key to dictionary in string mode")

        super().__setitem__(key, value)
        self._save_assumptions(key)
        self._save_is_variable(key)

    def _clean_update(self, d):
        self.clear()
        self._assumptions.clear()
        self._is_variable.clear()

        self.update(d)

    def update(self, other=None, **kwargs):
        """
        Update the dictionary with the key-value pairs of another mapping.

        Unlike :meth:`dict.update`, this also merges the stored assumptions and variable flags when the other
        mapping is a SymbolDictionary.

        Parameters
        ----------
        other : dict, optional
            Mapping to take key-value pairs from. Defaults to an empty mapping.
        **kwargs
            Additional key-value pairs to add.
        """
        if other is None:
            other = {}

        # Handle SymbolDictionary specially to preserve assumptions
        if isinstance(other, SymbolDictionary):
            super().update(other)
            self._assumptions.update(other._assumptions)
            self._is_variable.update(other._is_variable)
            if len(self) == len(other) or not self.is_sympy:
                self.is_sympy = other.is_sympy
        else:
            super().update(other)

        if kwargs:
            super().update(kwargs)

    def to_sympy(self, inplace=False, new_assumptions=None, new_is_variable=None):
        new_assumptions = new_assumptions if new_assumptions is not None else {}
        new_is_variable = new_is_variable if new_is_variable is not None else {}

        assumptions = self._assumptions.copy()
        is_variable = self._is_variable.copy()

        assumptions.update(new_assumptions)
        is_variable.update(new_is_variable)

        d = SymbolDictionary(string_keys_to_sympy(self, assumptions, is_variable))

        if inplace:
            self._clean_update(d)
            return None

        return d

    def _step_dict_keys(self, func_name):
        is_sympy = self.is_sympy
        dict_copy = self.copy()
        if not is_sympy:
            dict_copy.to_sympy(inplace=True)

        d = {}
        for k, v in dict_copy.items():
            if hasattr(k, func_name):
                d[getattr(k, func_name)()] = v
            else:
                d[k] = v

        d = SymbolDictionary(d)
        if not is_sympy:
            d.to_string(inplace=True)

        return d

    def step_backward(self, inplace=False):
        d = self._step_dict_keys("step_backward")

        if inplace:
            self._clean_update(d)
            return None
        return d

    def step_forward(self, inplace=False):
        d = self._step_dict_keys("step_forward")

        if inplace:
            self._clean_update(d)
            return None
        return d

    def to_ss(self, inplace=False):
        d = self._step_dict_keys("to_ss")

        if inplace:
            self._clean_update(d)
            return None
        return d

    def to_string(self, inplace=False):
        copy_dict = self.copy()
        d = SymbolDictionary(sympy_keys_to_strings(copy_dict))
        d._assumptions = copy_dict._assumptions.copy()
        d._is_variable = copy_dict._is_variable.copy()

        if inplace:
            self._clean_update(d)
            return None

        return d

    def sort_keys(self, inplace=False):
        is_sympy = self.is_sympy
        d = SymbolDictionary(sort_dictionary(self.copy().to_string()))
        d._assumptions = self._assumptions.copy()
        d._is_variable = self._is_variable.copy()

        if is_sympy:
            d = d.to_sympy()

        if inplace:
            self._clean_update(d)
            return None

        return d

    def values_to_float(self, inplace=False):
        d = self.copy()
        d = sympy_number_values_to_floats(d)
        d._assumptions = self._assumptions.copy()
        d._is_variable = self._is_variable.copy()

        if inplace:
            self._clean_update(d)
            return None

        return d

    def float_to_values(self, inplace=False):
        d = self.copy()
        d = float_values_to_sympy_float(d)
        d._assumptions = self._assumptions.copy()
        d._is_variable = self._is_variable.copy()

        if inplace:
            self._clean_update(d)
            return None

        return d


class SteadyStateResults(SymbolDictionary):
    """
    Steady-state values of the model variables, keyed by variable.

    Parameters
    ----------
    *args
        Positional arguments forwarded to :class:`~gEconpy.classes.containers.SymbolDictionary`.
    **kwargs
        Keyword arguments forwarded to :class:`~gEconpy.classes.containers.SymbolDictionary`.

    Attributes
    ----------
    success : bool
        Whether the steady state solver reported success. Set to False when the results are created, and updated
        by the solver.
    """

    def __init__(self, *args, **kwargs):
        self.success = False
        super().__init__(*args, **kwargs)
