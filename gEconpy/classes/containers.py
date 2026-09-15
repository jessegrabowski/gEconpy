from collections.abc import Iterable
from typing import Any

import sympy as sp

from sympy.polys.domains.mpelements import ComplexElement

from gEconpy.classes.time_aware_symbol import TimeAwareSymbol

SAFE_STRING_TO_INDEX_DICT = {"ss": "ss", "tp1": 1, "tm1": -1, "t": 0}
TIME_SUFFIXES = tuple(f"_{suffix}" for suffix in SAFE_STRING_TO_INDEX_DICT)


def safe_string_to_sympy(s: str | sp.Symbol, assumptions: dict[str, dict[str, bool]] | None = None) -> sp.Symbol:
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

    assumptions = assumptions or {}

    *name_parts, suffix = s.split("_")
    if suffix not in SAFE_STRING_TO_INDEX_DICT:
        return sp.Symbol(s, **assumptions.get(s, {}))

    name = "_".join(name_parts)
    return TimeAwareSymbol(name, SAFE_STRING_TO_INDEX_DICT[suffix], **assumptions.get(name, {}))


def symbol_to_string(symbol: str | sp.Symbol) -> str:
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


def string_keys_to_sympy(
    d: dict[str | sp.Symbol, Any],
    assumptions: dict[str, dict[str, bool]] | None = None,
    is_variable: dict[str, bool] | None = None,
) -> dict[sp.Symbol, Any]:
    """
    Convert the string keys of a dictionary to symbols, leaving the values alone.

    Parameters
    ----------
    d : dict
        Dictionary with string or symbol keys.
    assumptions : dict mapping str to dict, optional
        Sympy assumptions to attach, keyed by symbol name. Defaults to no assumptions.
    is_variable : dict mapping str to bool, optional
        Marks which names are model variables, and so eligible to become time aware symbols. Names absent from this
        mapping are treated as variables. Defaults to treating every name as a variable.

    Returns
    -------
    result : dict
        Dictionary with symbol keys.
    """
    assumptions = assumptions if assumptions is not None else {}
    is_variable = is_variable if is_variable is not None else {}

    result = {}
    for key, value in d.items():
        if isinstance(key, sp.Symbol):
            result[key] = value
        elif is_variable.get(key, True) and key.endswith(TIME_SUFFIXES):
            result[safe_string_to_sympy(key, assumptions)] = value
        else:
            result[sp.Symbol(key, **assumptions.get(key, {}))] = value

    return result


def sympy_keys_to_strings(d: dict[str | sp.Symbol, Any]) -> dict[str, Any]:
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
    return {symbol_to_string(key): value for key, value in d.items()}


def sympy_number_values_to_floats(d: dict[sp.Symbol, Any]) -> dict[sp.Symbol, Any]:
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


def float_values_to_sympy_float(d: dict[sp.Symbol, Any]) -> dict[sp.Symbol, Any]:
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


def sort_dictionary(d: dict) -> dict:
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
    return {key: d[key] for key in sorted(d.keys())}


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
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._assumptions: dict[str, dict[str, bool]] = {}
        self._is_variable: dict[str, bool] = {}

        keys = list(self.keys())
        if any(not isinstance(key, sp.Symbol | str) for key in keys):
            raise KeyError("All keys should be either string or Sympy symbols")

        n_string_keys = sum(isinstance(key, str) for key in keys)
        if 0 < n_string_keys < len(keys):
            raise KeyError("Cannot mix sympy and string keys")

        self.is_sympy: bool = len(keys) > 0 and n_string_keys == 0
        self._record_keys(keys)

    def __reduce__(self):
        # Default dict pickling calls __setitem__ before __init__, which fails because _assumptions does not exist
        # yet. Serialize the raw dict contents and metadata, then reconstruct via __new__ and dict.update, bypassing
        # __setitem__.
        base_attrs = {"is_sympy", "_assumptions", "_is_variable"}
        extra_attrs = {name: value for name, value in self.__dict__.items() if name not in base_attrs}
        return (
            _unpickle_symbol_dictionary,
            (type(self), dict(self), self.is_sympy, self._assumptions, self._is_variable, extra_attrs),
        )

    def __or__(self, other: dict):
        if not isinstance(other, dict):
            raise TypeError("__or__ not defined on non-dictionary objects")
        if not isinstance(other, SymbolDictionary):
            other = SymbolDictionary(other)

        merged = self.copy()

        if len(merged) == 0:
            other_copy = other.copy()
            other_copy._assumptions.update(self._assumptions)
            other_copy._is_variable.update(self._is_variable)
            return other_copy

        if len(other) == 0:
            merged._assumptions.update(other._assumptions)
            merged._is_variable.update(self._is_variable)
            return merged

        if other.is_sympy != self.is_sympy:
            raise ValueError("Cannot merge string-mode SymbolDictionary with sympy-mode SymbolDictionary")

        merged.update(other)
        merged._assumptions.update(other._assumptions)
        merged._is_variable.update(other._is_variable)

        return merged

    def __setitem__(self, key, value):
        if len(self) == 0:
            self.is_sympy = isinstance(key, sp.Symbol)
        elif self.is_sympy and not isinstance(key, sp.Symbol):
            raise KeyError("Cannot add string key to dictionary in sympy mode")
        elif not self.is_sympy and isinstance(key, sp.Symbol):
            raise KeyError("Cannot add sympy key to dictionary in string mode")

        super().__setitem__(key, value)
        self._record_keys([key])

    def copy(self) -> "SymbolDictionary":
        """
        Return a shallow copy of the same class that shares the sympy flag, the assumptions and the variable flags.

        Subclass attributes such as ``SteadyStateResults.success`` are carried over.
        """
        new_d = type(self)(super().copy())
        new_d.is_sympy = self.is_sympy
        new_d._assumptions = self._assumptions
        new_d._is_variable = self._is_variable

        base_attrs = {"is_sympy", "_assumptions", "_is_variable"}
        for name, value in self.__dict__.items():
            if name not in base_attrs:
                setattr(new_d, name, value)

        return new_d

    def update(self, other=None, **kwargs):
        """
        Update the dictionary with the key-value pairs of another mapping.

        When the other mapping is a SymbolDictionary, its stored assumptions and variable flags are merged as well.

        Parameters
        ----------
        other : dict, optional
            Mapping to take key-value pairs from. Defaults to an empty mapping.
        **kwargs
            Additional key-value pairs to add. Each goes through ``__setitem__``, so a keyword (string) key is
            rejected when the dictionary is in sympy mode.
        """
        if other is None:
            other = {}

        super().update(other)
        if isinstance(other, SymbolDictionary):
            self._assumptions.update(other._assumptions)
            self._is_variable.update(other._is_variable)
            if len(self) == len(other) or not self.is_sympy:
                self.is_sympy = other.is_sympy

        for key, value in kwargs.items():
            self[key] = value

    def to_sympy(self, inplace: bool = False, new_assumptions=None, new_is_variable=None):
        """
        Convert string keys to sympy symbols.

        Parameters
        ----------
        inplace : bool, optional
            Whether to replace the contents of this dictionary. Defaults to ``False``, returning a new dictionary.
        new_assumptions : dict mapping str to dict, optional
            Assumptions to attach to the new symbols, keyed by name. Override the stored assumptions on conflict.
            Defaults to ``None``.
        new_is_variable : dict mapping str to bool, optional
            Variable flags to apply, keyed by name. Override the stored flags on conflict. Defaults to ``None``.

        Returns
        -------
        result : SymbolDictionary or None
            The converted dictionary, or ``None`` when ``inplace`` is ``True``.
        """
        assumptions = {**self._assumptions, **(new_assumptions or {})}
        is_variable = {**self._is_variable, **(new_is_variable or {})}

        d = SymbolDictionary(string_keys_to_sympy(self, assumptions, is_variable))
        return self._replace_or_return(d, inplace=inplace)

    def to_string(self, inplace: bool = False):
        """
        Convert sympy keys to strings, keeping the assumptions for a later :meth:`to_sympy`.

        Parameters
        ----------
        inplace : bool, optional
            Whether to replace the contents of this dictionary. Defaults to ``False``, returning a new dictionary.

        Returns
        -------
        result : SymbolDictionary or None
            The converted dictionary, or ``None`` when ``inplace`` is ``True``.
        """
        d = self._with_metadata(SymbolDictionary(sympy_keys_to_strings(self)))
        return self._replace_or_return(d, inplace=inplace)

    def step_forward(self, inplace: bool = False):
        """
        Increment the time index of every time aware key by one.

        Parameters
        ----------
        inplace : bool, optional
            Whether to replace the contents of this dictionary. Defaults to ``False``, returning a new dictionary.

        Returns
        -------
        result : SymbolDictionary or None
            The stepped dictionary, or ``None`` when ``inplace`` is ``True``.
        """
        return self._replace_or_return(self._map_time_aware_keys("step_forward"), inplace=inplace)

    def step_backward(self, inplace: bool = False):
        """
        Decrement the time index of every time aware key by one.

        Parameters
        ----------
        inplace : bool, optional
            Whether to replace the contents of this dictionary. Defaults to ``False``, returning a new dictionary.

        Returns
        -------
        result : SymbolDictionary or None
            The stepped dictionary, or ``None`` when ``inplace`` is ``True``.
        """
        return self._replace_or_return(self._map_time_aware_keys("step_backward"), inplace=inplace)

    def to_ss(self, inplace: bool = False):
        """
        Move every time aware key to the steady state.

        Parameters
        ----------
        inplace : bool, optional
            Whether to replace the contents of this dictionary. Defaults to ``False``, returning a new dictionary.

        Returns
        -------
        result : SymbolDictionary or None
            The steady state dictionary, or ``None`` when ``inplace`` is ``True``.
        """
        return self._replace_or_return(self._map_time_aware_keys("to_ss"), inplace=inplace)

    def sort_keys(self, inplace: bool = False):
        """
        Order the items by the string form of their keys.

        Parameters
        ----------
        inplace : bool, optional
            Whether to replace the contents of this dictionary. Defaults to ``False``, returning a new dictionary.

        Returns
        -------
        result : SymbolDictionary or None
            The sorted dictionary, or ``None`` when ``inplace`` is ``True``.
        """
        d = self._with_metadata(SymbolDictionary(sort_dictionary(self.to_string())))

        if self.is_sympy:
            d = d.to_sympy()

        return self._replace_or_return(d, inplace=inplace)

    def values_to_float(self, inplace: bool = False):
        """
        Replace sympy numeric values with Python floats or complex numbers.

        Parameters
        ----------
        inplace : bool, optional
            Whether to replace the contents of this dictionary. Defaults to ``False``, returning a new dictionary.

        Returns
        -------
        result : SymbolDictionary or None
            The converted dictionary, or ``None`` when ``inplace`` is ``True``.
        """
        d = self._with_metadata(sympy_number_values_to_floats(self.copy()))
        return self._replace_or_return(d, inplace=inplace)

    def float_to_values(self, inplace: bool = False):
        """
        Replace Python numeric values with sympy numbers.

        Parameters
        ----------
        inplace : bool, optional
            Whether to replace the contents of this dictionary. Defaults to ``False``, returning a new dictionary.

        Returns
        -------
        result : SymbolDictionary or None
            The converted dictionary, or ``None`` when ``inplace`` is ``True``.
        """
        d = self._with_metadata(float_values_to_sympy_float(self.copy()))
        return self._replace_or_return(d, inplace=inplace)

    def _record_keys(self, keys: Iterable) -> None:
        if not self.is_sympy:
            return
        for key in keys:
            if isinstance(key, TimeAwareSymbol):
                self._assumptions[key.base_name] = key.assumptions0
                self._is_variable[key.base_name] = True
            else:
                self._assumptions[key.name] = key.assumptions0
                self._is_variable[key.name] = False

    def _with_metadata(self, d: "SymbolDictionary") -> "SymbolDictionary":
        d._assumptions = self._assumptions.copy()
        d._is_variable = self._is_variable.copy()
        return d

    def _replace_or_return(self, d: "SymbolDictionary", inplace: bool):
        if not inplace:
            return d
        self.clear()
        self._assumptions.clear()
        self._is_variable.clear()
        self.update(d)
        return None

    def _map_time_aware_keys(self, method_name: str) -> "SymbolDictionary":
        symbolic = self if self.is_sympy else self.to_sympy()

        def shift(key):
            return getattr(key, method_name)() if isinstance(key, TimeAwareSymbol) else key

        d = SymbolDictionary({shift(key): value for key, value in symbolic.items()})
        return d if self.is_sympy else d.to_string()


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


def _unpickle_symbol_dictionary(cls, items, is_sympy, assumptions, is_variable, extra_attrs):
    d = cls.__new__(cls)
    d.is_sympy = is_sympy
    d._assumptions = assumptions
    d._is_variable = is_variable
    for name, value in extra_attrs.items():
        setattr(d, name, value)
    dict.update(d, items)
    return d
