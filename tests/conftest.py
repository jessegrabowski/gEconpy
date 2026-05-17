from pathlib import Path

import sympy as sp

from gEconpy.classes.time_aware_symbol import DEFAULT_ASSUMPTIONS, TimeAwareSymbol

PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
TESTS_ROOT = PROJECT_ROOT / "tests"
RESOURCES = TESTS_ROOT / "_resources"
TEST_GCNS = RESOURCES / "test_gcns"
ERROR_GCNS = RESOURCES / "error_gcns"


def parsed_symbol(name: str, **extra) -> sp.Symbol:
    """Construct a Symbol matching the parser's defaults (``real=True, finite=True``).

    Use in test fixtures that compare against ``ast_to_sympy`` output. ``sp.Symbol("x") != sp.Symbol("x", real=True)``
    in sympy's view, so hand-constructed expected Symbols must carry the same assumptions the parser injects.
    """
    return sp.Symbol(name, **{**DEFAULT_ASSUMPTIONS, **extra})


def parsed_var(name: str, t, **extra) -> TimeAwareSymbol:
    """Construct a TimeAwareSymbol matching the parser's defaults (``real=True, finite=True``)."""
    return TimeAwareSymbol(name, t, **{**DEFAULT_ASSUMPTIONS, **extra})


def parsed_symbols(names, **extra):
    """Plural form of :func:`parsed_symbol`, mirroring :func:`sympy.symbols`. ``names`` is a string or list."""
    return sp.symbols(names, **{**DEFAULT_ASSUMPTIONS, **extra})
