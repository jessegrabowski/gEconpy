import re

import sympy as sp

from sympy.abc import _clash1, _clash2

LOCAL_DICT = {}
for letter in _clash1.keys():
    LOCAL_DICT[letter] = sp.Symbol(letter)
for letter in _clash2.keys():
    LOCAL_DICT[letter] = sp.Symbol(letter)

OPERATORS = re.escape("+-*/^=();:")

BLOCK_START_TOKEN = "{"
BLOCK_END_TOKEN = "};"
LAG_TOKEN = "[-1]"
LEAD_TOKEN = "[1]"
SS_TOKEN = "[ss]"
EXPECTATION_TOKEN = "E[]"
CALIBRATING_EQ_TOKEN = "->"


SPECIAL_BLOCK_NAMES = ["OPTIONS", "TRYREDUCE", "ASSUMPTIONS"]
STEADY_STATE_NAMES = ["STEADY_STATE", "SS", "STEADYSTATE", "STEADY"]
BLOCK_COMPONENTS = [
    "DEFINITIONS",
    "CONTROLS",
    "OBJECTIVE",
    "CONSTRAINTS",
    "IDENTITIES",
    "SHOCKS",
    "CALIBRATION",
]

TIME_INDEX_DICT = {"ss": "ss", "t": 0, "tL1": -1, "t1": 1}

SYMPY_ASSUMPTIONS = [
    "finite",
    "infinite",
    "even",
    "odd",
    "prime",
    "composite",
    "positive",
    "negative",
    "zero",
    "nonzero",
    "nonpositive",
    "nonnegative",
    "integer",
    "rational",
    "irrational",
    "real",
    "extended real",
    "hermitian",
    "complex",
    "imaginary",
    "antihermitian",
    "algebraic",
    "transcendental",
    "unit",
]

DEFAULT_ASSUMPTIONS = {"real": True}
