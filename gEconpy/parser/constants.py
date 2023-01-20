import re
from enum import Enum

import sympy as sp
from sympy.abc import _clash1, _clash2

from gEconpy.shared.utilities import IterEnum

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


class SPECIAL_BLOCK_NAMES(Enum, metaclass=IterEnum):
    OPTIONS = "OPTIONS"
    TRYREDUCE = "TRYREDUCE"
    ASSUMPTIONS = "ASSUMPTIONS"


class STEADY_STATE_NAMES(Enum, metaclass=IterEnum):
    STEADY_STATE = "STEADY_STATE"
    SS = "SS"
    STEADYSTATE = "STEADYSTATE"
    STEADY = "STEADY"


class BLOCK_COMPONENTS(Enum, metaclass=IterEnum):
    DEFINITIONS = "DEFINITIONS"
    CONTROLS = "CONTROLS"
    OBJECTIVE = "OBJECTIVE"
    CONSTRAINTS = "CONSTRAINTS"
    IDENTITIES = "IDENTITIES"
    SHOCKS = "SHOCKS"
    CALIBRATION = "CALIBRATION"


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
]
