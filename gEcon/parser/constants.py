import sympy as sp
from sympy.abc import _clash1, _clash2
import re
from enum import Enum

LOCAL_DICT = {}
for letter in _clash1.keys():
    LOCAL_DICT[letter] = sp.Symbol(letter)
for letter in _clash2.keys():
    LOCAL_DICT[letter] = sp.Symbol(letter)

OPERATORS = re.escape('+-*/^=();:')

BLOCK_START_TOKEN = '{'
BLOCK_END_TOKEN = '};'
LAG_TOKEN = '[-1]'
LEAD_TOKEN = '[1]'
SS_TOKEN = '[ss]'
EXPECTATION_TOKEN = 'E[]'
CALIBRATING_EQ_TOKEN = '->'


class SPECIAL_BLOCK_NAMES(Enum):
    OPTIONS = 'OPTIONS'
    TRYREDUCE = 'TRYREDUCE'


class BLOCK_COMPONENTS(Enum):
    DEFINITIONS = 'DEFINITIONS'
    CONTROLS = 'CONTROLS'
    OBJECTIVE = 'OBJECTIVE'
    CONSTRAINTS = 'CONSTRAINTS'
    IDENTITIES = 'IDENTITIES'
    SHOCKS = 'SHOCKS'
    CALIBRATION = 'CALIBRATION'
