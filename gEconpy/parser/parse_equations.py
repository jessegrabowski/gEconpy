import re
from collections import defaultdict
from typing import Dict, List, Optional

import sympy as sp

from gEconpy.classes.time_aware_symbol import TimeAwareSymbol
from gEconpy.parser.constants import CALIBRATING_EQ_TOKEN, LOCAL_DICT, TIME_INDEX_DICT


def rebuild_eqs_from_parser_output(parser_output: List[str]) -> List[List[str]]:
    """
    Takes the output of pyparsing applied to the text of a model block and returns a list of lists of tokens
    that represent equations.

    The heavy lifting of parsing the model blocks is done by the pyparsing, which returns a list of tokens found
    between the BLOCK_START_TOKEN and BLOCK_END_TOKEN. These lists need to be further decomposed into equations,
    which is done using the LINE_TERMINATOR_TOKEN, ';'. The result is a list of lists, with each list containing the
    tokens for a single model equation.

    The function consumes the LINE_TERMINATOR_TOKENs in the process, and the returned lists are assumed to contain only
    variables, parameters, and valid mathematical operators.

    Parameters
    ----------
    parser_output: list of str
        Output of pyparsing applied to the text of a model block.

    Returns
    -------
    eqs: list of lists of str
        A list of lists, with each list containing the tokens for a single model equation.

    Examples
    --------
    ..code-block:: text
        rebuild_eqs_from_parser_output([A[], +, B[], +, C[], ;, Y[], =, L[], ^, alpha, *, K[-1], ^, (, 1, -, alpha, ), ;]) [[A[], +, B[], +, C[]],
         [Y[], =, L[], ^, alpha, *, K[-1], ^, (, 1, -, alpha, )]
    """

    eqs = []
    eq = []
    for element in parser_output:
        # TODO: Why are commas appearing in the control list after parsing? This is a hack fix.
        element = element.replace(",", "").strip()
        if len(element) > 0:
            eq.append(element)
        if ";" in element:
            # ; signals end of line, needs to be removed now
            eq[-1] = eq[-1].replace(";", "").strip()
            if len(eq[-1]) == 0:
                del eq[-1]

            eqs.append(eq)
            eq = []
    return eqs


def token_classifier(token: str) -> str:
    """
    Classify tokens as variables, parameters, operators, or special.

    Parameters
    ----------
    token: str
        Token from a gEcon model equation

    Returns
    -------
    classification: str
        Classification of the model token. One of ['operator', 'variable', 'number', 'lagrange_definition', 'calibration_definition', 'parameter']

    Notes
    -----
    There are two special tokens:
        1. The colon ":", that signifies that the variable after it should be the name of the lagrange multiplier
         associated with the equation.
        2. The arrow "->", that defines an calibrating equation for a parameter.
    """

    if token in ["E[]", "log", "exp"] or token in "+-*/^=()[]":
        return "operator"
    if "[" in token and "]" in token:
        return "variable"
    if all(s in "0123456789." for s in token):
        return "number"
    if token == ":":
        return "lagrange_definition"
    if token == "->":
        return "calibration_definition"

    return "parameter"


def has_num_index(token: str) -> bool:
    """
    Check if a token has a numerical lag/lead index.

    Lag and lead indices are of the form X[1] or X[-1], this function checks if such an index exists. Split on the
    opening square bracket first to be robust against variables with numbers in their names, i.e. alpha_1[1].

    Parameters
    ----------
    token : str
        A plaintext model variable.

    Returns
    -------
    bool
        Whether the variable has a numerical lag/lead index.
    """

    numbers = list("0123456789")
    index_part = token.split("[")[-1]
    return any([n in index_part for n in numbers])


def extract_time_index(token: str) -> str:
    """
    Extract a time index from a model variable.

    Parameters
    ----------
    token : str
        A string representing a model variable.

    Returns
    -------
    time_index : str
        A time-index string.

    Examples
    --------
    >>> extract_time_index("alpha[1]")
    't1'
    >>> extract_time_index("alpha[-1]")
    'tL1'
    >>> extract_time_index("alpha[ss]")
    'ss'
    >>> extract_time_index("alpha")
    't'
    """

    if has_num_index(token) and "-" not in token:
        lead = re.findall(r"\d+", token)[0]
        time_index = "t" + lead
    elif has_num_index(token) and "-" in token:
        lag = re.findall(r"\d+", token)[0]
        time_index = "tL" + lag
    elif "[ss]" in token:
        time_index = "ss"
    else:
        time_index = "t"

    return time_index


def remove_timing_information(token: str) -> str:
    """
    Remove timing information from a model variable token.

    A variable's timing information is contained in the square brackets next to it. This needs to be removed and
    replaced with some plaintext before it can be passed to the SymPy parser.

    Parameters
    ----------
    token : str
        A string representing a model variable.

    Returns
    -------
    str
        The same token with the timing information removed.
    """
    token = re.sub(r"\[.*?\]", "", token)
    return token.strip()


def convert_to_python_operator(token: str) -> str:
    """
    Convert a GCN operation token to a python-compatible one.

    Parameters
    ----------
    token : str
        A string representing a mathematical operation.

    Returns
    ---------
    str
        A string representing the same operation in python syntax.

    Notes
    ----------
    The syntax of a gEcon GCN file is slightly different from what SymPy expects, this function resolves the
    differences. In particular:
        1. Exponents are marked with a caret "^" in the GCN file, and must be converted to python's **
        2. SymPy's parse_expr cannot handle equalities, but can handle a list of equations. Equalities are thus
           converted to two separate equations then set as equal later.
        3. Remove the expectation operator completely
        4. Replace square brackets with parenthesis

    TODO: Implement an expectation operation in Sympy that can inserted here.
    """

    if token == "^":
        return "**"
    if token == "=":
        return ","
    if token == "E[]":
        return ""
    if token == "[":
        return "("
    if token == "]":
        return ")"

    return token


def rename_time_indexes(eq: sp.Eq) -> sp.Eq:
    """
    Rename time indices of the form "tL1" or "t1" to the normal form "t", "t+1", or "t-1".

    Parameters
    ----------
    eq : sp.Eq
        A sympy equation representing a model equation.

    Returns
    -------
    sp.Eq
        The same equation, with time indices renamed.

    Notes
    -----
    The function assumes the index is always at the end of the variable name.
    """

    ret_eq = eq.copy()
    for atom in ret_eq.atoms():
        if isinstance(atom, sp.core.Symbol):
            if re.search(r"tL?\d+", atom.name) is not None:
                name_tokens = atom.name.split("_")
                index_token = name_tokens[-1]
                operator = "-" if "L" in index_token else "+"
                number = re.search(r"\d+", index_token)[0]
                new_index = "".join(["_{t", operator, number, "}"])

                var_name = "_".join(s for s in name_tokens[:-1])
                atom.name = var_name + new_index

    return ret_eq


def convert_symbols_to_time_symbols(eq: sp.Eq) -> sp.Eq:
    """
    Convert SymPy symbols to time aware symbols.

    Parameters
    ----------
    eq : SymPy.Eq
        Sympy representation of a model equation.

    Returns
    -------
    Sympy.Eq
        The same equation with Symbols replaced with TimeAwareSymbols

    Notes
    -----
    Despite having time indexes, SymPy symbols are not "time aware". This function replaces all sp.Symbols with
    TimeAwareSymbols, which are extended to include a time index.
    """

    sub_dict = {}
    var_list = [variable for variable in eq.atoms() if isinstance(variable, sp.Symbol)]

    for variable in var_list:
        var_name = variable.name
        if re.search(r"_\{?t[-+ ]?\d*\}?$", var_name) is not None:
            name_tokens = var_name.split("_")
            name_part = "_".join(s for s in name_tokens[:-1])
            time_part = name_tokens[-1]

            time_part = re.sub(r"[\{\}t]", "", time_part)
            if len(time_part) == 0:
                time_index = 0
            else:
                time_index = int(time_part)
            time_var = TimeAwareSymbol(name_part, time_index)
            sub_dict[variable] = time_var
        elif "_ss" in var_name:
            base_name = var_name.replace("_ss", "")
            time_var = TimeAwareSymbol(base_name, 0)
            time_var = time_var.to_ss()
            sub_dict[variable] = time_var

    return eq.subs(sub_dict)


def single_symbol_to_sympy(
    variable: str, assumptions: Optional[Dict] = None
) -> TimeAwareSymbol:
    """
    Convert a single gEcon style variable (e.g. X[], or X[-1]) to a Time-Aware Sympy symbol. If it seems to be a
    parameter (no []), it returns a standard Sympy symbol instead.

    Parameters
    ----------
    variable : str
        A gEcon variable or parameter.
    assumptions : Optional[Dict]
        Assumptions for the symbol.

    Returns
    -------
    TimeAwareSymbol
        The same variable.
    """
    if assumptions is None:
        assumptions = defaultdict(dict)

    if "[" not in variable and "]" not in variable:
        return sp.Symbol(variable, **assumptions[variable])

    variable_name, time_part = variable.split("[")
    time_part = time_part.replace("]", "")
    if time_part == "ss":
        return TimeAwareSymbol(variable_name, 0).to_ss()
    else:
        time_index = int(time_part) if time_part != "" else 0
        return TimeAwareSymbol(variable_name, time_index, **assumptions[variable_name])


def build_sympy_equations(
    eqs: List[List[str]], assumptions: Optional[Dict] = None
) -> List[sp.Eq]:

    """
    Convert processed list of equation tokens to sympy equations.

    Parameters
    ----------
    eqs : list of lists of str
        A list of list of equation tokens associated with a model
    assumptions : dict, optional
        A dictionary of assumptions for each variable or parameter. Keys are variable names, values are dictionaries
        of assumptions.

    Returns
    -------
    list of sp.Eq
        A list of SymPy equations

    Notes
    -----
    To convert SymPy Symbols to the TimeAwareSymbol class, variables are re-named with a time index.
    Time-aware symbols are extended to include a time index.

    TODO: Improve error handling before parse_expr is called

    Examples
    --------
    .. code-block:: py
        eqs = [['Y', '=', 'C', '+', 'I']]
        assumptions = {'Y': {'real': True}, 'C': {'real': True}, 'I': {'real': True}}
        build_sympy_equations(eqs, assumptions)
    """

    if assumptions is None:
        assumptions = defaultdict(defaultdict)

    eqs_processed = []
    for eq in eqs:
        eq_str = ""
        calibrating_parameter = None
        sub_dict = LOCAL_DICT.copy()

        if CALIBRATING_EQ_TOKEN in eq:
            arrow_idx = eq.index(CALIBRATING_EQ_TOKEN)
            calibrating_parameter = eq[arrow_idx + 1]
            eq = eq[:arrow_idx]

        for token in eq:
            token_type = token_classifier(token)
            token = token.strip()
            if token_type == "variable":
                time_index = extract_time_index(token)
                token_base = remove_timing_information(token)
                token = token_base + "_" + time_index

                symbol = TimeAwareSymbol(
                    token_base, TIME_INDEX_DICT[time_index], **assumptions[token_base]
                )
                sub_dict[token] = symbol

            elif token_type == "parameter":
                symbol = sp.Symbol(token, **assumptions[token])
                sub_dict[token] = symbol

            elif token_type == "operator":
                token = convert_to_python_operator(token)

            eq_str += token

        try:
            eq_sympy = sp.parse_expr(eq_str, evaluate=False, local_dict=sub_dict)
        except Exception as e:
            print(f"Error encountered while parsing {eq_str}")
            print(e)
            raise e

        eq_sympy = sp.Eq(*eq_sympy)
        if calibrating_parameter is not None:
            param = sp.Symbol(
                calibrating_parameter, **assumptions[calibrating_parameter]
            )
            eq_sympy = sp.Eq(param, eq_sympy.lhs - eq_sympy.rhs)

        # eq_sympy = rename_time_indexes(eq_sympy)
        # eq_sympy = convert_symbols_to_time_symbols(eq_sympy)

        eqs_processed.append(eq_sympy)

    return eqs_processed
