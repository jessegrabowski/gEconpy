import re
from typing import Dict, List, Tuple, Union

import sympy as sp
from sympy.abc import greeks

from gEconpy.classes.time_aware_symbol import TimeAwareSymbol
from gEconpy.shared.utilities import make_all_var_time_combos, string_keys_to_sympy

OPERATORS = list("+-/*^()=")


def get_name(x: Union[str, sp.Symbol]) -> str:
    """
    This function returns the name of a string, TimeAwareSymbol, or sp.Symbol object.

    Parameters
    ----------
    x : str, or sp.Symbol
        The object whose name is to be returned. If str, x is directly returned.

    Returns
    -------
    str
        The name of the object.
    """

    if isinstance(x, str):
        return x

    elif isinstance(x, TimeAwareSymbol):
        return x.safe_name

    elif isinstance(x, sp.Symbol):
        return x.name


def build_hash_table(
    items_to_hash: List[Union[str, sp.Symbol]]
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    This function builds a pair of hash tables, one mapping variable names to hash values
    and the other mapping hash values to variable names.

    To safely distinguish between numeric values, variables, parameters, and time-indices
    when converting sympy code to a Dynare model, all variables are first hashed to
    strictly positive int64 objects using the square of the built-in `hash` function.

    Parameters
    ----------
    items_to_hash : str or sp.Symbol
        A list of variables to be hashed. Can contain strings or sp.Symbol objects.

    Returns
    -------
    tuple of (dict, dict)
        A tuple containing two dictionaries: the first maps variable names to
         hash values, and the second maps hash values to variable names.
    """

    var_to_hash = {}
    hash_to_var = {}
    name_list = [get_name(x) for x in items_to_hash]
    for thing in sorted(name_list, key=len, reverse=True):
        # ensure the hash value is positive so the minus sign isn't confused as part of the equation
        hashkey = str(hash(thing) ** 2)
        var_to_hash[thing] = hashkey
        hash_to_var[hashkey] = thing

    return var_to_hash, hash_to_var


def substitute_equation_from_dict(eq_str: str, hash_dict: Dict[str, str]) -> str:
    """
    This function substitutes variables in an equation string with their corresponding values from a dictionary.

    Parameters
    ----------
    eq_str : str
        The equation string containing variables to be replaced.
    hash_dict : Dict[str, str]
        A dictionary mapping variables to their corresponding values.

    Returns
    -------
    str
        The equation string with the variables replaced by their values.
    """
    for key, value in hash_dict.items():
        eq_str = eq_str.replace(key, value)
    return eq_str


def make_var_to_matlab_sub_dict(
    var_list: List[Union[str, TimeAwareSymbol, sp.Symbol]], clash_prefix: str = "a"
) -> Dict[Union[str, TimeAwareSymbol, sp.Symbol], str]:
    """
    This function builds a dictionary that maps variables to their corresponding names that
    can be used in a Matlab script.

    Parameters
    ----------
    var_list : List[Union[str, TimeAwareSymbol, sp.Symbol]]
        A list of variables to be mapped. Can contain strings, TimeAwareSymbol objects,
         or sp.Symbol objects.
    clash_prefix : str, optional
        A prefix to add to the names of variables that might clash with Matlab keywords
        (e.g. greek letters). Default is 'a'.

    Returns
    -------
    Dict[Union[str, TimeAwareSymbol, sp.Symbol], str]
        A dictionary mapping the variables in `var_list` to their corresponding
        names that can be used in a Matlab script.

    Examples
    --------
    .. code-block:: py
        make_var_to_matlab_sub_dict([sp.Symbol('beta')])
        # {sp.Symbol('beta'): 'abeta'}
    """

    sub_dict = {}

    for var in var_list:
        if isinstance(var, str):
            var_name = var if var.lower() not in greeks else clash_prefix + var
        elif isinstance(var, TimeAwareSymbol):
            var_name = (
                var.base_name
                if var.base_name.lower() not in greeks
                else clash_prefix + var.base_name
            )
            time_index = var.safe_name.split("_")[-1]
            var_name += f"_{time_index}"
        elif isinstance(var, sp.Symbol):
            var_name = (
                var.name if var.name.lower() not in greeks else clash_prefix + var.name
            )
        else:
            raise ValueError(
                "var_list should contain only strings, symbols, or TimeAwareSymbols"
            )

        sub_dict[var] = var_name

    return sub_dict


def convert_var_timings_to_matlab(var_list: List[str]) -> List[str]:
    """
    This function converts the timing notation in a list of variable names to a
    form that can be used in a Dynare mod file.

    Parameters
    ----------
    var_list : list of str
        A list of variable names with "mathematical" timing notation (e.g. '_t+1', '_t-1', '_t').

    Returns
    -------
    list of str
        A list of variable names with the timing notation converted to a
        form that can be used in a Dynare mod file (e.g. '(1)', '(-1)', '').
    """
    matlab_var_list = [
        var.replace("_t+1", "(1)").replace("_t-1", "(-1)").replace("_t", "")
        for var in var_list
    ]

    return matlab_var_list


def write_lines_from_list(
    l: List[str], file: str, line_start: str = "", line_max: int = 50
) -> str:
    """
    This function writes a list of items to a string, inserting line
    breaks at a specified maximum line length.

    Parameters
    ----------
    l : list of strings
        A list of items to be written to the string.
    file : str
        A string to which the items will be appended.
    line_start : str, optional
        A string to be prepended to each line. Default is an empty string.
    line_max : int, optional
        The maximum line length. Default is 50.

    Returns
    -------
    str
        The modified `file` string with the items from `l` appended to it.
    """

    line = line_start
    for item in sorted(l):
        line += f" {item},"
        if len(line) > line_max:
            line = line[:-1]
            line = line + ";\n"
            file += line
            line = line_start

    if line != line_start:
        line = line[:-1]
        file += line + ";\n"

    return file


UNDER_T_PATTERN = r"_t(?=[^\w]|$)"


def make_mod_file(model) -> str:
    """
    This function generates a string representation of a Dynare model file for
    a dynamic stochastic general equilibrium (DSGE) model. For more information,
    see [1].

    Parameters
    ----------
    model : gEconModel
        A gEconModel object with solved steady state.

    Returns
    -------
    str
        A string representation of a Dynare model file.

    References
    ----------
    ..[1] Adjemian, Stéphane, et al. "Dynare: Reference manual, version 4." (2011).
    """

    var_list = model.variables
    param_dict = model.free_param_dict
    param_dict.update(model.calib_param_dict)

    shocks = model.shocks
    ss_value_dict = model.steady_state_dict

    var_to_matlab = make_var_to_matlab_sub_dict(
        make_all_var_time_combos(var_list), clash_prefix="var_"
    )
    par_to_matlab = make_var_to_matlab_sub_dict(
        param_dict.keys(), clash_prefix="param_"
    )
    shock_to_matlab = make_var_to_matlab_sub_dict(shocks, clash_prefix="exog_")

    items_to_hash = (
        list(var_to_matlab.keys())
        + list(par_to_matlab.keys())
        + list(shock_to_matlab.keys())
    )

    file = ""
    file = write_lines_from_list(
        [re.sub(UNDER_T_PATTERN, "", var_to_matlab[x]) for x in model.variables],
        file,
        line_start="var",
    )
    file = write_lines_from_list(
        [re.sub(UNDER_T_PATTERN, "", x) for x in shock_to_matlab.values()],
        file,
        line_start="varexo",
    )
    file += "\n"
    file = write_lines_from_list(par_to_matlab.values(), file, line_start="parameters")
    file += "\n"

    for model_param in sorted(param_dict.keys()):
        matlab_param = par_to_matlab[model_param]
        value = param_dict[model_param]
        file += f"{matlab_param} = {value};\n"

    file += "\n"
    file += "model;\n"
    for var, val in ss_value_dict.items():
        if var in var_to_matlab.keys():
            matlab_var = var_to_matlab[var]
            file += f"#{matlab_var}_ss = {val:0.4f};\n"

    for eq in model.system_equations:
        matlab_subdict = {}

        for atom in eq.atoms():
            if not isinstance(atom, TimeAwareSymbol) and isinstance(
                atom, sp.core.Symbol
            ):
                if atom in par_to_matlab.keys():
                    matlab_subdict[atom] = sp.Symbol(par_to_matlab[atom])
            elif isinstance(atom, TimeAwareSymbol):
                if atom in var_to_matlab.keys():
                    matlab_subdict[atom] = var_to_matlab[atom]
                elif atom in shock_to_matlab.keys():
                    matlab_subdict[atom] = shock_to_matlab[atom]

        eq_str = eq.subs(matlab_subdict)
        eq_str = str(eq_str)
        prepare_eq = eq_str.replace("**", "^")
        var_to_hash, hash_to_var = build_hash_table(items_to_hash)

        hash_eq = substitute_equation_from_dict(prepare_eq, var_to_hash)

        for operator in OPERATORS:
            hash_eq = hash_eq.replace(operator, " " + operator + " ")
        hash_eq = re.sub(" +", " ", hash_eq)
        hash_eq = hash_eq.strip()
        final_eq = substitute_equation_from_dict(hash_eq, hash_to_var)

        matlab_eq = final_eq.replace("_tp1", "(1)").replace("_tm1", "(-1)")
        split_eq = matlab_eq.split(" ")

        new_eq = []
        for atom in split_eq:
            if atom in par_to_matlab.keys():
                atom = par_to_matlab[atom]
            elif atom in var_to_matlab.keys():
                atom = var_to_matlab[atom]
            elif atom in shock_to_matlab.keys():
                atom = shock_to_matlab[atom]

            new_eq.append(atom)

        matlab_eq = ""
        for i, atom in enumerate(new_eq):
            if i == 0:
                matlab_eq += atom
            elif i == 1 and new_eq[0] == "-":
                matlab_eq += atom
            else:
                if atom in "()":
                    matlab_eq += atom
                elif new_eq[i - 1] in "(":
                    matlab_eq += atom
                else:
                    matlab_eq += " " + atom
        matlab_eq += " = 0;"
        matlab_eq = re.sub(UNDER_T_PATTERN, "", matlab_eq)

        file += matlab_eq + "\n"

    file += "end;\n\n"

    file += "initval;\n"
    for var, val in string_keys_to_sympy(ss_value_dict).items():
        matlab_var = var_to_matlab[var].replace("_ss", "")
        file += f"{matlab_var} = {val:0.4f};\n"

    file += "end;\n\n"
    file += "steady;\n"
    file += "check(qz_zero_threshold=1e-20);\n\n"

    file += "shocks;\n"
    for shock in shocks:
        file += "var " + re.sub(UNDER_T_PATTERN, "", shock_to_matlab[shock]) + ";\n"
        file += "stderr 0.01;\n"
    file += "end;\n\n"
    file += "stoch_simul(order=1, irf=100, qz_zero_threshold=1e-20);"

    return file
