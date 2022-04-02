from gEcon.classes.time_aware_symbol import TimeAwareSymbol
from gEcon.shared.utilities import string_keys_to_sympy

from sympy.abc import greeks
import sympy as sp
import re

OPERATORS = list('+-/*^()=')


def get_name(x):
    if isinstance(x, str):
        return x

    elif isinstance(x, TimeAwareSymbol):
        return x.safe_name

    elif isinstance(x, sp.Symbol):
        return x.name


def build_hash_table(items_to_hash):
    var_to_hash = {}
    hash_to_var = {}
    name_list = [get_name(x) for x in items_to_hash]
    for thing in sorted(name_list, key=len, reverse=True):
        # ensure the hash value is positive so the minus sign isn't confused as part of the equation
        hashkey = str(hash(thing) ** 2)
        var_to_hash[thing] = hashkey
        hash_to_var[hashkey] = thing

    return var_to_hash, hash_to_var


def substitute_equation_from_dict(eq_str, hash_dict):
    for key, value in hash_dict.items():
        eq_str = eq_str.replace(key, value)
    return eq_str


def make_var_to_matlab_sub_dict(var_list, clash_prefix='a'):
    sub_dict = {}

    for var in var_list:
        if isinstance(var, str):
            var_name = var if var.lower() not in greeks else clash_prefix + var
        elif isinstance(var, TimeAwareSymbol):
            var_name = var.base_name if var.base_name.lower() not in greeks else clash_prefix + var.base_name
            time_index = var.safe_name.split('_')[-1]
            var_name += f'_{time_index}'
        elif isinstance(var, sp.Symbol):
            var_name = var.name if var.name.lower() not in greeks else clash_prefix + var.name
        else:
            raise ValueError('var_list should contain only strings, symbols, or TimeAwareSymbols')

        sub_dict[var] = var_name

    return sub_dict


def make_all_var_time_combos(var_list):
    result = []
    for x in var_list:
        result.extend([x.set_t(-1), x.set_t(0), x.set_t(1), x.set_t('ss')])

    return result


def convert_var_timings_to_matlab(var_list):
    matlab_var_list = [var.replace('_t+1', '(1)').replace('_t-1', '(-1)').replace('_t', '') \
                       for var in var_list]

    return matlab_var_list


def write_lines_from_list(l, file, line_start='', line_max=50):
    line = line_start
    for item in sorted(l):
        line += f' {item},'
        if len(line) > line_max:
            line = line[:-1]
            line = line + ';\n'
            file += line
            line = line_start

    if line != line_start:
        line = line[:-1]
        file += line + ';\n'

    return file


UNDER_T_PATTERN = '_t(?=[^\w]|$)'


def make_mod_file(model):
    var_list = model.variables
    param_dict = model.param_dict
    shocks = model.shocks
    ss_value_dict = model.steady_state_dict

    var_to_matlab = make_var_to_matlab_sub_dict(make_all_var_time_combos(var_list), clash_prefix='var_')
    par_to_matlab = make_var_to_matlab_sub_dict(param_dict.keys(), clash_prefix='param_')
    shock_to_matlab = make_var_to_matlab_sub_dict(shocks, clash_prefix='exog_')

    items_to_hash = list(var_to_matlab.keys()) + list(par_to_matlab.keys()) + list(shock_to_matlab.keys())

    file = ''
    file = write_lines_from_list([re.sub(UNDER_T_PATTERN, '', var_to_matlab[x]) for x in model.variables],
                                 file, line_start='var')
    file = write_lines_from_list([re.sub(UNDER_T_PATTERN, '', x) for x in shock_to_matlab.values()],
                                 file, line_start='varexo')
    file += '\n'
    file = write_lines_from_list(par_to_matlab.values(), file, line_start='parameters')
    file += '\n'

    for model_param in sorted(param_dict.keys()):
        matlab_param = par_to_matlab[model_param]
        value = param_dict[model_param]
        file += f'{matlab_param} = {value};\n'

    file += '\n'
    file += 'model;\n'
    for var, val in ss_value_dict.items():
        if var in var_to_matlab.keys():
            matlab_var = var_to_matlab[var]
            file += f'#{matlab_var}_ss = {val:0.4f};\n'

    for eq in model.system_equations:
        matlab_subdict = {}

        for atom in eq.atoms():
            if not isinstance(atom, TimeAwareSymbol) and isinstance(atom, sp.core.Symbol):
                if atom in par_to_matlab.keys():
                    matlab_subdict[atom] = sp.Symbol(par_to_matlab[atom])
            elif isinstance(atom, TimeAwareSymbol):
                if atom in var_to_matlab.keys():
                    matlab_subdict[atom] = var_to_matlab[atom]
                elif atom in shock_to_matlab.keys():
                    matlab_subdict[atom] = shock_to_matlab[atom]

        eq_str = eq.subs(matlab_subdict)
        eq_str = str(eq_str)
        prepare_eq = eq_str.replace('**', '^')
        var_to_hash, hash_to_var = build_hash_table(items_to_hash)

        hash_eq = substitute_equation_from_dict(prepare_eq, var_to_hash)

        for operator in OPERATORS:
            hash_eq = hash_eq.replace(operator, ' ' + operator + ' ')
        hash_eq = re.sub(' +', ' ', hash_eq)
        hash_eq = hash_eq.strip()
        final_eq = substitute_equation_from_dict(hash_eq, hash_to_var)

        matlab_eq = final_eq.replace('_tp1', '(1)').replace('_tm1', '(-1)')
        split_eq = matlab_eq.split(' ')

        new_eq = []
        for atom in split_eq:
            if atom in par_to_matlab.keys():
                atom = par_to_matlab[atom]
            elif atom in var_to_matlab.keys():
                atom = var_to_matlab[atom]
            elif atom in shock_to_matlab.keys():
                atom = shock_to_matlab[atom]

            new_eq.append(atom)

        matlab_eq = ''
        for i, atom in enumerate(new_eq):
            if i == 0:
                matlab_eq += atom
            elif i == 1 and new_eq[0] == '-':
                matlab_eq += atom
            else:
                if atom in '()':
                    matlab_eq += atom
                elif new_eq[i - 1] in '(':
                    matlab_eq += atom
                else:
                    matlab_eq += ' ' + atom
        matlab_eq += ' = 0;'
        matlab_eq = re.sub(UNDER_T_PATTERN, '', matlab_eq)

        file += matlab_eq + "\n"

    file += 'end;\n\n'

    file += 'initval;\n'
    for var, val in string_keys_to_sympy(ss_value_dict).items():
        matlab_var = var_to_matlab[var].replace('_ss', '')
        file += f'{matlab_var} = {val:0.4f};\n'

    file += 'end;\n\n'
    file += 'steady;\n'
    file += 'check(qz_zero_threshold=1e-20);\n\n'

    file += 'shocks;\n'
    for shock in shocks:
        file += 'var ' + re.sub(UNDER_T_PATTERN, '', shock_to_matlab[shock]) + ';\n'
        file += 'stderr 0.01;\n'
    file += 'end;\n\n'
    file += 'stoch_simul(order=1, irf=100, qz_zero_threshold=1e-20);'

    return file
