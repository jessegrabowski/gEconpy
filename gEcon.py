import sympy as sp
import re
import pyparsing
import numpy as np
from TimeAwareSympy import *


def remove_comments(raw):
    lines = raw.split('\n')
    lines = [line for line in lines if len(line) > 0]
    output = []

    for line in lines:
        if line[0] != '#':
            if '#' in line:
                hash_idx = line.find('#')
                output.append(line[:hash_idx])
            else:
                output.append(line)

    return '\n'.join(output)


def load_gcn(gcn_path):
    with open(gcn_path, 'r', encoding='utf-8') as file:
        gcn_raw = file.read()
    return gcn_raw


def preprocess_gcn(gcn_raw):
    gcn_nolines = gcn_raw.replace('\n', ' ')
    gcn_nolines = gcn_nolines.replace('\t', ' ')

    gcn_nolines = re.sub(' +', ' ', gcn_nolines)

    operators = '+-*/^=();'
    for operator in operators:
        gcn_nolines = gcn_nolines.replace(operator, f' {operator} ')
    gcn_nolines = re.sub(' +', ' ', gcn_nolines)
    gcn_nolines = re.sub('} ;', '};', gcn_nolines)
    gcn_nolines = re.sub(r'\[ - 1\]', r'[-1]', gcn_nolines)

    return gcn_nolines


def split_gcn_by_block(gcn, return_header=False):
    gcn_blocks = gcn.split('block')
    header_stuff = gcn_blocks[0]
    gcn_blocks = gcn_blocks[1:]

    if return_header:
        return gcn_blocks, header_stuff
    else:
        return gcn_blocks


def rebuild_eqs_from_parser_output(parser_output):
    eqs = []
    eq = []
    for element in parser_output:
        eq.append(element)
        if ';' in element:
            eqs.append(eq)
            eq = []
    return eqs


def build_element_type_dict(parser_output):
    element_type_dict = {}
    for element in parser_output:
        if '[' in element and ']' in element:
            element_type_dict[element] = 'variable'
        elif element in '+-*/^=()':
            element_type_dict[element] = 'operator'
        elif all(s in '0123456789.' for s in element):
            element_type_dict[element] = 'number'
        elif element == ';':
            element_type_dict[element] = 'line_terminator'
        elif element == ':':
            element_type_dict[element] = 'lagrange_definition'
        else:
            element_type_dict[element] = 'parameter'

    return element_type_dict


def build_local_dict():
    from sympy.abc import _clash1, _clash2

    local_dict = {}
    for letter in _clash1.keys():
        local_dict[letter] = sp.Symbol(letter)
    for letter in _clash2.keys():
        local_dict[letter] = sp.Symbol(letter)

    return local_dict


def rename_time_indexes(eq):
    ret_eq = eq.copy()
    for atom in ret_eq.atoms():
        if isinstance(atom, sp.core.Symbol):
            if re.search('t\d', atom.name) is not None:
                lead = re.search('\d', atom.name)
                if isinstance(lead, list):
                    lead = lead[-1]
                else:
                    lead = lead[0]

                var_name = '_'.join(s for s in atom.name.split('_')[:-1])
                atom.name = var_name + '_{t+' + lead + '}'
            elif re.search('tL\d', atom.name) is not None:
                lag = re.search('\d', atom.name)
                if isinstance(lag, list):
                    lag = lag[-1]
                else:
                    lag = lag[0]
                var_name = '_'.join(s for s in atom.name.split('_')[:-1])
                atom.name = var_name + '_{t-' + lag + '}'

    return ret_eq


def build_sympy_equations(eqs, element_type_dict, local_dict):
    eqs_processed = []
    for eq in eqs:
        eq_str = ''
        for element in eq:
            element_type = element_type_dict[element]
            element = element.strip()
            if element_type == 'variable':
                if re.search(r'\[\d\]', element) is not None:
                    lead = re.findall('[\d]', element)[-1]
                    lead = re.sub('[\[\]]', '', lead)
                    time_index = 't' + lead
                elif re.search(r'\[-\d\]', element) is not None:
                    lag = re.findall('[\d]', element)[-1]
                    lag = re.sub('[\[\]]', '', lag)
                    time_index = 'tL' + lag
                elif re.search(r'\[ss\]', element) is not None:
                    time_index = 'ss'
                else:
                    time_index = 't'

                element = re.sub(r'E?\[-?\d?s?s?\]\[?', '', element)
                element = re.sub(r'[\[\]]', '', element)
                element = element + '_' + time_index

            elif element_type == 'operator':
                if element == '^':
                    element = '**'
                if element == '=':
                    element = ','

            if element_type != 'line_terminator':
                eq_str += element

        eq_str = re.sub(r'E?\[-?\d?s?s?\]\[?', '', eq_str)
        eq_str = re.sub(r'[\[\]]', '', eq_str)

        try:
            eq_sympy = sp.parse_expr(eq_str, evaluate=False, local_dict=local_dict)
        except Exception as e:
            print(f'Error encountered while parsing {eq_str}')
            print(e)
            raise e

        eq_sympy = sp.Eq(*eq_sympy)
        eq_sympy = rename_time_indexes(eq_sympy)
        eq_sympy = convert_symbols_to_time_symbols(eq_sympy)

        eqs_processed.append(eq_sympy)

    return eqs_processed


def gEcon_to_steady_state(eqs, element_type_dict, local_dict):
    eqs_processed = []
    for eq in eqs:
        eq_str = ''
        for element in eq:
            element_type = element_type_dict[element]
            element = element.strip()
            if element_type == 'variable':
                element = re.sub(r'E?\[-?\d?s?s?\]\[?', '', element)
                element = re.sub(r'[\[\]]', '', element)
            elif element_type == 'operator':
                if element == '^':
                    element = '**'
                if element == '=':
                    element = ','

            if element_type != 'line_terminator':
                eq_str += element

        eq_str = re.sub(r'E?\[-?\d?s?s?\]\[?', '', eq_str)
        eq_str = re.sub(r'[\[\]]', '', eq_str)

        try:
            eq_sympy = sp.parse_expr(eq_str, evaluate=False, local_dict=local_dict)
        except Exception as e:
            print(f'Error encountered while parsing {eq_str}')
            print(e)
            raise e

        eq_sympy = sp.Eq(*eq_sympy)

        eqs_processed.append(eq_sympy)

    return eqs_processed


def is_float(s):
    return all([item in '0123456789.' for item in s])


def get_model_variables(model, param_dict):
    variables = set()
    for eq in model:
        candidates = eq.atoms()
        for candidate in candidates:
            if candidate not in param_dict.keys() and not isinstance(candidate, sp.core.numbers.Number):
                variables.add(candidate)

    return list(variables)


def str_to_symbol(string_var):
    var_name = string_var.split('[')[0]
    time_index = string_var.split('[')[-1].replace(']', '')
    time_val = re.search('\d+', time_index)
    if time_val is not None:
        if '-' in time_index:
            time_suffix = "{" + f't-{time_val}' + "}"
        else:
            time_suffix = "{" + f't+{time_val}' + "}"
    else:
        time_suffix = 't'
    final_name = var_name + '_' + time_suffix
    symbol = sp.Symbol(final_name)
    symbol = convert_symbols_to_time_symbols(symbol)

    return symbol


def get_var_lists_from_eq(eq):
    var_list = []
    var_name_list = []
    for x in eq.atoms():
        if isinstance(x, TimeAwareSymbol):
            var_name_list.append(x.base_name)
            var_list.append(x)

    return np.array(var_list), np.array(var_name_list)


def check_for_objective_recursion(objective):
    lhs_vars, lhs_var_names = get_var_lists_from_eq(objective.lhs)
    rhs_vars, rhs_var_names = get_var_lists_from_eq(objective.rhs)

    is_recursive = False
    discount_factor = 1

    lhs_name_match_mask = np.array([lhs in rhs_var_names for lhs in lhs_var_names])
    rhs_name_match_mask = np.array([rhs in lhs_var_names for rhs in rhs_var_names])

    if any(lhs_name_match_mask):
        lhs_match = lhs_vars[np.flatnonzero(lhs_name_match_mask)][0]
        rhs_match = rhs_vars[np.flatnonzero(rhs_name_match_mask)][0]

        is_recursive = np.abs(lhs_match.time_index - rhs_match.time_index) == 1

    if is_recursive:
        future_match = lhs_match if lhs_match.time_index > rhs_match.time_index else rhs_match
        discount_factor = objective.rhs.coeff(future_match)

    return discount_factor


def extract_model_equations_from_gcn(gcn_blocks):
    param_dict = {}
    model = []
    shocks = []
    model_type_dict = {}

    local_dict = build_local_dict()

    for block in gcn_blocks:
        sub_dict = {}
        named_lms = {}
        control_vars = []
        constraints = []

        has_controls = False
        has_objective = False
        has_constraints = False

        block_name_idx = re.search('[A-Z _]+', block).span()[1]
        block_name = block[:block_name_idx].strip()
        block_name_short = ''.join(word[0] for word in block_name.split('_'))
        block = block[block_name_idx:]

        parsed_block = pyparsing.nestedExpr('{', '};').parseString(block.strip()).asList()[0]
        idx = 0
        while idx < len(parsed_block):
            section = parsed_block[idx]
            next_section = parsed_block[idx + 1]
            element_type_dict = build_element_type_dict(next_section)
            model_type_dict.update(element_type_dict)
            eqs = rebuild_eqs_from_parser_output(next_section)

            if section == 'definitions':
                eqs = build_sympy_equations(eqs, element_type_dict, local_dict)
                for eq in eqs:
                    sub_dict[eq.lhs] = eq.rhs

            elif section == 'controls':
                has_controls = True
                control_list = eqs[0]
                control_list = [variable.strip() for variable in control_list if variable != ';']

                for control in control_list:
                    control_symbol = str_to_symbol(control)
                    control_vars.append(control_symbol)

            elif section == 'objective':
                has_objective = True
                lm = None
                objective = eqs[0]

                if ':' in objective:
                    lm = sp.Symbol(objective[-2].replace('[]', '').replace(',', ''))
                    objective = objective[:-3]

                objective = build_sympy_equations([objective], element_type_dict, local_dict)[0]
                objective = objective.subs(sub_dict)

                discount_factor = check_for_objective_recursion(objective)

                model.append(objective)

                if lm is not None:
                    named_lms[objective.lhs] = lm

            elif section == 'identities':
                eqs = build_sympy_equations(eqs, element_type_dict, local_dict)

                for eq in eqs:
                    model.append(eq.subs(sub_dict))

            elif section == 'constraints':
                has_constraints = True
                lms = {}

                for i, eq in enumerate(eqs):
                    if ':' in eq:
                        lm = TimeAwareSymbol(eq[-2].replace('[]', '').replace(',', ''), 0)
                        lms[i] = lm
                        eqs[i] = eq[:-3]

                eqs = build_sympy_equations(eqs, element_type_dict, local_dict)
                for key in lms.keys():
                    named_lms[eqs[key].lhs] = lms[key]

                for eq in eqs:
                    constraints.append(eq.subs(sub_dict))
                    model.append(eq.subs(sub_dict))

            elif section == 'calibration':
                next_section = parsed_block[idx + 1]
                param_values = rebuild_eqs_from_parser_output(next_section)
                for param in param_values:
                    if ('-' in param) and ('>' in param):
                        eq = build_sympy_equations([param[:-4]], element_type_dict, local_dict)[0]
                        if len(eq.lhs.atoms()) == 1 and len(eq.rhs.atoms()) == 1:
                            param_dict[eq.lhs] = eq.rhs
                        else:
                            model.append(eq)

                    else:
                        param_name = sp.Symbol(param[0])

                        val_idx = [is_float(element) for element in param]
                        val_idx = np.flatnonzero(val_idx)

                        param = np.array(param)
                        param_val = float(param[val_idx])
                        if param[val_idx - 1] == '-':
                            param_val = param_val * -1
                        param_dict[param_name] = param_val

            elif section == 'shocks':
                for shock in eqs[0]:
                    if shock != ';':
                        shock = TimeAwareSymbol(shock.replace('[]', '').replace(',', ''), 0)
                        shocks.append(shock)
                        param_dict[shock] = 0

            idx = idx + 2
        if has_controls and has_objective and has_constraints:
            lagrange = objective.rhs
            i = 1
            for constraint in constraints:
                if constraint.lhs in named_lms.keys():
                    lm = named_lms[constraint.lhs]
                else:
                    lm = TimeAwareSymbol(f'lambda__{block_name_short}{i}', 0)
                    i += 1

                lagrange = lagrange - lm * (constraint.lhs - constraint.rhs)

            if objective.lhs in named_lms.keys():
                model.append(named_lms[objective.lhs] - diff_through_time(lagrange, objective.lhs, discount_factor))

            for control in control_vars:
                foc = diff_through_time(lagrange, control, discount_factor)
                model.append(foc.powsimp())

    return model, param_dict, shocks


def parse_gcn_file(gcn_filepath):
    gcn_raw = load_gcn(gcn_filepath)
    gcn_raw = remove_comments(gcn_raw)
    gcn_processed = preprocess_gcn(gcn_raw)
    gcn_blocks, header = split_gcn_by_block(gcn_processed, return_header=True)

    try_reduce_vars = re.sub('[\{\};\[\]]', '', header.split('tryreduce')[1])
    try_reduce_vars = try_reduce_vars.strip().split(',')
    try_reduce_vars = [x.strip() for x in try_reduce_vars]

    model, param_dict, shocks = extract_model_equations_from_gcn(gcn_blocks)
    var_list = get_model_variables(model, param_dict)

    return model, param_dict, shocks, var_list, try_reduce_vars
