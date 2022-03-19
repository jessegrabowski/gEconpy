import sympy as sp
from gEcon.classes.TimeAwareSymbol import TimeAwareSymbol


def unpack_keys_and_values(d):
    keys = list(d.keys())
    values = list(d.values())

    return keys, values


def set_equality_equals_zero(eq):
    if not isinstance(eq, sp.Eq):
        return eq

    return eq.rhs - eq.lhs


def step_equation_forward(eq):
    to_step = []

    for variable in set(eq.atoms()):
        if hasattr(variable, 'step_forward'):
            if variable.time_index != 'ss':
                to_step.append(variable)

    for variable in sorted(to_step, key=lambda x: x.time_index, reverse=True):
        eq = eq.subs({variable: variable.step_forward()})

    return eq


def step_equation_backward(eq):
    to_step = []

    for variable in set(eq.atoms()):
        if hasattr(variable, 'step_forward'):
            to_step.append(variable)

    for variable in sorted(to_step, key=lambda x: x.time_index, reverse=False):
        eq = eq.subs({variable: variable.step_backward()})

    return eq


def diff_through_time(eq, dx, discount_factor=1):
    total_dydx = 0
    next_dydx = 1

    while next_dydx != 0:
        next_dydx = eq.diff(dx)
        eq = step_equation_forward(eq) * discount_factor
        total_dydx += next_dydx

    return total_dydx


def is_variable(x):
    return isinstance(x, TimeAwareSymbol)


def eq_to_ss(eq):
    var_list = [x for x in eq.atoms() if isinstance(x, TimeAwareSymbol)]
    sub_dict = dict(zip(var_list, [x.to_ss() for x in var_list]))
    return eq.subs(sub_dict)
