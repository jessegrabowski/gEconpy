import sympy as sp
from sympy.core.cache import cacheit
import re


class TimeAwareSymbol(sp.Symbol):
    def __new__(cls, name, time_index, **assumptions):
        cls._sanitize(assumptions, cls)

        return TimeAwareSymbol.__xnew__(cls, name, time_index, **assumptions)

    def __getnewargs__(self):
        return self.name, self.time_index

    @staticmethod
    @cacheit
    def __xnew__(cls, name, time_index, **assumptions):
        obj = sp.Symbol.__xnew__(cls, name, **assumptions)
        obj.time_index = time_index
        obj.base_name = name
        obj.name = obj._create_name_from_time_index()
        return obj

    def _determine_operator(self):
        if self.time_index == 'ss':
            return ''
        if self.time_index > 0:
            operator = '+'
        elif self.time_index < 0:
            operator = '-'
        else:
            operator = ''
        return operator

    def _create_name_from_time_index(self):
        operator = self._determine_operator()
        name = self.base_name
        idx = self.time_index
        idx = idx if isinstance(idx, str) else str(abs(idx))

        if idx == 'ss':
            time_name = r'%s_%s' % (name, idx)
        elif idx == '0':
            time_name = r'%s_t' % name
        else:
            time_name = r'%s_t%s%s' % (name, operator, idx)

        return time_name

    def _hashable_content(self):
        return super()._hashable_content() + (self.time_index,)

    def step_forward(self):
        obj = TimeAwareSymbol(self.base_name, self.time_index + 1)
        return obj

    def step_backward(self):
        obj = TimeAwareSymbol(self.base_name, self.time_index - 1)
        return obj

    def to_ss(self):
        obj = TimeAwareSymbol(self.base_name, 'ss')
        return obj

    def exit_ss(self):
        obj = TimeAwareSymbol(self.base_name, 0)
        return obj


def step_equation_forward(eq):
    to_step = []

    for variable in set(eq.atoms()):
        if hasattr(variable, 'step_forward'):
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


def convert_symbols_to_time_symbols(eq, ):
    sub_dict = {}
    var_list = [variable for variable in eq.atoms() if isinstance(variable, sp.Symbol)]

    for variable in var_list:
        var_name = variable.name
        if re.search('_\{?t[-+ ]?\d?\}?$', var_name) is not None:
            var_name_pieces = var_name.split('_')
            name_part = '_'.join(s for s in var_name_pieces[:-1])
            time_part = var_name_pieces[-1]

            time_part = re.sub('[\{\}t]', '', time_part)
            if len(time_part) == 0:
                time_index = 0
            else:
                time_index = int(time_part)
            time_var = TimeAwareSymbol(name_part, time_index)
            sub_dict[variable] = time_var

    return eq.subs(sub_dict)


def diff_through_time(eq, dx, discount_factor=1):
    total_dydx = 0
    next_dydx = 1

    while next_dydx != 0:
        next_dydx = eq.diff(dx)
        eq = step_equation_forward(eq) * discount_factor
        total_dydx += next_dydx

    return total_dydx
