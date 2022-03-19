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
