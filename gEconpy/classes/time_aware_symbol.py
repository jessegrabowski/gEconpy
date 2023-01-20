from typing import Union

import sympy as sp
from sympy.core.cache import cacheit


class TimeAwareSymbol(sp.Symbol):

    __slots__ = ("time_index", "base_name", "__dict__")
    time_index: Union[int, str]
    base_name: str

    def __new__(cls, name, time_index, **assumptions):
        cls._sanitize(assumptions, cls)

        return TimeAwareSymbol.__xnew__(cls, name, time_index, **assumptions)

    def __getnewargs__(self):
        return self.name, self.time_index

    @staticmethod
    @cacheit
    def __xnew__(cls, name, time_index, **assumptions):
        obj = super().__xnew__(cls, name, **assumptions)
        obj.time_index = time_index
        obj.base_name = name
        obj.name = obj._create_name_from_time_index()
        obj.safe_name = obj.name.replace("+", "p").replace("-", "m")
        return obj

    def _determine_operator(self):
        if self.time_index == "ss":
            return ""
        if self.time_index > 0:
            operator = "+"
        elif self.time_index < 0:
            operator = "-"
        else:
            operator = ""
        return operator

    def _create_name_from_time_index(self):
        operator = self._determine_operator()
        name = self.base_name
        idx = self.time_index
        idx = idx if isinstance(idx, str) else str(abs(idx))

        if idx == "ss":
            time_name = rf"{name}_{idx}"
        elif idx == "0":
            time_name = r"%s_t" % name
        else:
            time_name = rf"{name}_t{operator}{idx}"

        return time_name

    def _hashable_content(self):
        return super()._hashable_content() + (self.time_index,)

    def __getnewargs_ex__(self):
        return (
            (
                self.base_name,
                self.time_index,
            ),
            self.assumptions0,
        )

    def step_forward(self):
        obj = TimeAwareSymbol(self.base_name, self.time_index + 1, **self.assumptions0)
        return obj

    def step_backward(self):
        obj = TimeAwareSymbol(self.base_name, self.time_index - 1, **self.assumptions0)
        return obj

    def to_ss(self):
        obj = TimeAwareSymbol(self.base_name, "ss", **self.assumptions0)
        return obj

    def exit_ss(self):
        obj = TimeAwareSymbol(self.base_name, 0, **self.assumptions0)
        return obj

    def set_t(self, t):
        obj = TimeAwareSymbol(self.base_name, t, **self.assumptions0)
        return obj
