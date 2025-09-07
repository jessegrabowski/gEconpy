import sympy as sp

from sympy.core.cache import cacheit


class TimeAwareSymbol(sp.Symbol):
    """
    Subclass of :class:`sympy.Symbol` with a time index.

    A TimeAwareSymbol is identical to a :class:`symPy.Symbol` in all respects, except that it has a
    time index property that is used when determining equality and hashability. Two symbols with the same name,
    assumptions, and time index evaluate to equal.

    Examples
    --------

    .. code-block:: python

        from gEconpy.classes.time_aware_symbol import TimeAwareSymbol

        x1 = TimeAwareSymbol("x", time_index=1)
        x2 = TimeAwareSymbol("x", time_index=2)

        print(x1 == x2)  # False, time indexes are different
        print(x1 == x2.set_t(1))  # True, time indexes are the same
        print(x1.step_forward() == x2)  # True, time indexes are the same
    """

    __slots__ = ("__dict__", "base_name", "time_index")
    time_index: int | str
    base_name: str
    safe_name: str

    def __new__(cls, name, time_index, **assumptions):
        cls._sanitize(assumptions, cls)

        return TimeAwareSymbol.__xnew__(cls, name, time_index, **assumptions)

    def __getnewargs__(self):
        return self.name, self.time_index

    def _numpycode(self, *args, **kwargs):  # noqa: ARG002
        return self.safe_name

    @staticmethod
    @cacheit
    def __xnew__(symbol_class, name, time_index, **assumptions):
        obj = super().__xnew__(symbol_class, name, **assumptions)
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
            time_name = rf"{name}_t"
        else:
            time_name = rf"{name}_t{operator}{idx}"

        return time_name

    def _hashable_content(self):
        return (*super()._hashable_content(), self.time_index)

    def __getnewargs_ex__(self):
        return (
            (
                self.base_name,
                self.time_index,
            ),
            self.assumptions0,
        )

    def step_forward(self):
        """Increment the time index by one."""
        return TimeAwareSymbol(self.base_name, self.time_index + 1, **self.assumptions0)

    def step_backward(self):
        """Decrement the time index by one."""
        return TimeAwareSymbol(self.base_name, self.time_index - 1, **self.assumptions0)

    def to_ss(self):
        """
        Set the time index to steady state.

        Once in the steady state, :meth:`step_forward` and :meth:`step_backward` will not change the time index.
        """
        return TimeAwareSymbol(self.base_name, "ss", **self.assumptions0)

    def exit_ss(self):
        """Set the time index to zero if in the steady state, otherwise do nothing."""
        return TimeAwareSymbol(self.base_name, 0, **self.assumptions0) if self.time_index == "ss" else self

    def set_t(self, t):
        """
        Set the time index to a specific value.

        Parameters
        ----------
        t: int | str
            The time index to set. If str, must be "ss" .
        """
        if isinstance(t, str) and t != "ss":
            raise ValueError("Time index must be an integer or 'ss'.")
        return TimeAwareSymbol(self.base_name, t, **self.assumptions0)
