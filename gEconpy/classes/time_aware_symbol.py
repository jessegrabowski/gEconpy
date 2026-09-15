import sympy as sp

from sympy.core.cache import cacheit

# Domain defaults injected into every parsed Symbol unless the user's assumptions block overrides them. Every DSGE
# quantity (variable, parameter, shock, multiplier) is real-valued and finite, and together these two facts give sympy
# enough type information to apply simplifications it refuses on bare Symbols. positive, nonzero and integer are not
# defaulted: Lagrange multipliers can be negative and shocks are zero in steady state, so those stay opt-in.
DEFAULT_ASSUMPTIONS: dict[str, bool] = {"real": True, "finite": True}


def merge_assumptions(user_assumptions: dict[str, bool] | None) -> dict[str, bool]:
    """
    Merge ``DEFAULT_ASSUMPTIONS`` with user-declared assumptions.

    Parameters
    ----------
    user_assumptions : dict mapping str to bool, optional
        Assumptions declared by the user. Defaults to no assumptions.

    Returns
    -------
    assumptions : dict mapping str to bool
        The merged assumptions. User values win on conflict.
    """
    return {**DEFAULT_ASSUMPTIONS, **(user_assumptions or {})}


class TimeAwareSymbol(sp.Symbol):
    """
    Subclass of :class:`~sympy.core.symbol.Symbol` with a time index.

    The time index enters equality and hashing, so two symbols compare equal only when their name, assumptions, and
    time index all match. Everything else is inherited from :class:`~sympy.core.symbol.Symbol` unchanged.

    Parameters
    ----------
    name : str
        Base name of the symbol, without any time index.
    time_index : int or str
        Time index of the symbol. Use ``"ss"`` for the steady state.
    **assumptions
        Sympy assumptions to attach to the symbol.

    Examples
    --------
    Two symbols with the same base name compare equal only when their time indexes match:

    .. code-block:: python

        from gEconpy.classes.time_aware_symbol import TimeAwareSymbol

        x1 = TimeAwareSymbol("x", time_index=1)
        x2 = TimeAwareSymbol("x", time_index=2)

        print(x1 == x2)  # False
        print(x1 == x2.set_t(1))  # True
        print(x1.step_forward() == x2)  # True
    """

    __slots__ = ("__dict__", "base_name", "time_index")
    time_index: int | str
    base_name: str
    safe_name: str

    def __new__(cls, name, time_index, **assumptions):
        cls._sanitize(assumptions, cls)

        return TimeAwareSymbol.__xnew__(cls, name, time_index, **assumptions)

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

    def _create_name_from_time_index(self):
        if self.time_index == "ss":
            return f"{self.base_name}_ss"
        if self.time_index == 0:
            return f"{self.base_name}_t"
        sign = "+" if self.time_index > 0 else "-"
        return f"{self.base_name}_t{sign}{abs(self.time_index)}"

    def _hashable_content(self):
        return (*super()._hashable_content(), self.time_index)

    def __getnewargs_ex__(self):
        return (self.base_name, self.time_index), self.assumptions0

    def step_forward(self):
        """
        Increment the time index by one. A steady-state symbol is returned unchanged.

        Returns
        -------
        symbol : TimeAwareSymbol
            A new symbol with the same base name and assumptions, one period later, or this symbol if it is at the
            steady state.
        """
        if self.time_index == "ss":
            return self
        return TimeAwareSymbol(self.base_name, self.time_index + 1, **self.assumptions0)

    def step_backward(self):
        """
        Decrement the time index by one. A steady-state symbol is returned unchanged.

        Returns
        -------
        symbol : TimeAwareSymbol
            A new symbol with the same base name and assumptions, one period earlier, or this symbol if it is at the
            steady state.
        """
        if self.time_index == "ss":
            return self
        return TimeAwareSymbol(self.base_name, self.time_index - 1, **self.assumptions0)

    def to_ss(self):
        """
        Set the time index to steady state.

        A steady-state symbol has no time offset, so ``step_forward`` and ``step_backward`` return it unchanged.

        Returns
        -------
        symbol : TimeAwareSymbol
            A new symbol with the same base name and assumptions, at the steady state.
        """
        return TimeAwareSymbol(self.base_name, "ss", **self.assumptions0)

    def exit_ss(self):
        """
        Set the time index to zero if in the steady state, otherwise do nothing.

        Returns
        -------
        symbol : TimeAwareSymbol
            A new symbol at time index zero, or this symbol if it is not at the steady state.
        """
        return TimeAwareSymbol(self.base_name, 0, **self.assumptions0) if self.time_index == "ss" else self

    def set_t(self, t):
        """
        Set the time index to a specific value.

        Parameters
        ----------
        t : int or str
            The time index to set. A string must be ``"ss"``.

        Returns
        -------
        symbol : TimeAwareSymbol
            A new symbol with the same base name and assumptions, at the requested time index.
        """
        if isinstance(t, str) and t != "ss":
            raise ValueError(f"Time index must be an integer or 'ss', but got {t!r}.")
        return TimeAwareSymbol(self.base_name, t, **self.assumptions0)
