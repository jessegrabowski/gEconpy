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


# Curated rather than derived from ``sympy.core.alphabets.greeks``, which includes ``omicron``. There is no
# ``\omicron`` command in LaTeX or amsmath, so a derived table emits an undefined control sequence. ``epsilon``
# is the one name whose conventional macro rendering is not the plain letter: the shock literature writes
# ``\varepsilon`` throughout.
_LOWERCASE_GREEK = (
    "alpha",
    "beta",
    "gamma",
    "delta",
    "zeta",
    "eta",
    "theta",
    "iota",
    "kappa",
    "lambda",
    "mu",
    "nu",
    "xi",
    "pi",
    "rho",
    "sigma",
    "tau",
    "upsilon",
    "phi",
    "chi",
    "psi",
    "omega",
)

# The capital of every other greek letter is an ordinary Latin letter, so it has no command either.
_UPPERCASE_GREEK = ("Gamma", "Delta", "Theta", "Lambda", "Xi", "Pi", "Sigma", "Upsilon", "Phi", "Psi", "Omega")

_GREEK_COMMANDS = (
    {name: f"\\{name}" for name in _LOWERCASE_GREEK}
    | {name: f"\\{name}" for name in _UPPERCASE_GREEK}
    | {"epsilon": "\\varepsilon"}
)


def render_latex(symbol: "TimeAwareSymbol", stem_override: str | None = None) -> str:
    r"""
    Render one time-aware symbol as LaTeX.

    The stem becomes a greek letter when it names one, and is wrapped in ``\\text{}`` when it is longer than one
    character so a multi-letter name does not read as a product. Anything after the first underscore is a
    subscript, and the time index joins it.

    Parameters
    ----------
    symbol : TimeAwareSymbol
        The symbol to render.
    stem_override : str, optional
        LaTeX for the stem, from a ``symbols`` block ``latex`` field, replacing what would be inferred. The
        subscripts and time index still compose onto it. Defaults to inferring the stem.

    Returns
    -------
    latex : str
        The rendered symbol, without surrounding math delimiters.
    """
    stem, _, remainder = symbol.base_name.partition("_")
    rendered_stem = _latex_stem(stem) if stem_override is None else stem_override

    # Every underscore separates a subscript, and each one is a name in its own right: ``epsilon_beta`` is a
    # greek letter subscripted by another, not by the four letters ``beta``.
    subscripts = [_latex_stem(part) for part in remainder.split("_") if part]
    parts = [*subscripts, symbol._time_subscript()]
    return f"{rendered_stem}_{{{','.join(parts)}}}"


def _latex_stem(stem: str) -> str:
    if stem.istitle() and stem in _GREEK_COMMANDS:
        return _GREEK_COMMANDS[stem]
    if stem.islower() and stem in _GREEK_COMMANDS:
        return _GREEK_COMMANDS[stem]
    return stem if len(stem) == 1 else f"\\text{{{stem}}}"


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

    def _latex(self, printer=None):
        """
        Render as LaTeX, so :func:`sympy.latex` prints a whole equation correctly without further help.

        Sympy calls this before it consults its own ``symbol_names`` setting, so the setting is read here or it
        would never reach a time-aware symbol.
        """
        override = printer._settings.get("symbol_names", {}).get(self) if printer is not None else None
        return override if override is not None else render_latex(self)

    def _time_subscript(self) -> str:
        if self.time_index == "ss":
            return "ss"
        if self.time_index == 0:
            return "t"
        sign = "+" if self.time_index > 0 else "-"
        return f"t{sign}{abs(self.time_index)}"

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
