import sympy as sp

from gEconpy.classes.time_aware_symbol import TimeAwareSymbol
from gEconpy.model.block.basic import Block
from gEconpy.model.block.registry import register_block
from gEconpy.utilities import diff_through_time


def _decompose_monomial(prod_expr: sp.Expr) -> tuple[sp.Symbol, list[tuple[sp.Symbol, sp.Expr]]] | None:
    r"""Decompose ``A * prod(x_i ** a_i)`` into ``A`` and the input list ``[(x_i, a_i), ...]``.

    Walks ``Mul.make_args`` and classifies each factor: ``Pow(Symbol, expr)`` is an input, a bare ``Symbol`` is the
    productivity coefficient ``A``. Rejects expressions containing constants, non-Symbol Pow bases, multiple bare
    Symbols (ambiguous A), or duplicate input bases.

    Parameters
    ----------
    prod_expr : sympy.Expr
        Expression assumed to be of the form :math:`A \prod_i x_i^{a_i}`.

    Returns
    -------
    decomposition : tuple of (Symbol, list of (Symbol, Expr)) or None
        ``(A, [(x_1, a_1), ..., (x_k, a_k)])`` on success, None on rejection.
    """
    factors = sp.Mul.make_args(prod_expr)
    A = None
    inputs: list[tuple[sp.Symbol, sp.Expr]] = []

    for f in factors:
        if isinstance(f, sp.Pow):
            base, exp = f.args
            if not isinstance(base, sp.Symbol):
                return None
            inputs.append((base, exp))
        elif isinstance(f, sp.Symbol):
            if A is not None:
                return None
            A = f
        else:
            return None

    if A is None or not inputs:
        return None

    bases = [x for x, _ in inputs]
    if len(set(bases)) != len(bases):
        return None

    return A, inputs


def _match_cobb_douglas_constraint(constraints: dict[int, sp.Eq] | None) -> dict | None:
    r"""Match a single Cobb-Douglas (monomial) production constraint of arbitrary arity.

    The general form is

    .. math::

        Y = A \cdot \prod_{i=1}^{k} x_i^{a_i}

    for any :math:`k \geq 1` and any symbolic exponents :math:`a_i`. The constant-returns-to-scale special case
    :math:`\sum_i a_i = 1` is the most common parameterization but is not required: the closed-form first-order
    conditions :math:`\partial Y / \partial x_i = a_i Y / x_i` hold for any monomial regardless of the exponent sum.

    Match is conservative: requires exactly one constraint, the residual to decompose cleanly into
    :math:`-Y + A \cdot \prod x_i^{a_i}` (no extra additive terms, no extra multiplicative constants), all input
    bases :math:`x_i` to be distinct sympy ``Symbol`` instances with explicit ``Pow`` exponents (so a bare
    ``Y = A * x`` is rejected as ambiguous between a single-input monomial and a degenerate two-Symbol product), and
    ``Y`` and ``A`` to be distinct ``Symbol`` instances disjoint from the input bases.

    Parameters
    ----------
    constraints : dict mapping int to sympy.Eq, optional
        Block constraints keyed by equation index, as held on a :class:`Block`.

    Returns
    -------
    match : dict or None
        On match, ``{"idx": int, "Y": Symbol, "A": Symbol, "inputs": list of (Symbol, Expr)}``. Otherwise None.
    """
    if not constraints or len(constraints) != 1:
        return None
    idx, eq = next(iter(constraints.items()))

    for raw in (eq.rhs - eq.lhs, eq.lhs - eq.rhs):
        residual = sp.expand(raw)
        if not isinstance(residual, sp.Add) or len(residual.args) != 2:
            continue

        # One term must be -Y (a bare Symbol with coefficient -1); the other is A * prod(x_i^a_i).
        Y_sym = None
        prod_term = None
        for term in residual.args:
            coeff, rest = term.as_coeff_Mul()
            if coeff == -1 and isinstance(rest, sp.Symbol) and Y_sym is None:
                Y_sym = rest
            else:
                prod_term = term

        if Y_sym is None or prod_term is None:
            continue

        decomp = _decompose_monomial(prod_term)
        if decomp is None:
            continue
        A, inputs = decomp

        all_syms = {Y_sym, A, *(x for x, _ in inputs)}
        if len(all_syms) != len(inputs) + 2:
            continue

        return {"idx": idx, "Y": Y_sym, "A": A, "inputs": inputs}

    return None


@register_block
class CobbDouglasBlock(Block):
    r"""A :class:`Block` whose constraint is a Cobb-Douglas production function of arbitrary input count.

    The constraint takes the general monomial form

    .. math::

        Y = A \cdot \prod_{i=1}^{k} x_i^{a_i}

    where :math:`k \geq 1` and the exponents :math:`a_i` are arbitrary symbolic expressions. The two most common
    cases in DSGE practice are :math:`k = 2` (capital and labor, with the constant-returns-to-scale special case
    :math:`a_2 = 1 - a_1`) and :math:`k = 3` (capital, labor, and energy or land). Higher arities are supported
    uniformly with no additional code.

    The first-order conditions for the constraint side are emitted in closed form via the identity
    :math:`\partial Y / \partial x_i = a_i Y / x_i`:

    .. math::

        \frac{\partial \mathcal{L}}{\partial x_i}
            = \frac{\partial \text{obj}}{\partial x_i} + \mu \cdot a_i \cdot \frac{Y}{x_i}

    where :math:`\mu` is the Lagrange multiplier on the production constraint. The constraint itself is never
    differentiated by :func:`sympy.diff`, avoiding the chain-rule expansion :math:`a_i A x_i^{a_i - 1} \prod_{j \neq
    i} x_j^{a_j}` that would otherwise propagate through downstream pytensor compilation.
    """

    @classmethod
    def detect(
        cls,
        constraints: dict[int, sp.Eq] | None,
        objective: dict[int, sp.Eq] | None,
        identities: dict[int, sp.Eq] | None,  # noqa: ARG003 — part of the dispatch contract; other subclasses use it
    ) -> bool:
        """Conservative match for a Cobb-Douglas production block.

        The block must have an objective (it is an optimization, not an identity-only block) and exactly one
        constraint whose residual matches the Cobb-Douglas form via :func:`_match_cobb_douglas_constraint`. The
        objective itself is not constrained: the closed-form constraint derivative is exact regardless of what the
        firm is maximizing, so :meth:`_compute_foc` simply differentiates the objective symbolically (cheap for
        typical linear cost functions) and adds the closed-form constraint term.

        Parameters
        ----------
        constraints : dict mapping int to sympy.Eq, optional
            Block constraints keyed by equation index.
        objective : dict mapping int to sympy.Eq, optional
            Block objective keyed by equation index. Must be present.
        identities : dict mapping int to sympy.Eq, optional
            Block identities. Not used for matching; accepted for interface compatibility with the registry.

        Returns
        -------
        match : bool
            True if the block is a canonical Cobb-Douglas optimization. False otherwise; caller falls back to the
            general :class:`Block`.
        """
        if objective is None:
            return False
        return _match_cobb_douglas_constraint(constraints) is not None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cd_match = _match_cobb_douglas_constraint(self.constraints)
        if self._cd_match is None:
            raise RuntimeError(
                f"CobbDouglasBlock {self.name!r} constructed without matching Cobb-Douglas constraint. "
                "This is a dispatcher bug."
            )

    def _compute_foc(
        self,
        control: TimeAwareSymbol,
        lagrange: sp.Expr,
        discount_factor: sp.Expr | int,
    ) -> sp.Expr:
        r"""Closed-form first-order condition for a Cobb-Douglas production constraint.

        For the Lagrangian

        .. math::

            \mathcal{L} = \text{obj} - \mu \cdot \left(Y - A \prod_i x_i^{a_i}\right)

        the first-order condition with respect to input :math:`x_i` reduces algebraically to

        .. math::

            \frac{\partial \mathcal{L}}{\partial x_i}
                = \frac{\partial \text{obj}}{\partial x_i} + \mu \cdot a_i \cdot \frac{Y}{x_i}

        The objective derivative is taken via :func:`diff_through_time` so multi-period objectives with continuation
        values compose correctly. If ``control`` is not one of the production inputs (e.g. an extra control variable
        in the same block), falls back to standard differentiation of the full Lagrangian.

        Parameters
        ----------
        control : TimeAwareSymbol
            Control variable to differentiate against.
        lagrange : sympy.Expr
            Full Lagrangian, used only for the fallback path.
        discount_factor : sympy.Expr or int
            Discount factor applied to forward time-shifted derivative terms.

        Returns
        -------
        foc : sympy.Expr
            First-order condition residual.
        """
        Y = self._cd_match["Y"]
        mu = self.multipliers[self._cd_match["idx"]]
        if mu is None:
            raise RuntimeError(
                f"CobbDouglasBlock {self.name!r}: no multiplier for the production constraint. "
                "This is a base class bug."
            )

        # Mirror the @minimize sign flip applied in Block._build_lagrangian.
        obj_idx, obj_eq = next(iter(self.objective.items()))
        objective_rhs = obj_eq.rhs
        if self.equation_flags.get(obj_idx, {}).get("minimize", False):
            objective_rhs = -objective_rhs
        obj_term = diff_through_time(objective_rhs, control, discount_factor)

        for x_i, a_i in self._cd_match["inputs"]:
            if control == x_i:
                return obj_term + mu * a_i * Y / x_i

        return diff_through_time(lagrange, control, discount_factor)
