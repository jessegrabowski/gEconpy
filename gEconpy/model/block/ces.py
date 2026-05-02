import sympy as sp

from gEconpy.classes.time_aware_symbol import TimeAwareSymbol
from gEconpy.model.block.basic import Block
from gEconpy.model.block.registry import register_block
from gEconpy.utilities import diff_through_time


def _decompose_ces_outer(prod_term: sp.Expr) -> tuple[sp.Symbol | None, sp.Add, sp.Expr] | None:
    r"""Decompose ``[A *] (inner_sum)^outer_exp`` into its three pieces.

    Walks ``Mul.make_args`` and demands at most one bare ``Symbol`` (productivity ``A``, optional) and exactly one
    ``Pow`` whose base is an ``Add`` (the inner CES bracket). The leading ``A`` is optional: many calibrated DSGE
    models drop it (or absorb it into the shares), so :math:`Y = (\sum_i \text{share}_i \cdot x_i^s)^{1/s}` is a
    legitimate form. Rejects anything else: extra constants, multiple Pows, Pow whose base is not an Add.

    Parameters
    ----------
    prod_term : sympy.Expr
        Expression assumed to take the form :math:`[A \cdot] (\sum_i \text{share}_i \cdot x_i^s)^{1/s}`.

    Returns
    -------
    decomposition : tuple of (Symbol or None, Add, Expr) or None
        ``(A, inner_sum, outer_exp)`` on success (``A`` is None when no leading productivity term is present),
        None on rejection.
    """
    A = None
    pow_factor = None
    for f in sp.Mul.make_args(prod_term):
        if isinstance(f, sp.Symbol):
            if A is not None:
                return None
            A = f
        elif isinstance(f, sp.Pow) and isinstance(f.args[0], sp.Add):
            if pow_factor is not None:
                return None
            pow_factor = f
        else:
            return None
    if pow_factor is None:
        return None
    inner_sum, outer_exp = pow_factor.args
    return A, inner_sum, outer_exp


def _decompose_ces_inner(
    inner_sum: sp.Add, outer_exp: sp.Expr
) -> tuple[sp.Expr, list[tuple[sp.Symbol, sp.Expr]]] | None:
    r"""Decompose the CES inner bracket :math:`\sum_i \text{share}_i \cdot x_i^s`.

    The defining algebraic property of CES is that the outer exponent is the reciprocal of the input exponent:
    :math:`\text{outer\_exp} \cdot s = 1`. The matcher uses this identity to pick out which exponent in the inner sum
    is the input exponent (versus, e.g., the share parameter's own exponent like :math:`\alpha^{1/\psi}`, which has a
    different value).

    Each addend must contain exactly one ``Pow`` whose base is a ``Symbol`` and whose exponent equals the chosen
    :math:`s`; the remaining factors multiply to give the share coefficient. Returns None on any structural deviation.

    Parameters
    ----------
    inner_sum : sympy.Add
        The bracket expression.
    outer_exp : sympy.Expr
        The exponent applied to the bracket in the outer ``Pow``.

    Returns
    -------
    decomposition : tuple of (Expr, list of (Symbol, Expr)) or None
        ``(s, [(x_1, share_1), ..., (x_k, share_k)])`` on success, None on rejection.
    """
    candidate_exponents: list[sp.Expr] = []
    for term in inner_sum.args:
        for f in sp.Mul.make_args(term):
            if isinstance(f, sp.Pow) and not any(sp.simplify(f.args[1] - c) == 0 for c in candidate_exponents):
                candidate_exponents.append(f.args[1])

    s_candidates = [c for c in candidate_exponents if sp.simplify(outer_exp * c - 1) == 0]
    if len(s_candidates) != 1:
        return None
    s = s_candidates[0]

    inputs: list[tuple[sp.Symbol, sp.Expr]] = []
    for term in inner_sum.args:
        x = None
        share_factors = []
        for f in sp.Mul.make_args(term):
            if isinstance(f, sp.Pow) and sp.simplify(f.args[1] - s) == 0:
                if x is not None:
                    return None
                if not isinstance(f.args[0], sp.Symbol):
                    return None
                x = f.args[0]
            else:
                share_factors.append(f)
        if x is None:
            return None
        share = sp.Mul(*share_factors) if share_factors else sp.S.One
        inputs.append((x, share))

    bases = [x for x, _ in inputs]
    if len(set(bases)) != len(bases):
        return None

    return s, inputs


def _match_ces_constraint(constraints: dict[int, sp.Eq] | None) -> dict | None:
    r"""Match a single CES production constraint of arbitrary input arity.

    The general form is

    .. math::

        Y = A \cdot \left( \sum_{i=1}^{k} \text{share}_i \cdot x_i^s \right)^{1/s}

    where :math:`s = (\psi - 1)/\psi` for an elasticity of substitution :math:`\psi`. The shares :math:`\text{share}_i`
    are typically parameter expressions (e.g. :math:`\alpha^{1/\psi}` and :math:`(1-\alpha)^{1/\psi}` in the standard
    two-input form), but the matcher accepts any sympy expression for the share — the closed-form FOC identity does
    not depend on the share's internal structure.

    Match is conservative: requires exactly one constraint, the residual to decompose cleanly into
    :math:`-Y + [A \cdot] (\text{inner})^{1/s}` (no extra additive terms, no extra multiplicative constants beyond the
    optional leading productivity Symbol :math:`A`), the inner bracket to be a sum of terms each containing exactly
    one ``Pow`` of a ``Symbol`` with a shared exponent :math:`s`, and the outer exponent to satisfy
    :math:`\text{outer\_exp} \cdot s = 1` (the defining CES algebraic identity). The leading :math:`A` is optional;
    when absent, the closed-form FOC drops the :math:`A^s` factor.

    Parameters
    ----------
    constraints : dict mapping int to sympy.Eq, optional
        Block constraints keyed by equation index, as held on a :class:`Block`.

    Returns
    -------
    match : dict or None
        On match, ``{"idx": int, "Y": Symbol, "A": Symbol, "s": Expr, "inputs": list of (Symbol, Expr)}``. Otherwise
        None.
    """
    if not constraints or len(constraints) != 1:
        return None
    idx, eq = next(iter(constraints.items()))

    # NOTE: do NOT call sp.expand on the residual. The CES outer Pow over a sum gets distributed in nasty ways
    # (e.g. K^((psi-1)/psi) = K/K^(1/psi), which sp.expand pulls out across the bracket), destroying the structure
    # the matcher relies on.
    for residual in (eq.rhs - eq.lhs, eq.lhs - eq.rhs):
        if not isinstance(residual, sp.Add) or len(residual.args) != 2:
            continue

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

        outer = _decompose_ces_outer(prod_term)
        if outer is None:
            continue
        A, inner_sum, outer_exp = outer

        inner = _decompose_ces_inner(inner_sum, outer_exp)
        if inner is None:
            continue
        s, inputs = inner

        all_syms = {Y_sym, *(x for x, _ in inputs)}
        if A is not None:
            all_syms.add(A)
        expected_count = len(inputs) + (2 if A is not None else 1)
        if len(all_syms) != expected_count:
            continue

        return {"idx": idx, "Y": Y_sym, "A": A, "s": s, "inputs": inputs}

    return None


@register_block
class CESBlock(Block):
    r"""A :class:`Block` whose constraint is a CES production function of arbitrary input count.

    The constraint takes the general form

    .. math::

        Y = A \cdot \left( \sum_{i=1}^{k} \text{share}_i \cdot x_i^s \right)^{1/s}

    where :math:`s = (\psi - 1)/\psi` is the input exponent (elasticity of substitution :math:`\psi` parameterizes
    :math:`s`). The two-input form with :math:`\text{share}_1 = \alpha^{1/\psi}` and
    :math:`\text{share}_2 = (1-\alpha)^{1/\psi}` is the standard parameterization in DSGE practice, but the matcher
    accepts any share expressions. Cobb-Douglas is the :math:`\psi \to 1` limit; the matcher does not collapse to
    Cobb-Douglas in that case (parameter values are runtime data, not structural).

    The first-order conditions for the constraint side are emitted in closed form via the identity
    :math:`\partial Y / \partial x_i = \text{share}_i \cdot A^s \cdot (Y / x_i)^{1-s}`:

    .. math::

        \frac{\partial \mathcal{L}}{\partial x_i}
            = \frac{\partial \text{obj}}{\partial x_i}
              + \mu \cdot \text{share}_i \cdot A^s \cdot \left(\frac{Y}{x_i}\right)^{1-s}

    where :math:`\mu` is the Lagrange multiplier on the production constraint. The constraint itself is never
    differentiated by :func:`sympy.diff`, avoiding the chain-rule expansion that involves both the inner-sum
    subgraph and the double-exponent :math:`\text{inner}^{1/s - 1}`.
    """

    @classmethod
    def detect(
        cls,
        constraints: dict[int, sp.Eq] | None,
        objective: dict[int, sp.Eq] | None,
        identities: dict[int, sp.Eq] | None,  # noqa: ARG003 — part of the dispatch contract; other subclasses use it
    ) -> bool:
        """Conservative match for a CES production block.

        The block must have an objective (it is an optimization, not an identity-only block) and exactly one
        constraint whose residual matches the CES form via :func:`_match_ces_constraint`. The objective itself is not
        constrained: the closed-form constraint derivative is exact regardless of what the firm is maximizing, so
        :meth:`_compute_foc` simply differentiates the objective symbolically and adds the closed-form constraint
        term.

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
            True if the block is a canonical CES optimization. False otherwise; caller falls back to the general
            :class:`Block` (or another more-specific subclass earlier in the registry).
        """
        if objective is None:
            return False
        return _match_ces_constraint(constraints) is not None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._ces_match = _match_ces_constraint(self.constraints)
        if self._ces_match is None:
            raise RuntimeError(
                f"CESBlock {self.name!r} constructed without matching CES constraint. This is a dispatcher bug."
            )

    def _compute_foc(
        self,
        control: TimeAwareSymbol,
        lagrange: sp.Expr,
        discount_factor: sp.Expr | int,
    ) -> sp.Expr:
        r"""Closed-form first-order condition for a CES production constraint.

        For the Lagrangian

        .. math::

            \mathcal{L} = \text{obj}
                - \mu \cdot \left( Y - A \left( \sum_i \text{share}_i \cdot x_i^s \right)^{1/s} \right)

        the first-order condition with respect to input :math:`x_i` reduces algebraically to

        .. math::

            \frac{\partial \mathcal{L}}{\partial x_i}
                = \frac{\partial \text{obj}}{\partial x_i}
                  + \mu \cdot \text{share}_i \cdot A^s \cdot \left(\frac{Y}{x_i}\right)^{1-s}

        The objective derivative is taken via :func:`diff_through_time` so multi-period objectives with continuation
        values compose correctly. If ``control`` is not one of the production inputs, falls back to standard
        differentiation of the full Lagrangian.

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
        Y = self._ces_match["Y"]
        A = self._ces_match["A"]
        s = self._ces_match["s"]
        mu = self.multipliers[self._ces_match["idx"]]
        if mu is None:
            raise RuntimeError(
                f"CESBlock {self.name!r}: no multiplier for the production constraint. This is a base class bug."
            )

        obj_idx, obj_eq = next(iter(self.objective.items()))
        objective_rhs = obj_eq.rhs
        if self.equation_flags.get(obj_idx, {}).get("minimize", False):
            objective_rhs = -objective_rhs
        obj_term = diff_through_time(objective_rhs, control, discount_factor)

        for x_i, share_i in self._ces_match["inputs"]:
            if control == x_i:
                # If the user wrote the constraint without a leading productivity ``A``, the closed-form FOC drops
                # the ``A^s`` factor entirely.
                productivity_factor = A**s if A is not None else sp.S.One
                return obj_term + mu * share_i * productivity_factor * (Y / x_i) ** (1 - s)

        return diff_through_time(lagrange, control, discount_factor)
