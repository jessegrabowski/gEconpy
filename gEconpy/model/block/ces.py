from dataclasses import dataclass

import sympy as sp

from gEconpy.classes.time_aware_symbol import TimeAwareSymbol
from gEconpy.model.block.basic import Block
from gEconpy.model.block.cobb_douglas import _split_output_and_product, _symbols_are_distinct
from gEconpy.model.block.registry import register_block
from gEconpy.utilities import diff_through_time


@register_block
class CESBlock(Block):
    r"""
    A :class:`~gEconpy.model.block.basic.Block` whose single constraint is a CES production function.

    The constraint has the form

    .. math::

        Y = A \left( \sum_{i=1}^{k} \text{share}_i \, x_i^s \right)^{1/s}

    for any :math:`k \geq 2`, where :math:`s = (\psi - 1)/\psi` for elasticity of substitution :math:`\psi`. The
    productivity term :math:`A` is optional and the shares may be any sympy expressions. Cobb-Douglas is the
    :math:`\psi \to 1` limit, and the matcher never collapses to it, because detection is structural and parameter
    values are runtime data.

    The first-order condition for each input uses the identity
    :math:`\partial Y / \partial x_i = \text{share}_i \, A^s \, (Y / x_i)^{1-s}`:

    .. math::

        \frac{\partial \mathcal{L}}{\partial x_i}
            = \frac{\partial \text{obj}}{\partial x_i}
              + \mu \, \text{share}_i \, A^s \left(\frac{Y}{x_i}\right)^{1-s}

    where :math:`\mu` is the multiplier on the production constraint and the :math:`A^s` factor is dropped when the
    constraint has no productivity term. The constraint itself is never passed to :func:`sympy.diff`, so the
    chain-rule expansion through the inner sum and the double exponent :math:`\text{inner}^{1/s - 1}` never enters
    the compiled graph.

    The parser constructs this class through :func:`~gEconpy.model.block.registry.dispatch_block` whenever
    :meth:`detect` matches. Its constructor takes the same arguments as :class:`~gEconpy.model.block.basic.Block`.
    """

    @classmethod
    def detect(
        cls,
        constraints: dict[int, sp.Eq] | None,
        objective: dict[int, sp.Eq] | None,
        identities: dict[int, sp.Eq] | None,  # noqa: ARG003 -- part of the dispatch contract
    ) -> bool:
        """
        Report whether a block is a CES optimization problem.

        The block must have an objective and exactly one constraint of the form
        ``Y = [A *] (sum(share_i * x_i ** s)) ** (1 / s)``. The objective is unconstrained, because the closed-form
        constraint derivative is exact whatever the block maximizes.

        Parameters
        ----------
        constraints : dict mapping int to sympy.Eq, optional
            Block constraints, keyed by equation index.
        objective : dict mapping int to sympy.Eq, optional
            Block objective, keyed by equation index.
        identities : dict mapping int to sympy.Eq, optional
            Block identities. Unused, accepted so every registered subclass shares one signature.

        Returns
        -------
        matched : bool
            True when the block should be constructed as a :class:`CESBlock`.

        Examples
        --------
        The second call prints False because the objective is missing:

        .. code-block:: python

            import sympy as sp

            from gEconpy.model.block.ces import CESBlock

            Y, A, K, L, alpha, psi, Pi, r, w = sp.symbols("Y A K L alpha psi Pi r w")
            s = (psi - 1) / psi
            inner = alpha ** (1 / psi) * K**s + (1 - alpha) ** (1 / psi) * L**s
            constraints = {0: sp.Eq(Y, A * inner ** (1 / s))}
            objective = {1: sp.Eq(Pi, Y - r * K - w * L)}

            print(CESBlock.detect(constraints, objective, identities=None))
            print(CESBlock.detect(constraints, objective=None, identities=None))
        """
        if objective is None:
            return False
        return _match_ces_constraint(constraints) is not None

    def __init__(self, *args, **kwargs):
        """Construct the block. See :class:`~gEconpy.model.block.basic.Block` for the arguments."""
        super().__init__(*args, **kwargs)
        self._match = _match_ces_constraint(self.constraints)
        if self._match is None:
            raise RuntimeError(
                f"CESBlock {self.name!r} constructed without a matching CES constraint. Construct blocks through "
                "dispatch_block, which only selects this class when detect matches."
            )

    def _compute_foc(
        self,
        control: TimeAwareSymbol,
        lagrange: sp.Expr,
        discount_factor: sp.Expr | int,
    ) -> sp.Expr:
        r"""
        Compute the first-order condition for one control, in closed form when the control is a production input.

        For an input :math:`x_i` the condition is

        .. math::

            \frac{\partial \text{obj}}{\partial x_i}
              + \mu \, \text{share}_i \, A^s \left(\frac{Y}{x_i}\right)^{1-s}

        with the objective derivative taken by :func:`~gEconpy.utilities.diff_through_time`. Any other control
        falls back to differentiating the full Lagrangian.

        Parameters
        ----------
        control : TimeAwareSymbol
            Control variable to differentiate against.
        lagrange : sympy.Expr
            Full Lagrangian, used only on the fallback path.
        discount_factor : sympy.Expr or int
            Discount factor applied to forward-shifted derivative terms.

        Returns
        -------
        foc : sympy.Expr
            First-order condition residual.
        """
        match = self._match
        mu = self.multipliers[match.idx]
        if mu is None:
            raise RuntimeError(
                f"CESBlock {self.name!r} has no multiplier on its production constraint. Call solve_optimization, "
                "which generates one, before computing first-order conditions."
            )

        obj_idx, obj_eq = next(iter(self.objective.items()))
        objective_rhs = obj_eq.rhs
        if self.equation_flags.get(obj_idx, {}).get("minimize", False):
            objective_rhs = -objective_rhs
        objective_term = diff_through_time(objective_rhs, control, discount_factor)

        productivity_factor = sp.S.One if match.productivity is None else match.productivity**match.exponent
        for input_symbol, share in match.inputs:
            if control == input_symbol:
                marginal_product = share * productivity_factor * (match.output / input_symbol) ** (1 - match.exponent)
                return objective_term + mu * marginal_product

        return diff_through_time(lagrange, control, discount_factor)


@dataclass(frozen=True)
class _CESMatch:
    idx: int
    output: sp.Symbol
    productivity: sp.Symbol | None
    exponent: sp.Expr
    inputs: list[tuple[sp.Symbol, sp.Expr]]


def _match_ces_constraint(constraints: dict[int, sp.Eq] | None) -> _CESMatch | None:
    r"""
    Match a single CES production constraint of any arity.

    The match is conservative. It requires exactly one constraint whose residual is
    :math:`-Y + [A] (\text{inner})^{1/s}` with no extra additive terms and no numeric coefficient, an inner bracket
    that is a sum of terms each holding exactly one ``Pow`` of a distinct ``Symbol`` with the shared exponent
    :math:`s`, an outer exponent satisfying :math:`\text{outer} \cdot s = 1`, and :math:`Y`, :math:`A`, and the
    inputs all distinct.

    Parameters
    ----------
    constraints : dict mapping int to sympy.Eq, optional
        Block constraints, keyed by equation index.

    Returns
    -------
    match : _CESMatch or None
        The decomposed constraint, or None when the constraint does not have the form.
    """
    if not constraints or len(constraints) != 1:
        return None
    idx, eq = next(iter(constraints.items()))

    # The residual is deliberately not expanded. sp.expand distributes the outer power across the bracket, for
    # example rewriting K**((psi - 1)/psi) as K/K**(1/psi), and destroys the structure the matcher relies on.
    for residual in (eq.rhs - eq.lhs, eq.lhs - eq.rhs):
        split = _split_output_and_product(residual)
        if split is None:
            continue
        output, product_term = split

        outer = _decompose_ces_outer(product_term)
        if outer is None:
            continue
        productivity, inner_sum, outer_exponent = outer

        inner = _decompose_ces_inner(inner_sum, outer_exponent)
        if inner is None:
            continue
        exponent, inputs = inner

        if not _symbols_are_distinct(output, productivity, inputs):
            continue

        return _CESMatch(idx=idx, output=output, productivity=productivity, exponent=exponent, inputs=inputs)

    return None


def _decompose_ces_outer(product_term: sp.Expr) -> tuple[sp.Symbol | None, sp.Add, sp.Expr] | None:
    """
    Decompose ``[A *] (inner_sum) ** outer_exponent`` into the optional productivity symbol, the sum, and the exponent.

    At most one bare ``Symbol`` is allowed and is taken as the productivity term. Exactly one ``Pow`` with an ``Add``
    base is required. Any other factor rejects the expression.
    """
    productivity = None
    pow_factor = None
    for factor in sp.Mul.make_args(product_term):
        if isinstance(factor, sp.Symbol):
            if productivity is not None:
                return None
            productivity = factor
        elif isinstance(factor, sp.Pow) and isinstance(factor.args[0], sp.Add):
            if pow_factor is not None:
                return None
            pow_factor = factor
        else:
            return None
    if pow_factor is None:
        return None
    inner_sum, outer_exponent = pow_factor.args
    return productivity, inner_sum, outer_exponent


def _decompose_ces_inner(
    inner_sum: sp.Add, outer_exponent: sp.Expr
) -> tuple[sp.Expr, list[tuple[sp.Symbol, sp.Expr]]] | None:
    r"""
    Decompose the CES bracket :math:`\sum_i \text{share}_i \, x_i^s` into the exponent and the input-share pairs.

    The input exponent is the unique exponent in the bracket whose product with ``outer_exponent`` simplifies to 1.
    That identity is what distinguishes it from exponents inside the shares, such as :math:`\alpha^{1/\psi}`. Each
    addend must hold exactly one ``Pow`` of a ``Symbol`` with that exponent, and its remaining factors multiply to
    the share.
    """
    candidate_exponents: list[sp.Expr] = []
    for term in inner_sum.args:
        for factor in sp.Mul.make_args(term):
            if not isinstance(factor, sp.Pow):
                continue
            is_new = not any(sp.simplify(factor.args[1] - candidate) == 0 for candidate in candidate_exponents)
            if is_new:
                candidate_exponents.append(factor.args[1])

    reciprocal_candidates = [c for c in candidate_exponents if sp.simplify(outer_exponent * c - 1) == 0]
    if len(reciprocal_candidates) != 1:
        return None
    exponent = reciprocal_candidates[0]

    inputs: list[tuple[sp.Symbol, sp.Expr]] = []
    for term in inner_sum.args:
        input_symbol = None
        share_factors = []
        for factor in sp.Mul.make_args(term):
            if isinstance(factor, sp.Pow) and sp.simplify(factor.args[1] - exponent) == 0:
                if input_symbol is not None:
                    return None
                if not isinstance(factor.args[0], sp.Symbol):
                    return None
                input_symbol = factor.args[0]
            else:
                share_factors.append(factor)
        if input_symbol is None:
            return None
        share = sp.Mul(*share_factors) if share_factors else sp.S.One
        inputs.append((input_symbol, share))

    bases = [base for base, _ in inputs]
    if len(set(bases)) != len(bases):
        return None

    return exponent, inputs
