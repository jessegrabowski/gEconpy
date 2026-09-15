from dataclasses import dataclass

import sympy as sp

from gEconpy.classes.time_aware_symbol import TimeAwareSymbol
from gEconpy.model.block.basic import Block
from gEconpy.model.block.registry import register_block
from gEconpy.utilities import diff_through_time


@register_block
class CobbDouglasBlock(Block):
    r"""
    A :class:`~gEconpy.model.block.basic.Block` whose single constraint is a Cobb-Douglas production function.

    The constraint has the monomial form

    .. math::

        Y = A \prod_{i=1}^{k} x_i^{a_i}

    for any :math:`k \geq 1` and any symbolic exponents :math:`a_i`. The productivity term :math:`A` is optional.
    Constant returns to scale, :math:`\sum_i a_i = 1`, is the common case and is not required.

    The first-order condition for each input uses the identity :math:`\partial Y / \partial x_i = a_i Y / x_i`:

    .. math::

        \frac{\partial \mathcal{L}}{\partial x_i}
            = \frac{\partial \text{obj}}{\partial x_i} + \mu \, a_i \, \frac{Y}{x_i}

    where :math:`\mu` is the multiplier on the production constraint. The constraint itself is never passed to
    :func:`sympy.diff`, so the chain-rule expansion :math:`a_i A x_i^{a_i - 1} \prod_{j \neq i} x_j^{a_j}` never
    enters the compiled graph.

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
        Report whether a block is a Cobb-Douglas optimization problem.

        The block must have an objective and exactly one constraint of the form ``Y = [A *] prod(x_i ** a_i)``. The
        objective is unconstrained, because the closed-form constraint derivative is exact whatever the block
        maximizes.

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
            True when the block should be constructed as a :class:`CobbDouglasBlock`.

        Examples
        --------
        The second call prints False because the objective is missing:

        .. code-block:: python

            import sympy as sp

            from gEconpy.model.block.cobb_douglas import CobbDouglasBlock

            Y, A, K, L, alpha, Pi, r, w = sp.symbols("Y A K L alpha Pi r w")
            constraints = {0: sp.Eq(Y, A * K**alpha * L ** (1 - alpha))}
            objective = {1: sp.Eq(Pi, Y - r * K - w * L)}

            print(CobbDouglasBlock.detect(constraints, objective, identities=None))
            print(CobbDouglasBlock.detect(constraints, objective=None, identities=None))
        """
        if objective is None:
            return False
        return _match_cobb_douglas_constraint(constraints) is not None

    def __init__(self, *args, **kwargs):
        """Construct the block. See :class:`~gEconpy.model.block.basic.Block` for the arguments."""
        super().__init__(*args, **kwargs)
        self._match = _match_cobb_douglas_constraint(self.constraints)
        if self._match is None:
            raise RuntimeError(
                f"CobbDouglasBlock {self.name!r} constructed without a matching Cobb-Douglas constraint. Construct "
                "blocks through dispatch_block, which only selects this class when detect matches."
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

            \frac{\partial \text{obj}}{\partial x_i} + \mu \, a_i \, \frac{Y}{x_i}

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
        for input_symbol, exponent in match.inputs:
            if control == input_symbol:
                mu = self._constraint_multiplier(match.idx)
                marginal_product = exponent * match.output / input_symbol
                return self._objective_derivative(control, discount_factor) + mu * marginal_product

        return diff_through_time(lagrange, control, discount_factor)


@dataclass(frozen=True)
class _CobbDouglasMatch:
    idx: int
    output: sp.Symbol
    productivity: sp.Symbol | None
    inputs: list[tuple[sp.Symbol, sp.Expr]]


def _match_cobb_douglas_constraint(constraints: dict[int, sp.Eq] | None) -> _CobbDouglasMatch | None:
    r"""
    Match a single Cobb-Douglas production constraint of any arity.

    The match is conservative. It requires exactly one constraint whose residual is
    :math:`-Y + [A] \prod_i x_i^{a_i}` with no extra additive terms and no numeric coefficient, every input written
    as an explicit ``Pow`` of a distinct ``Symbol``, and :math:`Y`, :math:`A`, and the inputs all distinct. A bare
    ``Y = A * x`` is rejected because it is ambiguous between a one-input monomial and a two-symbol product.

    Parameters
    ----------
    constraints : dict mapping int to sympy.Eq, optional
        Block constraints, keyed by equation index.

    Returns
    -------
    match : _CobbDouglasMatch or None
        The decomposed constraint, or None when the constraint does not have the form.
    """
    if not constraints or len(constraints) != 1:
        return None
    idx, eq = next(iter(constraints.items()))

    for raw_residual in (eq.rhs - eq.lhs, eq.lhs - eq.rhs):
        residual = sp.expand(raw_residual)
        split = _split_output_and_product(residual)
        if split is None:
            continue
        output, product_term = split

        decomposition = _decompose_monomial(product_term)
        if decomposition is None:
            continue
        productivity, inputs = decomposition

        if not _symbols_are_distinct(output, productivity, inputs):
            continue

        return _CobbDouglasMatch(idx=idx, output=output, productivity=productivity, inputs=inputs)

    return None


def _split_output_and_product(residual: sp.Expr) -> tuple[sp.Symbol, sp.Expr] | None:
    """Split a two-term residual into the bare output symbol carrying coefficient -1 and the remaining term."""
    if not isinstance(residual, sp.Add) or len(residual.args) != 2:
        return None

    output = None
    product_term = None
    for term in residual.args:
        coeff, rest = term.as_coeff_Mul()
        if coeff == -1 and isinstance(rest, sp.Symbol) and output is None:
            output = rest
        else:
            product_term = term

    if output is None or product_term is None:
        return None
    return output, product_term


def _decompose_monomial(product_term: sp.Expr) -> tuple[sp.Symbol | None, list[tuple[sp.Symbol, sp.Expr]]] | None:
    """
    Decompose ``[A *] prod(x_i ** a_i)`` into the optional productivity symbol and the ``(x_i, a_i)`` pairs.

    Every ``Pow`` factor with a ``Symbol`` base is an input. At most one bare ``Symbol`` is allowed and is taken as
    the productivity term. Any other factor, a repeated input base, or an empty input list rejects the expression.
    """
    productivity = None
    inputs: list[tuple[sp.Symbol, sp.Expr]] = []

    for factor in sp.Mul.make_args(product_term):
        if isinstance(factor, sp.Pow):
            base, exponent = factor.args
            if not isinstance(base, sp.Symbol):
                return None
            inputs.append((base, exponent))
        elif isinstance(factor, sp.Symbol):
            if productivity is not None:
                return None
            productivity = factor
        else:
            return None

    if not inputs:
        return None

    bases = [base for base, _ in inputs]
    if len(set(bases)) != len(bases):
        return None

    return productivity, inputs


def _symbols_are_distinct(
    output: sp.Symbol,
    productivity: sp.Symbol | None,
    inputs: list[tuple[sp.Symbol, sp.Expr]],
) -> bool:
    symbols = {output, *(base for base, _ in inputs)}
    expected_count = len(inputs) + 1
    if productivity is not None:
        symbols.add(productivity)
        expected_count += 1
    return len(symbols) == expected_count
