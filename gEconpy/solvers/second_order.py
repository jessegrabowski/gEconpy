from typing import NamedTuple

import pytensor.tensor as pt

from pytensor.tensor import TensorVariable

from gEconpy.model.perturbation import DROrder
from gEconpy.solvers.stein import AdjointMethod, solve_generalized_sylvester


class SecondOrderTerms(NamedTuple):
    r"""
    Second-order coefficients of the decision rule, rows in decision-rule order.

    The decision rule is

    .. math::

        \hat{y}_t = g_x \hat{y}^s_{t-1} + g_u \varepsilon_t + \tfrac{1}{2} g_{xx} (\hat{y}^s_{t-1} \otimes
        \hat{y}^s_{t-1}) + g_{xu} (\hat{y}^s_{t-1} \otimes \varepsilon_t) + \tfrac{1}{2} g_{uu} (\varepsilon_t
        \otimes \varepsilon_t) + \tfrac{1}{2} g_{\sigma\sigma},

    where :math:`\hat{y}^s` is the state block and :math:`g_{\sigma\sigma} = G \operatorname{vec}(\Sigma)` for shock
    covariance :math:`\Sigma`.

    Attributes
    ----------
    ghxx : TensorVariable
        Second derivative with respect to the states, of shape ``(n_variables, n_states, n_states)``.
    ghxu : TensorVariable
        Cross derivative with respect to the states and the shocks, of shape ``(n_variables, n_states, n_shocks)``.
    ghuu : TensorVariable
        Second derivative with respect to the shocks, of shape ``(n_variables, n_shocks, n_shocks)``.
    ghs2_operator : TensorVariable
        The matrix :math:`G` mapping :math:`\operatorname{vec}(\Sigma)` to the risk correction, of shape
        ``(n_variables, n_shocks ** 2)``.
    """

    ghxx: TensorVariable
    ghxu: TensorVariable
    ghuu: TensorVariable
    ghs2_operator: TensorVariable


def solve_second_order(
    jacobians: list[TensorVariable],
    composition_hessian: TensorVariable,
    ghx: TensorVariable,
    ghu: TensorVariable,
    dr_order: DROrder,
    method: AdjointMethod = "doubling",
) -> SecondOrderTerms:
    r"""
    Build the second-order perturbation solution as a graph.

    Parameters
    ----------
    jacobians : list of TensorVariable
        Jacobians ``[A, B, C, D]`` of the model with respect to the variables at t-1, t and t+1 and the shocks, rows
        in ``dr_order.eq_order`` and variable columns in ``dr_order.var_order``, as returned by
        :meth:`~gEconpy.model.model.Model.symbolic_linearization`.
    composition_hessian : TensorVariable
        Hessian of the model composed with the first-order policy, from
        :func:`~gEconpy.model.perturbation.composition_derivatives`, with the same log-linearization as
        ``jacobians``.
    ghx : TensorVariable
        First-order response of every variable to the lagged states, of shape ``(n_variables, n_states)``, rows in
        decision-rule order.
    ghu : TensorVariable
        First-order response of every variable to the shocks, of shape ``(n_variables, n_shocks)``, rows in
        decision-rule order.
    dr_order : DROrder
        Decision-rule ordering of the model.
    method : {"doubling", "qz", "sylvester", "direct"}, optional
        Method for the generalized Sylvester equation that determines ``ghxx``, as in
        :func:`~gEconpy.solvers.stein.solve_generalized_sylvester`. Defaults to ``"doubling"``.

    Returns
    -------
    terms : SecondOrderTerms
        Second-order coefficients of the decision rule.
    """
    _, jacobian_now, jacobian_lead, _ = jacobians
    lags, leads = dr_order.lag_positions, dr_order.lead_positions
    n_variables = jacobian_now.shape[0]
    n_states, n_shocks = len(lags), ghu.shape[1]

    ghx_states, ghu_states = ghx[lags], ghu[lags]
    lead_block = jacobian_lead[:, leads]

    # A acts on second-order terms that enter at t, B on those that enter through the policy at t+1.
    A = pt.inc_subtensor(jacobian_now[:, lags], lead_block @ ghx[leads])
    B = pt.set_subtensor(pt.zeros((n_variables, n_variables))[:, leads], lead_block)

    states = slice(0, n_states)
    shocks = slice(n_states, n_states + n_shocks)
    next_shocks = slice(n_states + n_shocks, None)

    hessian_xx = composition_hessian[:, states, states]
    ghxx = solve_generalized_sylvester(
        A=A, B=B, K=pt.linalg.kron(ghx_states, ghx_states), D=-_unfold(hessian_xx), method=method
    ).reshape(hessian_xx.shape)

    B_ghxx = pt.tensordot(B, ghxx, axes=1)
    rhs_xu = composition_hessian[:, states, shocks] + pt.einsum("nab,ai,bj->nij", B_ghxx, ghx_states, ghu_states)
    ghxu = pt.linalg.solve(A, -_unfold(rhs_xu)).reshape(rhs_xu.shape)

    rhs_uu = composition_hessian[:, shocks, shocks] + pt.einsum("nab,ai,bj->nij", B_ghxx, ghu_states, ghu_states)
    ghuu = pt.linalg.solve(A, -_unfold(rhs_uu)).reshape(rhs_uu.shape)

    # The risk correction solves (A + B) ghs2 = -(B ghuu + H_u'u') vec(Sigma), which is linear in vec(Sigma).
    rhs_risk = pt.tensordot(B, ghuu, axes=1) + composition_hessian[:, next_shocks, next_shocks]
    ghs2_operator = pt.linalg.solve(A + B, -_unfold(rhs_risk))

    return SecondOrderTerms(ghxx=ghxx, ghxu=ghxu, ghuu=ghuu, ghs2_operator=ghs2_operator)


def _unfold(folded: TensorVariable) -> TensorVariable:
    """Merge the two trailing axes of a ``(n, p, q)`` tensor into Kronecker columns ``i * q + j``."""
    return folded.reshape((folded.shape[0], -1))
