from functools import cache

import pytensor
import pytensor.tensor as pt

from pytensor.compile.executor import Function
from pytensor.tensor import TensorLike, TensorVariable


def pruned_second_order_path(
    ghx: TensorLike,
    ghu: TensorLike,
    ghxx: TensorLike,
    ghxu: TensorLike,
    ghuu: TensorLike,
    ghs2: TensorLike,
    shocks: TensorLike,
    state_positions: TensorLike,
) -> TensorVariable:
    r"""
    Build the pruned second-order simulation of a perturbation solution as a graph.

    The path starts from the steady state and splits each period's deviation into a first-order part
    :math:`\hat{y}^{(1)}` and a second-order part :math:`\hat{y}^{(2)}`, with the quadratic terms evaluated on the
    first-order part only:

    .. math::

        \hat{y}^{(1)}_t &= g_x \hat{y}^{(1)s}_{t-1} + g_u \varepsilon_t \\
        \hat{y}^{(2)}_t &= g_x \hat{y}^{(2)s}_{t-1} + \tfrac{1}{2} g_{\sigma\sigma} + \tfrac{1}{2} g_{xx}
        (\hat{y}^{(1)s}_{t-1} \otimes \hat{y}^{(1)s}_{t-1}) + g_{xu} (\hat{y}^{(1)s}_{t-1} \otimes \varepsilon_t)
        + \tfrac{1}{2} g_{uu} (\varepsilon_t \otimes \varepsilon_t),

    where :math:`s` selects the state variables. This is the pruning scheme of Andreasen, Fernandez-Villaverde and
    Rubio-Ramirez (2018).

    Parameters
    ----------
    ghx : TensorVariable
        First-order response to the lagged states, of shape ``(n_variables, n_states)``, rows in decision-rule order.
    ghu : TensorVariable
        First-order response to the shocks, of shape ``(n_variables, n_shocks)``, rows in decision-rule order.
    ghxx : TensorVariable
        Second derivative with respect to the states, of shape ``(n_variables, n_states, n_states)``.
    ghxu : TensorVariable
        Cross derivative with respect to the states and the shocks, of shape ``(n_variables, n_states, n_shocks)``.
    ghuu : TensorVariable
        Second derivative with respect to the shocks, of shape ``(n_variables, n_shocks, n_shocks)``.
    ghs2 : TensorVariable
        Risk correction of every variable, of shape ``(n_variables,)``.
    shocks : TensorVariable
        Shock path, of shape ``(..., n_periods, n_shocks)``. Leading dimensions are simulated independently.
    state_positions : TensorVariable
        Positions of the state variables in decision-rule order.

    Returns
    -------
    deviations : TensorVariable
        Deviation of every variable from the steady state, of shape ``(..., n_periods, n_variables)``, in
        decision-rule order.
    """
    ghx, ghu, ghxx, ghxu, ghuu, ghs2, shocks = map(pt.as_tensor, (ghx, ghu, ghxx, ghxu, ghuu, ghs2, shocks))
    state_positions = pt.as_tensor(state_positions)

    def step(shock, first_order_states, second_order_states):
        first_order = first_order_states @ ghx.T + shock @ ghu.T
        second_order = (
            second_order_states @ ghx.T
            + 0.5 * ghs2
            + 0.5 * _quadratic_form(ghxx, first_order_states, first_order_states)
            + _quadratic_form(ghxu, first_order_states, shock)
            + 0.5 * _quadratic_form(ghuu, shock, shock)
        )
        return first_order[..., state_positions], second_order[..., state_positions], first_order + second_order

    initial_states = pt.zeros((*shocks.shape[:-2], ghx.shape[1]))
    _, _, deviations = pytensor.scan(
        step,
        sequences=[pt.moveaxis(shocks, -2, 0)],
        outputs_info=[initial_states, initial_states, None],
        return_updates=False,
    )
    return pt.moveaxis(deviations, 0, -2)


@cache
def compile_pruned_second_order_path() -> Function:
    """
    Compile :func:`pruned_second_order_path` once, for solutions of any model.

    The compiled function takes the arguments of :func:`pruned_second_order_path` in order, with ``shocks`` of shape
    ``(n_paths, n_periods, n_shocks)``.
    """
    inputs = [
        pt.matrix("ghx"),
        pt.matrix("ghu"),
        pt.tensor3("ghxx"),
        pt.tensor3("ghxu"),
        pt.tensor3("ghuu"),
        pt.vector("ghs2"),
        pt.tensor3("shocks"),
        pt.lvector("state_positions"),
    ]
    return pytensor.function(inputs, pruned_second_order_path(*inputs))


def _quadratic_form(coefficients: TensorVariable, left: TensorVariable, right: TensorVariable) -> TensorVariable:
    """Contract ``(n, p, q)`` coefficients with ``(..., p)`` and ``(..., q)`` vectors into ``(..., n)``."""
    return (pt.tensordot(left, coefficients, axes=[[-1], [1]]) * right[..., None, :]).sum(axis=-1)
