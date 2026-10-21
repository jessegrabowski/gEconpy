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


def pruned_second_order_system(
    ghx: TensorLike,
    ghu: TensorLike,
    ghxx: TensorLike,
    ghxu: TensorLike,
    ghuu: TensorLike,
    ghs2_operator: TensorLike,
    shock_cov: TensorLike,
    state_positions: TensorLike,
) -> tuple[TensorVariable, TensorVariable, TensorVariable, TensorVariable]:
    r"""
    Build the pruned second-order simulation of a perturbation solution as a linear state-space system.

    The state stacks the two parts of the pruned deviation from :func:`pruned_second_order_path` and the distinct
    products of the first-order states, :math:`z_t = (\hat{y}^{(1)}_t, \hat{y}^{(2)}_t,
    \operatorname{vech}(\hat{y}^{(1)s}_t \hat{y}^{(1)s\top}_t))`, and follows

    .. math::

        z_t = c + A z_{t-1} + B u_t, \qquad u_t = \left(\varepsilon_t, \;
        \operatorname{vech}(\varepsilon_t \varepsilon_t^\top - \Sigma), \; \hat{y}^{(1)s}_{t-1} \otimes
        \varepsilon_t\right)

    for Gaussian shocks :math:`\varepsilon_t` with covariance :math:`\Sigma`. :math:`\operatorname{vech}` keeps the
    lower triangle of a symmetric matrix, in the order of ``numpy.tril_indices``. The innovations :math:`u_t` have
    mean zero and are uncorrelated with :math:`z_{t-1}` and across periods, although they are not independent. Their
    covariance is the unconditional one, so a Kalman filter on this system gives a quasi-likelihood. Every deviation
    :math:`\hat{y}_t = \hat{y}^{(1)}_t + \hat{y}^{(2)}_t` is a sum of two blocks of the state.

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
    ghs2_operator : TensorVariable
        Matrix of shape ``(n_variables, n_shocks ** 2)`` mapping the vectorized shock covariance to the risk
        correction.
    shock_cov : TensorVariable
        Covariance of the shocks, of shape ``(n_shocks, n_shocks)``.
    state_positions : TensorVariable
        Positions of the state variables in decision-rule order.

    Returns
    -------
    intercept : TensorVariable
        The constant :math:`c`, of length ``2 * n_variables + n_states * (n_states + 1) // 2``.
    transition : TensorVariable
        The square matrix :math:`A`, with the length of :math:`c` on each side.
    selection : TensorVariable
        The matrix :math:`B`, with the length of :math:`c` rows and ``n_shocks + n_shocks * (n_shocks + 1) // 2 +
        n_states * n_shocks`` columns.
    innovation_cov : TensorVariable
        Covariance of :math:`u_t`, square with the column count of :math:`B` on each side.
    """
    ghx, ghu, ghxx, ghxu, ghuu, ghs2_operator, shock_cov = map(
        pt.as_tensor, (ghx, ghu, ghxx, ghxu, ghuu, ghs2_operator, shock_cov)
    )
    state_positions = pt.as_tensor(state_positions)

    n_variables, n_states = ghx.shape
    n_shocks = ghu.shape[1]
    state_pairs, shock_pairs = _vech_positions(n_states), _vech_positions(n_shocks)
    n_state_pairs, n_shock_pairs = state_pairs.shape[0], shock_pairs.shape[0]
    state_transition, state_impact = ghx[state_positions], ghu[state_positions]
    vec_shock_cov = shock_cov.ravel()

    first_order_transition = ghx @ pt.eye(n_variables)[state_positions]
    kron_transition = pt.linalg.kron(state_transition, state_transition)
    kron_impact = pt.linalg.kron(state_impact, state_impact)
    # Expanding x_t (x) x_t with x_t = G x_{t-1} + H eps_t gives these coefficients on x_{t-1} (x) eps_t.
    kron_cross = (
        pt.einsum("ac,bk->abck", state_transition, state_impact)
        + pt.einsum("ak,bc->abck", state_impact, state_transition)
    ).reshape((n_states**2, n_states * n_shocks))

    pairs_transition = _merge_symmetric_columns(kron_transition, n_states)[state_pairs]
    pairs_impact = _merge_symmetric_columns(kron_impact, n_shocks)[state_pairs]
    pairs_cross = kron_cross[state_pairs]
    pairs_intercept = (kron_impact @ vec_shock_cov)[state_pairs]

    zeros = pt.zeros
    transition = pt.vertical_stack(
        pt.horizontal_stack(
            first_order_transition, zeros((n_variables, n_variables)), zeros((n_variables, n_state_pairs))
        ),
        pt.horizontal_stack(
            zeros((n_variables, n_variables)),
            first_order_transition,
            0.5 * _merge_symmetric_columns(ghxx.reshape((n_variables, -1)), n_states),
        ),
        pt.horizontal_stack(zeros((n_state_pairs, n_variables)), zeros((n_state_pairs, n_variables)), pairs_transition),
    )
    selection = pt.vertical_stack(
        pt.horizontal_stack(ghu, zeros((n_variables, n_shock_pairs)), zeros((n_variables, n_states * n_shocks))),
        pt.horizontal_stack(
            zeros((n_variables, n_shocks)),
            0.5 * _merge_symmetric_columns(ghuu.reshape((n_variables, -1)), n_shocks),
            ghxu.reshape((n_variables, -1)),
        ),
        pt.horizontal_stack(zeros((n_state_pairs, n_shocks)), pairs_impact, pairs_cross),
    )
    intercept = pt.concatenate(
        [
            zeros(n_variables),
            0.5 * (ghs2_operator + ghuu.reshape((n_variables, -1))) @ vec_shock_cov,
            pairs_intercept,
        ]
    )

    # Gaussian fourth moments give Cov(eps_i eps_j, eps_k eps_l) = Sigma_ik Sigma_jl + Sigma_il Sigma_jk.
    shock_square_cov = (
        pt.einsum("ik,jl->ijkl", shock_cov, shock_cov) + pt.einsum("il,jk->ijkl", shock_cov, shock_cov)
    ).reshape((n_shocks**2, n_shocks**2))
    state_cov = pt.linalg.solve_discrete_lyapunov(state_transition, state_impact @ shock_cov @ state_impact.T)
    innovation_cov = pt.linalg.block_diag(
        shock_cov, shock_square_cov[shock_pairs][:, shock_pairs], pt.linalg.kron(state_cov, shock_cov)
    )

    return intercept, transition, selection, innovation_cov


@cache
def compile_pruned_second_order_system() -> Function:
    """
    Compile :func:`pruned_second_order_system` once, for solutions of any model.

    The compiled function takes the arguments of :func:`pruned_second_order_system` in order.
    """
    inputs = [
        pt.matrix("ghx"),
        pt.matrix("ghu"),
        pt.tensor3("ghxx"),
        pt.tensor3("ghxu"),
        pt.tensor3("ghuu"),
        pt.matrix("ghs2_operator"),
        pt.matrix("shock_cov"),
        pt.lvector("state_positions"),
    ]
    return pytensor.function(inputs, pruned_second_order_system(*inputs))


def pruned_second_order_moments(
    ghx: TensorLike,
    ghu: TensorLike,
    ghxx: TensorLike,
    ghxu: TensorLike,
    ghuu: TensorLike,
    ghs2_operator: TensorLike,
    shock_cov: TensorLike,
    state_positions: TensorLike,
    n_lags: TensorLike,
) -> tuple[TensorVariable, TensorVariable]:
    r"""
    Compute the stationary mean and autocovariances of the pruned second-order deviations.

    The moments are those of the system from :func:`pruned_second_order_system`. The state has mean
    :math:`(I - A)^{-1} c` and covariance :math:`V` solving :math:`V = A V A^\top + B \Omega B^\top`, with
    :math:`\Omega` the innovation covariance, and the state's autocovariance at lag :math:`k` is :math:`A^k V`.

    Parameters
    ----------
    ghx, ghu, ghxx, ghxu, ghuu, ghs2_operator, shock_cov, state_positions : TensorVariable
        As in :func:`pruned_second_order_system`.
    n_lags : TensorVariable
        Number of lags, starting from lag 0.

    Returns
    -------
    mean : TensorVariable
        Stationary mean of every deviation from the steady state, of shape ``(n_variables,)``, in decision-rule
        order.
    autocovariances : TensorVariable
        :math:`\operatorname{Cov}(\hat{y}_{t+k}, \hat{y}_t)` for :math:`k` from 0 to ``n_lags - 1``, of shape
        ``(n_lags, n_variables, n_variables)``, in decision-rule order.
    """
    intercept, transition, selection, innovation_cov = pruned_second_order_system(
        ghx, ghu, ghxx, ghxu, ghuu, ghs2_operator, shock_cov, state_positions
    )
    n_variables = pt.as_tensor(ghx).shape[0]
    n_augmented = transition.shape[0]

    state_mean = pt.linalg.solve(pt.eye(n_augmented) - transition, intercept)
    state_cov = pt.linalg.solve_discrete_lyapunov(transition, selection @ innovation_cov @ selection.T)

    identity = pt.eye(n_variables)
    observe = pt.horizontal_stack(identity, identity, pt.zeros((n_variables, n_augmented - 2 * n_variables)))

    def lag_step(lagged_cov):
        return transition @ lagged_cov, observe @ lagged_cov @ observe.T

    _, autocovariances = pytensor.scan(lag_step, outputs_info=[state_cov, None], n_steps=n_lags, return_updates=False)
    return observe @ state_mean, autocovariances


@cache
def compile_pruned_second_order_moments() -> Function:
    """
    Compile :func:`pruned_second_order_moments` once, for solutions of any model.

    The compiled function takes the arguments of :func:`pruned_second_order_moments` in order.
    """
    inputs = [
        pt.matrix("ghx"),
        pt.matrix("ghu"),
        pt.tensor3("ghxx"),
        pt.tensor3("ghxu"),
        pt.tensor3("ghuu"),
        pt.matrix("ghs2_operator"),
        pt.matrix("shock_cov"),
        pt.lvector("state_positions"),
        pt.lscalar("n_lags"),
    ]
    return pytensor.function(inputs, pruned_second_order_moments(*inputs))


def _quadratic_form(coefficients: TensorVariable, left: TensorVariable, right: TensorVariable) -> TensorVariable:
    """Contract ``(n, p, q)`` coefficients with ``(..., p)`` and ``(..., q)`` vectors into ``(..., n)``."""
    return (pt.tensordot(left, coefficients, axes=[[-1], [1]]) * right[..., None, :]).sum(axis=-1)


def _vech_positions(size: TensorVariable) -> TensorVariable:
    """Positions in the row-major ``vec`` of a ``size x size`` symmetric matrix of the entries ``vech`` keeps."""
    rows, columns = pt.tril_indices(size)
    return rows * size + columns


def _merge_symmetric_columns(matrix: TensorVariable, size: TensorVariable) -> TensorVariable:
    """Multiply ``matrix`` by the duplication matrix, so it acts on ``vech(M)`` as it acted on ``vec(M)``."""
    rows, columns = pt.tril_indices(size)
    mirrored = matrix[:, columns * size + rows] * pt.neq(rows, columns)
    return matrix[:, rows * size + columns] + mirrored
