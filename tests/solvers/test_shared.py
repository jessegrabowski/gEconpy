import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest

from numpy.testing import assert_allclose
from pymc.pytensorf import rewrite_pregrad

from gEconpy.solvers.cycle_reduction import cycle_reduction_pt, scan_cycle_reduction
from gEconpy.solvers.gensys import gensys_pt
from gEconpy.solvers.shared import solve_with_static_elimination
from tests._resources.cache_compiled_models import load_and_cache_model


def linearized_one_block_1_ss():
    mod = load_and_cache_model("one_block_1_ss.gcn")
    matrices = mod.linearize_model(verbose=False, steady_state_kwargs={"verbose": False, "progressbar": False})
    return mod, [np.ascontiguousarray(x, dtype="float64") for x in matrices]


@pytest.mark.parametrize(
    ("solver", "mode"),
    [
        (cycle_reduction_pt, "FAST_RUN"),
        (scan_cycle_reduction, "FAST_RUN"),
        (scan_cycle_reduction, "JAX"),
        (gensys_pt, "FAST_RUN"),
    ],
    ids=["cycle_reduction", "scan_cycle_reduction", "scan_cycle_reduction-jax", "gensys"],
)
def test_static_elimination_matches_full_solve(solver, mode):
    mod, (A, B, C, D) = linearized_one_block_1_ss()

    # ``Model.linearize_model`` returns columns in model order, where the static variables are not leading, so the
    # mask also exercises the permutation back to the input column order.
    dr_order = mod.dr_order
    static_mask = np.isin(np.arange(A.shape[1]), dr_order.var_order[: dr_order.n_static_var])
    assert static_mask.any()
    assert not static_mask[: static_mask.sum()].all()

    inputs = [pt.tensor(name=name, shape=x.shape) for name, x in zip("ABCD", [A, B, C, D], strict=True)]
    T_full, R_full, *_ = solver(*inputs)
    T_reduced, R_reduced, *_ = solve_with_static_elimination(solver, *inputs, static_mask=static_mask)

    # PyMC canonicalizes the logp before differentiating it, which deletes ``disconnected_grad`` markers.
    T_reduced, R_reduced = rewrite_pregrad([T_reduced, R_reduced])

    def loss_grads(T, R):
        return pt.grad((T**2).sum() + (R**2).sum(), inputs)

    f = pytensor.function(
        inputs,
        [T_full, R_full, T_reduced, R_reduced, *loss_grads(T_full, R_full), *loss_grads(T_reduced, R_reduced)],
        mode=mode,
    )
    T_full, R_full, T_reduced, R_reduced, *grads = f(A, B, C, D)
    grads_full, grads_reduced = grads[:4], grads[4:]

    assert_allclose(T_reduced, T_full, atol=1e-10)
    assert_allclose(R_reduced, R_full, atol=1e-10)

    # The static columns of A and C are structural zeros that the reduced solve never reads, so only their gradients
    # are allowed to differ.
    dynamic = ~static_mask
    for name, grad_full, grad_reduced in zip("ABCD", grads_full, grads_reduced, strict=True):
        columns = dynamic if name in "AC" else slice(None)
        assert_allclose(grad_reduced[:, columns], grad_full[:, columns], atol=1e-8, err_msg=name)


def test_static_elimination_returns_nan_when_the_static_block_is_singular():
    mod, (A, B, C, D) = linearized_one_block_1_ss()
    dr_order = mod.dr_order
    static_mask = np.isin(np.arange(A.shape[1]), dr_order.var_order[: dr_order.n_static_var])
    first_static, second_static, *_ = np.flatnonzero(static_mask)
    B[:, second_static] = B[:, first_static]

    inputs = [pt.tensor(name=name, shape=x.shape) for name, x in zip("ABCD", [A, B, C, D], strict=True)]
    T, R = solve_with_static_elimination(cycle_reduction_pt, *inputs, static_mask=static_mask)
    T, R = pytensor.function(inputs, [T, R], mode="FAST_RUN")(A, B, C, D)

    assert np.isnan(T).all()
    assert np.isnan(R).all()


def test_static_elimination_rejects_a_model_with_no_dynamic_variables():
    A, B, C, D = (pt.dmatrix(name) for name in "ABCD")

    with pytest.raises(ValueError, match="Every variable is static"):
        solve_with_static_elimination(cycle_reduction_pt, A, B, C, D, static_mask=np.ones(3, dtype=bool))
