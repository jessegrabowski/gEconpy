import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest

from pytensor.compile.mode import get_mode
from pytensor.graph.basic import equal_computations
from pytensor.ifelse import ifelse
from pytensor.scan.op import Scan

from gEconpy.pytensorf.scan_rewrites import SCAN_UNTIL_TAG, scan_until_converged
from gEconpy.solvers.cycle_reduction import scan_cycle_reduction
from tests._resources.cache_compiled_models import load_and_cache_model
from tests._resources.rewrite_tester import RewriteTester


def scan_until_tester(inputs, outputs, custom_rewrite=scan_until_converged):
    # The rewrite only matches scans whose states are untraced, which the Numba passes before its position establish.
    return RewriteTester(
        inputs,
        outputs,
        include=["fast_run", "numba"],
        exclude=[SCAN_UNTIL_TAG, *get_mode("NUMBA").linker.incompatible_rewrites],
        position_cutoff=1.625,
        custom_rewrite=custom_rewrite,
        linker="numba",
    )


def scan_ops(fgraph):
    return [node.op for node in fgraph.toposort() if isinstance(node.op, Scan)]


def assert_rewrite_is_a_no_op(inputs, outputs):
    rewritten = scan_until_tester(inputs, outputs).rewr_fg
    prepared = scan_until_tester(inputs, outputs, custom_rewrite=None).rewr_fg
    assert equal_computations(rewritten.outputs, prepared.outputs, in_xs=rewritten.inputs, in_ys=prepared.inputs)


@pytest.fixture(scope="module")
def system():
    mod = load_and_cache_model("one_block_1_ss.gcn")
    return [
        np.ascontiguousarray(x, dtype="float64")
        for x in mod.linearize_model(verbose=False, steady_state_kwargs={"verbose": False, "progressbar": False})
    ]


def counting_scan(converged_step):
    x0 = pt.dscalar("x0")
    trace = pytensor.scan(
        lambda x: ifelse(x > 5.0, converged_step(x), x + 1.0), outputs_info=[x0], n_steps=10, return_updates=False
    )
    return x0, trace[-1]


def test_until_condition_is_the_convergence_test_on_the_step_outputs():
    x0, final = counting_scan(lambda x: x)
    tester = scan_until_tester([x0], [final])

    (scan_op,) = scan_ops(tester.rewr_fg)
    inner_outputs = scan_op.fgraph.outputs
    assert scan_op.info.as_while
    assert equal_computations([inner_outputs[-1]], [pt.gt(inner_outputs[0], 5.0)])
    tester.assert_eval(0.0)
    tester.assert_eval(9.0)


def test_converged_branch_that_changes_state_is_left_alone():
    x0, final = counting_scan(lambda x: 2.0 * x)
    assert_rewrite_is_a_no_op([x0], [final])


@pytest.mark.parametrize("tol", [1e-7, 1e10], ids=["converges_mid_loop", "converged_at_start"])
def test_cycle_reduction_scan_stops_with_unchanged_result(system, tol):
    symbols = [pt.dmatrix(name) for name in "ABCD"]
    T, _, n_steps = scan_cycle_reduction(*symbols, max_iter=50, tol=tol)
    tester = scan_until_tester(symbols, [T, n_steps])

    assert [op.info.as_while for op in scan_ops(tester.rewr_fg)] == [True]
    tester.assert_eval(*system)


def test_backprop_through_scan_is_left_alone():
    symbols = [pt.dmatrix(name) for name in "ABCD"]
    T, _, _ = scan_cycle_reduction(*symbols, max_iter=50, use_adjoint_gradients=False)
    assert_rewrite_is_a_no_op(symbols, pt.grad(T.sum(), symbols[:3]))


def test_adjoint_stops_early_for_the_gradient_but_not_for_the_hessian(system):
    symbols = [pt.dmatrix(name) for name in "ABCD"]
    T, _, _ = scan_cycle_reduction(*symbols, max_iter=50)
    grads = pt.grad(T.sum(), symbols[:3])
    directions = [pt.dmatrix(f"p_{name}") for name in "ABC"]
    hvp = pt.grad(sum((g * p).sum() for g, p in zip(grads, directions, strict=True)), symbols[:3])
    tester = scan_until_tester([*symbols, *directions], [*grads, *hvp])

    while_flags = [op.info.as_while for op in scan_ops(tester.rewr_fg)]
    assert any(while_flags)
    assert not all(while_flags)
    tester.assert_eval(*system, *((x != 0).astype(float) for x in system[:3]))


def test_numba_mode_applies_the_rewrite():
    x0, final = counting_scan(lambda x: x)
    fn = pytensor.function([x0], final, mode="NUMBA")

    assert [op.info.as_while for op in scan_ops(fn.maker.fgraph)] == [True]
    assert fn(0.0) == 6.0
