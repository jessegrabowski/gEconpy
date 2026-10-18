import multiprocessing
import re

import numpy as np
import pytest
import xarray as xr

from numpy.testing import assert_allclose

from gEconpy.model.perfect_foresight import sample_perfect_foresight, solve_perfect_foresight
from tests._resources.cache_compiled_models import load_and_cache_model


@pytest.fixture
def rbc_model():
    return load_and_cache_model("one_block_1.gcn")


def _trace(**values: list[list[float]]) -> xr.Dataset:
    """Parameter draws with one row per chain, in the layout a sampler's ``posterior`` group has."""
    n_chains, n_draws = np.shape(next(iter(values.values())))
    return xr.Dataset(
        {name: (("chain", "draw"), draws) for name, draws in values.items()},
        coords={"chain": np.arange(n_chains), "draw": np.arange(n_draws)},
    )


def test_each_draw_matches_a_solve_at_its_own_parameters(rbc_model):
    """
    Each draw ends at its own steady state, so a draw solved at the calibration would not match.

    The trace is a thinned posterior group with draw labels that do not start at zero. ``delta`` from
    ``steady_state_kwargs`` holds at every draw, and the drawn ``alpha`` overrides the one given there.
    """
    alphas = [[0.38, 0.40], [0.42, 0.44]]
    posterior = _trace(alpha=alphas).assign_coords(draw=[10, 20])
    steady_state = rbc_model.steady_state(verbose=False, progressbar=False)
    start = {"K": 0.9 * steady_state["K_ss"]}
    shock = np.zeros(50)
    shock[0] = 0.01

    paths, diagnostics = sample_perfect_foresight(
        rbc_model,
        xr.DataTree.from_dict({"posterior": posterior}),
        simulation_length=50,
        initial_conditions=start,
        shocks={"epsilon": shock},
        steady_state_kwargs={"delta": 0.025, "alpha": 0.1},
    )

    assert diagnostics["success"].all()
    assert paths["draw"].values.tolist() == [10, 20]
    for chain, draw in np.ndindex(2, 2):
        expected, _ = solve_perfect_foresight(
            rbc_model,
            simulation_length=50,
            initial_conditions=start,
            shocks={"epsilon": shock},
            steady_state_kwargs={"delta": 0.025, "alpha": alphas[chain][draw]},
        )
        for name in expected.columns:
            assert_allclose(paths[name].values[chain, draw], expected[name].to_numpy(), rtol=1e-8)


def test_a_start_with_draws_gives_each_draw_its_own(rbc_model):
    steady_state = rbc_model.steady_state(verbose=False, progressbar=False)
    shares = np.array([[0.8, 0.9]])
    start = xr.DataArray(
        shares[..., None] * steady_state["K_ss"],
        dims=("chain", "draw", "state"),
        coords={"chain": [0], "draw": [0, 1], "state": ["K"]},
    )

    paths, _ = sample_perfect_foresight(
        rbc_model, _trace(alpha=[[0.4, 0.4]]), simulation_length=50, initial_conditions=start
    )

    for draw, share in enumerate(shares[0]):
        expected, _ = solve_perfect_foresight(
            rbc_model, simulation_length=50, initial_conditions={"K": share * steady_state["K_ss"]}
        )
        assert_allclose(paths["K"].values[0, draw], expected["K"].to_numpy(), rtol=1e-8)


@pytest.mark.parametrize(
    ("betas", "sparse_root_kwargs", "match", "steady_state_success"),
    [
        ([0.99, 1.2], None, "steady state did not solve", [True, False]),
        ([0.99, 0.98], {"maxiter": 0}, "The solver reported", [True, True]),
    ],
    ids=["no_steady_state", "newton_stops"],
)
def test_a_failed_draw_is_nan_and_counted_by_cause(rbc_model, betas, sparse_root_kwargs, match, steady_state_success):
    """``beta > 1`` has no steady state, and a zero iteration limit stops Newton before it converges."""
    steady_state = rbc_model.steady_state(verbose=False, progressbar=False)
    with pytest.warns(UserWarning, match=f"draws failed to solve.*{match}"):
        paths, diagnostics = sample_perfect_foresight(
            rbc_model,
            _trace(beta=[betas]),
            simulation_length=20,
            initial_conditions={"K": 0.9 * steady_state["K_ss"]},
            sparse_root_kwargs=sparse_root_kwargs,
        )

    assert not diagnostics["success"].values[0, 1]
    assert diagnostics["steady_state_success"].values[0].tolist() == steady_state_success
    assert np.isnan(paths["K"].values[0, 1]).all()


def _per_draw_start(values: list[float], draws: list[int]) -> xr.DataArray:
    return xr.DataArray(
        np.array(values)[None, :, None],
        dims=("chain", "draw", "state"),
        coords={"chain": [0], "draw": draws, "state": ["K"]},
    )


@pytest.mark.parametrize(
    ("trace", "kwargs", "match"),
    [
        (_trace(alpha=[[0.4]]), {"initial_conditions": {"Z": 1.0}}, "Unknown variables in initial_conditions: {'Z'}"),
        (_trace(alpha=[[0.4]]), {"shocks": {"epsilon_typo": np.zeros(20)}}, "Unknown shocks: {'epsilon_typo'}"),
        (_trace(alpha=[[0.4]]), {"shocks": {"epsilon": np.full(20, np.nan)}}, "shocks contains non-finite values"),
        (
            _trace(alpha=[[0.4, 0.4]]),
            {"initial_conditions": _per_draw_start([30.0, np.nan], draws=[0, 1])},
            "chain 0, draw 1 is not finite for: K",
        ),
        (
            _trace(alpha=[[0.4, 0.4]]),
            {"initial_conditions": _per_draw_start([30.0], draws=[0])},
            "no start for these coordinates of the parameter draws: draw [1]",
        ),
        (_trace(unrelated=[[1.0, 2.0]]), {}, "none of the model's free parameters"),
    ],
    ids=[
        "unknown_variable",
        "unknown_shock",
        "non_finite_shock",
        "non_finite_start",
        "start_missing_a_draw",
        "no_free_parameter_drawn",
    ],
)
def test_inputs_it_cannot_use_raise(rbc_model, trace, kwargs, match):
    """A trace from another model would otherwise solve the calibration at every draw, with no error."""
    with pytest.raises(ValueError, match=re.escape(match)):
        sample_perfect_foresight(rbc_model, trace, simulation_length=20, **kwargs)


def test_a_deterministic_parameter_cannot_be_fixed():
    """It is rederived from the free parameters at every draw, so a value given for it would go unused."""
    model = load_and_cache_model("open_rbc.gcn")

    with pytest.raises(ValueError, match="deterministic"):
        sample_perfect_foresight(
            model, _trace(alpha=[[0.3]]), simulation_length=20, steady_state_kwargs={"rstar": 0.02}
        )


@pytest.mark.parametrize(
    "mp_ctx", [method for method in ("fork", "spawn") if method in multiprocessing.get_all_start_methods()]
)
def test_worker_processes_reproduce_the_serial_solve(rbc_model, mp_ctx):
    """One chain of three draws splits across workers, so the draws, not the chains, are the unit of work."""
    trace = _trace(alpha=[[0.38, 0.40, 0.42]])

    serial = sample_perfect_foresight(rbc_model, trace, simulation_length=30)
    parallel = sample_perfect_foresight(rbc_model, trace, simulation_length=30, cores=2, mp_ctx=mp_ctx)

    # MKL's results depend on memory alignment, so a worker's sparse LU can round differently in the last bits.
    xr.testing.assert_allclose(parallel[0], serial[0], rtol=1e-8)
    xr.testing.assert_identical(parallel[1], serial[1])
