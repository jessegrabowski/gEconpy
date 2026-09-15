import re

import numpy as np
import pytest

from numpy.testing import assert_allclose

from gEconpy.model.simulate import (
    _build_trajectory,
    _get_selected_shock_names,
    _infer_shocks_are_individual,
    _make_shock_spec,
    _shock_vector_from_spec,
    _simulate_linear_system,
    _validate_irf_shock_arguments,
)

CORRELATED_COV = np.array([[0.04, 0.018], [0.018, 0.09]])
SHOCK_NAMES = ["epsilon_A", "epsilon_B"]


def _draw_period_zero_shocks(orthogonalize, n_draws=20_000):
    spec = _make_shock_spec(None, CORRELATED_COV, None, orthogonalize)
    rng = np.random.default_rng(0)
    return np.stack(
        [_build_trajectory(spec, 1, 2, ["epsilon_A", "epsilon_B"], rng)[0] for _ in range(n_draws)],
    )


@pytest.mark.parametrize("orthogonalize", [True, False], ids=["orthogonalized", "correlated"])
def test_orthogonalize_shocks_removes_correlation(orthogonalize):
    """Orthogonalized draws keep each shock's variance and lose the covariance between them."""
    draws = _draw_period_zero_shocks(orthogonalize)
    sample_cov = np.cov(draws, rowvar=False)

    expected = np.diag(np.diag(CORRELATED_COV)) if orthogonalize else CORRELATED_COV
    assert_allclose(sample_cov, expected, atol=2e-3)


@pytest.mark.parametrize(
    "spec_kwargs, expected",
    [
        ({"shock_size": None}, True),
        ({"shock_size": 0.1}, True),
        ({"shock_size": {"epsilon_A": 0.1}}, True),
        ({"shock_size": np.array([0.1, 0.2])}, True),
        ({"shock_size": np.array([0.1, 0.2, 0.3])}, False),
        ({"shock_cov": np.diag([0.04, 0.09])}, True),
        ({"shock_cov": CORRELATED_COV}, False),
        ({"shock_cov": CORRELATED_COV, "orthogonalize_shocks": True}, True),
        ({"shock_trajectory": np.zeros((5, 2))}, False),
    ],
    ids=[
        "default-unit-impulse",
        "scalar-size",
        "dict-size",
        "array-size-one-per-shock",
        "array-size-wrong-length",
        "diagonal-cov",
        "correlated-cov",
        "correlated-cov-orthogonalized",
        "trajectory",
    ],
)
def test_shocks_are_separated_by_default_only_when_independent(spec_kwargs, expected):
    spec = _make_shock_spec(
        **{
            "shock_size": None,
            "shock_cov": None,
            "shock_trajectory": None,
            "orthogonalize_shocks": False,
            **spec_kwargs,
        }
    )
    assert _infer_shocks_are_individual(None, spec, n_shocks=2) is expected


@pytest.mark.parametrize("requested", [True, False])
def test_explicit_return_individual_shocks_overrides_inference(requested):
    spec = _make_shock_spec(None, CORRELATED_COV, None, orthogonalize_shocks=False)
    assert _infer_shocks_are_individual(requested, spec, n_shocks=2) is requested


@pytest.mark.parametrize(
    "size, expected",
    [
        (None, [1.0, 1.0]),
        (0.5, [0.5, 0.5]),
        ({"epsilon_B": 0.3}, [0.0, 0.3]),
        ([0.1, 0.2], [0.1, 0.2]),
        (np.array([0.1, 0.2]), [0.1, 0.2]),
    ],
    ids=["none", "scalar", "partial-dict", "list", "array"],
)
def test_period_zero_impulse_from_each_shock_size_spelling(size, expected):
    np.testing.assert_allclose(_shock_vector_from_spec(size, SHOCK_NAMES), expected)


def test_shock_size_array_of_wrong_length_raises():
    with pytest.raises(ValueError, match=r"shock_size array must have shape \(2,\); got \(3,\)"):
        _shock_vector_from_spec(np.array([0.1, 0.2, 0.3]), SHOCK_NAMES)


@pytest.mark.parametrize(
    "spec_kwargs, match",
    [
        ({"shock_trajectory": np.zeros((5, 3))}, r"shock_trajectory must have shape \(T, 2\); got \(5, 3\)"),
        ({"shock_trajectory": np.zeros(5)}, r"shock_trajectory must have shape \(T, 2\); got \(5,\)"),
        ({"shock_cov": np.eye(3)}, r"shock_cov must be \(2, 2\); got \(3, 3\)"),
    ],
    ids=["trajectory-wrong-width", "trajectory-one-dimensional", "cov-wrong-shape"],
)
def test_mismatched_shock_shapes_raise(spec_kwargs, match):
    spec = _make_shock_spec(
        **{
            "shock_size": None,
            "shock_cov": None,
            "shock_trajectory": None,
            "orthogonalize_shocks": False,
            **spec_kwargs,
        }
    )
    with pytest.raises(ValueError, match=match):
        _build_trajectory(spec, 5, 2, SHOCK_NAMES, np.random.default_rng(0))


def test_shock_size_dict_with_unknown_shock_raises():
    spec = _make_shock_spec({"epsilon_A": 0.1, "not_a_shock": 0.1}, None, None, orthogonalize_shocks=False)
    with pytest.raises(ValueError, match=re.escape("unknown shock names: {'not_a_shock'}")):
        _get_selected_shock_names(spec, SHOCK_NAMES)


def test_shock_size_dict_selects_shocks_in_model_order():
    spec = _make_shock_spec({"epsilon_B": 0.1, "epsilon_A": 0.1}, None, None, orthogonalize_shocks=False)
    assert _get_selected_shock_names(spec, SHOCK_NAMES) == SHOCK_NAMES


def test_linear_system_recursion_matches_closed_form_with_batch_dimension():
    """A diagonal transition with a period-zero impulse decays geometrically, independently for each batch entry."""
    T = np.diag([0.5, 0.9])
    R = np.array([[1.0, 0.0], [0.0, 2.0]])
    shocks = np.zeros((3, 6, 2))
    shocks[:, 0, :] = [[1.0, 1.0], [2.0, 0.0], [0.0, -1.0]]

    states = _simulate_linear_system(T, R, shocks)

    t = np.arange(6)
    decay = np.stack([0.5**t, 0.9**t], axis=-1)
    expected = (shocks[:, 0, :] @ R.T)[:, None, :] * decay[None]
    assert_allclose(states, expected)


def test_two_irf_shock_arguments_raise_the_intended_message():
    with pytest.raises(ValueError, match="Only one of shock_size, shock_dict may be specified, got 2"):
        _validate_irf_shock_arguments(("shock_size", 0.1), ("shock_dict", {"epsilon_A": 0.1}), ("shock_cov", None))


def test_no_irf_shock_argument_passes_validation():
    _validate_irf_shock_arguments(("shock_size", None), ("shock_dict", None), ("shock_cov", None))
