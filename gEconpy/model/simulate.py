from __future__ import annotations

import logging

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import xarray as xr

from gEconpy.model.statistics import _maybe_solve_model, _validate_shock_options, build_Q_matrix

if TYPE_CHECKING:
    from gEconpy.model.model import Model

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ShockSpec:
    """Representation of a shock input used to generate an impulse response function."""

    mode: str  # one of "trajectory",  "cov", or "size"
    trajectory: np.ndarray | None
    cov: np.ndarray | None
    size: float | np.ndarray | dict[str, float] | None
    orthogonalize: bool


def _validate_irf_shock_arguments(*values_with_names: tuple[str, Any]) -> None:
    """Ensure at most one of the provided options is non-None."""
    provided_names, _provided_values = zip(*[(n, v) for n, v in values_with_names if v is not None], strict=False)
    if len(provided_names) > 1:
        names = ", ".join(n for n, _ in provided_names)
        raise ValueError(f"Only one of {names} may be specified, got {len(provided_names)}.")


def _is_diagonal(M: np.ndarray) -> bool:
    return np.allclose(M, np.diag(np.diag(M)))


def _get_selected_shock_names(spec: ShockSpec, shock_names: list[str]) -> list[str]:
    """
    If user passed a dict for shock_size, return only those shock names (in model order).

    Otherwise, return None to indicate all shocks should be used.
    """
    if spec.mode == "size" and isinstance(spec.size, dict):
        if len(spec.size) == 0:
            raise ValueError("Shock size cannot be empty.")

        unknown_shocks = set(spec.size) - set(shock_names)
        if unknown_shocks:
            raise ValueError(f"shock_size dict contains unknown shock names: {unknown_shocks}")

        return [name for name in shock_names if name in spec.size]

    return shock_names


def _infer_shocks_are_individual(
    requested: bool | None,
    shock_spec: ShockSpec,
    n_shocks: int,
) -> bool:
    if requested is not None:
        return requested

    if shock_spec.mode == "size":
        if isinstance(shock_spec.size, int | float | dict):
            return True

        arr = np.asarray(shock_spec.size)
        return arr.ndim == 0 or arr.shape == (n_shocks,)

    if shock_spec.mode == "cov":
        return _is_diagonal(np.asarray(shock_spec.cov))

    if shock_spec.mode == "trajectory":
        return False

    return False


def _make_shock_spec(
    shock_size: float | np.ndarray | dict[str, float] | None,
    shock_cov: np.ndarray | None,
    shock_trajectory: np.ndarray | None,
    orthogonalize_shocks: bool,
) -> ShockSpec:
    _validate_irf_shock_arguments(
        ("shock_size", shock_size),
        ("shock_cov", shock_cov),
        ("shock_trajectory", shock_trajectory),
    )

    mode = "trajectory" if shock_trajectory is not None else "cov" if shock_cov is not None else "size"
    return ShockSpec(
        mode=mode, trajectory=shock_trajectory, cov=shock_cov, size=shock_size, orthogonalize=orthogonalize_shocks
    )


def _shock_vector_from_spec(
    size: float | np.ndarray | dict[str, float] | None,
    shock_names: Sequence[str],
) -> np.ndarray:
    """Return an (n_shocks,) vector for a one-period step size."""
    n = len(shock_names)
    if size is None:
        return np.ones(n)
    if isinstance(size, int | float):
        return np.full(n, float(size))
    if isinstance(size, dict):
        return np.array([float(size.get(name, 0.0)) for name in shock_names], dtype=float)
    arr = np.asarray(size, dtype=float)
    if arr.shape != (n,):
        raise ValueError(f"shock_size array must have shape ({n},); got {arr.shape}.")
    return arr


def _orthogonal_factor(cov: np.ndarray, make_unit_variance: bool = False) -> np.ndarray:
    """
    Compute L such that z ~ N(0, I), e = L @ z has Cov(e)=cov.

    If make_unit_variance=True, return L' whose columns are scaled to unit variance (orthonormal shocks).
    """
    L = np.linalg.cholesky(cov)
    if not make_unit_variance:
        return L

    col_norms = np.linalg.norm(L, axis=0)
    col_norms[col_norms == 0] = 1.0
    return L / col_norms


def _build_trajectory(
    spec: ShockSpec,
    simulation_length: int,
    n_shocks: int,
    shock_names: Sequence[str],
    rng: np.random.Generator,
) -> np.ndarray:
    """Convert a ShockSpec into a (simulation_length, n_shocks) shock trajectory."""
    match spec.mode:
        case "trajectory":
            traj = np.asarray(spec.trajectory, dtype=float)
            if traj.ndim != 2 or traj.shape[1] != n_shocks:
                raise ValueError(f"shock_trajectory must have shape (T, {n_shocks}); got {traj.shape}.")

        case "cov":
            traj = np.zeros((simulation_length, n_shocks), dtype=float)
            Q = np.asarray(spec.cov, dtype=float)
            if Q.shape != (n_shocks, n_shocks):
                raise ValueError(f"shock_cov must be ({n_shocks}, {n_shocks}); got {Q.shape}.")
            L = _orthogonal_factor(Q, make_unit_variance=False) if spec.orthogonalize else np.linalg.cholesky(Q)
            e0 = rng.standard_normal(n_shocks)
            traj[0] = L @ e0

        case "size":
            traj = np.zeros((simulation_length, n_shocks), dtype=float)
            shock_size = _shock_vector_from_spec(spec.size, shock_names)
            traj[0] = shock_size

        case _:
            raise RuntimeError(f"Unexpected ShockSpec mode: {spec.mode}. You shouldn't get here, please report a bug.")

    return traj


def _simulate_linear_system(T: np.ndarray, R: np.ndarray, shock_traj: np.ndarray) -> np.ndarray:
    """Simulate a linear system :math:`x_t = T x_{t-1} + R e_t`, given a shock trajectory :math:`e_t`."""
    T = np.asarray(T)
    R = np.asarray(R)
    T_len, _n_shocks = shock_traj.shape
    n_vars = T.shape[0]

    out = np.zeros((T_len, n_vars), dtype=float)
    out[0] = R @ shock_traj[0]
    for t in range(1, T_len):
        out[t] = T @ out[t - 1] + R @ shock_traj[t]
    return out


def _irf_to_xarray(
    data: np.ndarray,
    variable_names: list[str],
    shock_names: list[str] | None,
) -> xr.DataArray:
    if shock_names is None:
        coords = {"time": np.arange(data.shape[0]), "variable": list(variable_names)}
        return xr.DataArray(data, dims=["time", "variable"], coords=coords)
    coords = {
        "shock": list(shock_names),
        "time": np.arange(data.shape[1]),
        "variable": list(variable_names),
    }
    return xr.DataArray(data, dims=["shock", "time", "variable"], coords=coords)


def impulse_response_function(
    model: Model,
    T: np.ndarray | None = None,
    R: np.ndarray | None = None,
    simulation_length: int = 40,
    shock_size: float | np.ndarray | dict[str, float] | None = None,
    shock_cov: np.ndarray | None = None,
    shock_trajectory: np.ndarray | None = None,
    return_individual_shocks: bool | None = None,
    orthogonalize_shocks: bool = False,
    random_seed: int | np.random.RandomState | None = None,
    **solve_model_kwargs,
) -> xr.DataArray:
    """
    Generate impulse response functions (IRF) from state space model dynamics.

    An impulse response function represents the dynamic response of the state space model
    to an instantaneous shock applied to the system. This function calculates the IRF
    based on either provided shock specifications or the posterior state covariance matrix.

    Parameters
    ----------
    model: Model
        DSGE Model object
    T: np.ndarray, optional
        Transition matrix of the solved system. If None, this will be computed using the model's ``solve_model``
        method.
    R: np.ndarray, optional
        Selection matrix of the solved system. If None, this will be computed using the model's ``solve_model`` method.
    simulation_length : int, optional
        The number of periods to compute the IRFs over. The default is 40.
    shock_size : float, array, or dict; default=None
        The size of the shock applied to the system. If specified, it will create a covariance
        matrix for the shock with diagonal elements equal to `shock_size`:

            - If float, the covariance matrix will be the identity matrix scaled by `shock_size`.
            - If array, the covariance matrix will be ``diag(shock_size)``. In this case, the length of the
              provided array must match the number of shocks in the state space model.
            - If dictionary, a diagonal matrix will be created with entries corresponding to the keys in the
              dictionary. Shocks that are not specified will be set to zero.

        Only one of `use_stationary_cov`, `shock_cov`, `shock_size`, or `shock_trajectory` can be specified.
    shock_cov : Optional[np.ndarray], default=None
        A user-specified covariance matrix for the shocks. It should be a 2D numpy array with
        dimensions (n_shocks, n_shocks), where n_shocks is the number of shocks in the state space model.

        Only one of `use_stationary_cov`, `shock_cov`, `shock_size`, or `shock_trajectory` can be specified.
    shock_trajectory : Optional[np.ndarray], default=None
        A pre-defined trajectory of shocks applied to the system. It should be a 2D numpy array
        with dimensions (n, n_shocks), where n is the number of time steps and k_posdef is the
        number of shocks in the state space model.

        Only one of `use_stationary_cov`, `shock_cov`, `shock_size`, or `shock_trajectory` can be specified.
    return_individual_shocks: bool, optional
        If True, an IRF will be computed separately for each shock in the model. An additional dimension will be added
        to the output DataArray to show each shock. This is only valid if `shock_size` is a scalar, dictionary, or if
        the covariance matrix is diagonal.

        If not specified, this will be set to True if ``shock_size`` if the above conditions are met.
    orthogonalize_shocks : bool, default=False
        If True, orthogonalize the shocks using Cholesky decomposition when generating the impulse
        response. This option is ignored if `shock_trajectory` or `shock_size` are used, or if the covariance matrix is
        diagonal.
    random_seed : int, RandomState or Generator, optional
        Seed for the random number generator.
    **solve_model_kwargs
        Arguments forwarded to the ``solve_model`` method. Ignored if T and R are provided.

    Returns
    -------
    xr.DataArray
        The IRFs for each variable in the model.
    """
    rng = np.random.default_rng(random_seed)

    T, R = _maybe_solve_model(model, T, R, **solve_model_kwargs)

    spec = _make_shock_spec(shock_size, shock_cov, shock_trajectory, orthogonalize_shocks)

    variable_names = [x.base_name for x in model.variables]
    shock_names = [x.base_name for x in model.shocks]
    selected_shock_names = _get_selected_shock_names(spec, [x.base_name for x in model.shocks])

    n_vars = len(variable_names)
    n_shocks = len(model.shocks)
    n_selected_shocks = len(selected_shock_names)

    shock_idxs = [i for i, x in enumerate(model.shocks) if x.base_name in selected_shock_names]

    if spec.mode == "trajectory":
        simulation_length = spec.trajectory.shape[0]

    apply_shocks_individually = _infer_shocks_are_individual(return_individual_shocks, spec, n_selected_shocks)

    if apply_shocks_individually:
        data = np.zeros((n_selected_shocks, simulation_length, n_vars), dtype=float)

        if spec.mode == "trajectory":
            full = np.asarray(spec.trajectory, dtype=float)
            if full.shape[1] != n_shocks:
                raise ValueError(f"shock_trajectory must have n_shocks={n_shocks}.")
            for i, idx in enumerate(shock_idxs):
                traj = np.zeros_like(full)
                traj[:, idx] = full[:, idx]
                data[i] = _simulate_linear_system(T, R, traj)
        else:
            base = _build_trajectory(spec, simulation_length, n_shocks, shock_names, rng)
            for i, idx in enumerate(shock_idxs):
                traj = np.zeros_like(base)
                traj[:, idx] = base[:, idx]
                data[i] = _simulate_linear_system(T, R, traj)

        return _irf_to_xarray(data, variable_names, shock_names=selected_shock_names)

    traj = _build_trajectory(spec, simulation_length, n_shocks, selected_shock_names, rng)
    data = _simulate_linear_system(T, R, traj)
    return _irf_to_xarray(data, variable_names, shock_names=None)


def simulate(
    model: Model,
    T: np.ndarray | None = None,
    R: np.ndarray | None = None,
    n_simulations: int = 1,
    simulation_length: int = 40,
    shock_std_dict: dict[str, float] | None = None,
    shock_cov_matrix: np.ndarray | None = None,
    shock_std: np.ndarray | list | float | np.ndarray = None,
    random_seed: int | np.random.RandomState | None = None,
    **solve_model_kwargs,
) -> xr.DataArray:
    """
    Simulate the model over a certain number of time periods.

    Parameters
    ----------
    model: Model
        DSGE Model object
    T: np.ndarray, optional
        Transition matrix of the solved system. If None, this will be computed using the model's ``solve_model``
        method. Ignored if ``use_param_priors`` is True.
    R: np.ndarray, optional
        Selection matrix of the solved system. If None, this will be computed using the model's ``solve_model`` method.
        Ignored if ``use_param_priors`` is True.
    use_param_priors: bool, optional
        If True, each simulation will be generated using a different random draw from the model's
        prior distributions. Default is False, in which case a fixed T and R matrix will be used for each simulation.
        If True, T and R are ignored.
    n_simulations : int, optional
        Number of trajectories to simulate. Default is 1.
    simulation_length : int, optional
        Length of each simulated trajectory. Default is 40.
    shock_std_dict: dict, optional
        Dictionary of shock names and standard deviations to be used to build Q
    shock_cov_matrix: array, optional
        An (n_shocks, n_shocks) covariance matrix describing the exogenous shocks
    shock_std: float or sequence, optional
        Standard deviation of all model shocks.
    random_seed : int, RandomState or Generator, optional
        Seed for the random number generator.
    **solve_model_kwargs
        Arguments forwarded to the ``solve_model`` method. Ignored if T and R are provided.

    Returns
    -------
    xr.DataArray
        Simulated trajectories.
    """
    rng = np.random.default_rng(random_seed)

    shocks = model.shocks
    n_shocks = len(shocks)

    _validate_shock_options(
        shock_std_dict=shock_std_dict,
        shock_cov_matrix=shock_cov_matrix,
        shock_std=shock_std,
        shocks=shocks,
    )

    Q = build_Q_matrix(
        model_shocks=shocks,
        shock_std_dict=shock_std_dict,
        shock_cov_matrix=shock_cov_matrix,
        shock_std=shock_std,
    )

    epsilons = rng.multivariate_normal(
        mean=np.zeros(n_shocks),
        cov=Q,
        size=(n_simulations, simulation_length),
        method="svd",
    )

    data = np.zeros((n_simulations, simulation_length, len(model.variables)))
    T, R = _maybe_solve_model(model, T, R, **solve_model_kwargs)

    data[:, 0, :] = np.einsum("nk,sk->sn", R, epsilons[:, 0, :])
    for t in range(1, simulation_length):
        stochastic = np.einsum("nk,sk->sn", R, epsilons[:, t, :])
        deterministic = np.einsum("nm,sm->sn", T, data[:, t - 1, :])
        data[:, t, :] = deterministic + stochastic

    return xr.DataArray(
        data,
        dims=["simulation", "time", "variable"],
        coords={
            "variable": [x.base_name for x in model.variables],
            "simulation": np.arange(n_simulations),
            "time": np.arange(simulation_length),
        },
    )
