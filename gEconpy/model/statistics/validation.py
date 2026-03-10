from __future__ import annotations

import logging

from typing import TYPE_CHECKING, cast

import numpy as np

from gEconpy.classes.containers import SteadyStateResults
from gEconpy.classes.time_aware_symbol import TimeAwareSymbol
from gEconpy.exceptions import SteadyStateNotFoundError

if TYPE_CHECKING:
    from gEconpy.model.model import Model

_log = logging.getLogger(__name__)


def _maybe_solve_steady_state(
    model: Model,
    steady_state: dict | None,
    steady_state_kwargs: dict | None,
    parameter_updates: dict | None,
):
    if parameter_updates is None:
        parameter_updates = {}
    if steady_state is None:
        if model.is_linear:
            return model.f_ss(**model.parameters(**parameter_updates))

        return model.steady_state(**model.parameters(**parameter_updates), **steady_state_kwargs)

    param_dict = model.parameters(**parameter_updates)
    ss_resid = model.evaluate_residual(steady_state, param_dict)
    FLOAT_ZERO = 1e-8
    unsatisfied_flags = np.abs(ss_resid) > FLOAT_ZERO
    unsatisfied_eqs = [f"Equation {i}" for i, flag in enumerate(unsatisfied_flags) if flag]

    if np.any(unsatisfied_flags):
        raise SteadyStateNotFoundError(unsatisfied_eqs)
    steady_state.success = True

    return steady_state


def _maybe_linearize_model(
    model: Model,
    A: np.ndarray | None,
    B: np.ndarray | None,
    C: np.ndarray | None,
    D: np.ndarray | None,
    **linearize_model_kwargs,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Linearize a model if required, or return the provided matrices.

    Parameters
    ----------
    model: Model
        DSGE model
    A: np.ndarray, optional
        Jacobian w.r.t. variables at time t-1.
    B: np.ndarray, optional
        Jacobian w.r.t. variables at time t.
    C: np.ndarray, optional
        Jacobian w.r.t. variables at time t+1.
    D: np.ndarray, optional
        Jacobian w.r.t. stochastic innovations.
    linearize_model_kwargs
        Arguments forwarded to ``model.linearize_model``. Ignored if all of A, B, C, D are provided.

    Returns
    -------
    linear_system : tuple of ndarray
    """
    verbose = linearize_model_kwargs.get("verbose", True)
    n_matrices = sum(x is not None for x in [A, B, C, D])

    if n_matrices < 4 and n_matrices > 0 and verbose:
        _log.warning(
            f"Passing an incomplete subset of A, B, C, and D (you passed {n_matrices}) will still trigger "
            f"``model.linearize_model`` (which might be expensive). Pass all to avoid this, or None to silence "
            f"this warning."
        )
        A = None
        B = None
        C = None
        D = None

    if all(x is None for x in [A, B, C, D]):
        A, B, C, D = model.linearize_model(**linearize_model_kwargs)

    return A, B, C, D


def _maybe_solve_model(
    model: Model, T: np.ndarray | None, R: np.ndarray | None, **solve_model_kwargs
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """
    Solve for the linearized policy matrix of a model if required, or return the provided T and R.

    Parameters
    ----------
    model: Model
        DSGE Model associated with T and R.
    T: np.ndarray, optional
        Transition matrix.
    R: np.ndarray, optional
        Selection matrix.
    **solve_model_kwargs
        Arguments forwarded to ``solve_model``. Ignored if T and R are provided.

    Returns
    -------
    T, R : tuple of ndarray or tuple of None
    """
    n_matrices = sum(x is not None for x in [T, R])
    if n_matrices == 1:
        _log.warning(
            "Passing only one of T or R will still trigger ``model.solve_model`` (which might be expensive). "
            "Pass both to avoid this, or None to silence this warning."
        )
        T = None
        R = None

    if T is None and R is None:
        T, R = model.solve_model(**solve_model_kwargs)

    return T, R


def _validate_shock_options(
    shock_std_dict: dict[str, float] | None,
    shock_cov_matrix: np.ndarray | None,
    shock_std: float | np.ndarray | list | None,
    shocks: list[TimeAwareSymbol],
):
    n_shocks = len(shocks)
    n_provided = sum(x is not None for x in [shock_std_dict, shock_cov_matrix, shock_std])
    if n_provided > 1 or n_provided == 0:
        raise ValueError(
            "Exactly one of shock_std_dict, shock_cov_matrix, or shock_std should be provided. You passed "
            f"{n_provided}."
        )

    if shock_cov_matrix is not None and any(s != n_shocks for s in shock_cov_matrix.shape):
        raise ValueError(
            f"Incorrect covariance matrix shape. Expected ({n_shocks}, {n_shocks}), found {shock_cov_matrix.shape}"
        )

    if shock_std_dict is not None:
        shock_names = [x.base_name for x in shocks]
        missing = [x for x in shock_std_dict if x not in shock_names]
        extra = [x for x in shock_names if x not in shock_std_dict]
        if len(missing) > 0:
            raise ValueError(
                f"If shock_std_dict is specified, it must give values for all shocks. The following shocks were not "
                f"found among the provided keys: {', '.join(missing)}"
            )
        if len(extra) > 0:
            raise ValueError(
                f"Unexpected shocks in shock_std_dict. The following names were not found among the model shocks: "
                f"{', '.join(extra)}"
            )

    if shock_std is not None:
        if isinstance(shock_std, np.ndarray | list):
            shock_std = cast(np.ndarray | list, shock_std)
            if len(shock_std) != n_shocks:
                raise ValueError(
                    f"Length of shock_std ({len(shock_std)}) does not match the number of shocks ({n_shocks})"
                )
            if not np.all(shock_std > 0):
                raise ValueError("Shock standard deviations must be positive")
        elif isinstance(shock_std, int | float):
            if shock_std < 0:
                raise ValueError("Shock standard deviation must be positive")


def _validate_simulation_options(shock_size, shock_cov, shock_trajectory) -> None:
    options = [shock_size, shock_cov, shock_trajectory]
    n_options = sum(x is not None for x in options)

    if n_options != 1:
        raise ValueError("Specify exactly 1 of shock_size, shock_cov, or shock_trajectory")


def check_steady_state(
    model: Model,
    stead_state: SteadyStateResults | None = None,
    steady_state_kwargs: dict | None = None,
    **parameter_updates,
) -> None:
    if steady_state_kwargs is None:
        steady_state_kwargs = {}

    ss_dict = _maybe_solve_steady_state(model, stead_state, steady_state_kwargs, parameter_updates)
    if ss_dict.success:
        _log.warning("Steady state successfully found!")
        return

    parameters = model.parameters(**parameter_updates)
    residuals = model.evaluate_residual(ss_dict, parameters)
    _log.warning("Steady state NOT successful. The following equations have non-zero residuals:")

    FLOAT_ZERO = 1e-8
    for resid, eq in zip(residuals, model.equations, strict=False):
        if np.abs(resid) > FLOAT_ZERO:
            _log.warning(eq)
            _log.warning(f"Residual: {resid:0.4f}")
