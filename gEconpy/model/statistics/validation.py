import logging

from typing import TYPE_CHECKING

import numpy as np

from gEconpy.classes.containers import SteadyStateResults
from gEconpy.classes.time_aware_symbol import TimeAwareSymbol
from gEconpy.exceptions import SteadyStateNotFoundError

if TYPE_CHECKING:
    from gEconpy.model.model import Model

_log = logging.getLogger(__name__)


_FLOAT_ZERO_TOL = 1e-8


def check_steady_state(
    model: "Model",
    steady_state: SteadyStateResults | None = None,
    steady_state_kwargs: dict | None = None,
    **parameter_updates,
) -> None:
    """
    Log whether the model's steady state is satisfied, and which equations have non-zero residuals if not.

    Parameters
    ----------
    model : Model
        Model whose steady state is checked.
    steady_state : SteadyStateResults, optional
        Steady state to check. Solved from ``model`` when None. Defaults to None.
    steady_state_kwargs : dict, optional
        Keyword arguments forwarded to :meth:`~gEconpy.model.model.Model.steady_state` when ``steady_state`` is
        solved here. Defaults to None.
    **parameter_updates
        Parameter values overriding the model defaults.

    Examples
    --------
    Confirm that a hand-modified steady state still satisfies the model equations. A steady state that does not
    satisfy them raises :class:`~gEconpy.exceptions.SteadyStateNotFoundError` naming the violated equations:

    .. code-block:: python

        from gEconpy import check_steady_state, model_from_gcn
        from gEconpy.data import get_example_gcn

        model = model_from_gcn(get_example_gcn("RBC"), verbose=False)
        steady_state = model.steady_state(verbose=False, progressbar=False)

        check_steady_state(model, steady_state=steady_state)
    """
    if steady_state_kwargs is None:
        steady_state_kwargs = {}

    ss_dict = _maybe_solve_steady_state(model, steady_state, steady_state_kwargs, parameter_updates)
    if ss_dict.success:
        _log.warning("Steady state successfully found!")
        return

    parameters = model.parameters(**parameter_updates)
    residuals = model.evaluate_residual(ss_dict, parameters)
    _log.warning("Steady state NOT successful. The following equations have non-zero residuals:")

    equations = model.equations + list(model._calib_dict.to_sympy().values())
    for resid, eq in zip(residuals, equations, strict=True):
        if np.abs(resid) > _FLOAT_ZERO_TOL:
            _log.warning(eq)
            _log.warning(f"Residual: {resid:0.4f}")


def _maybe_solve_steady_state(
    model: "Model",
    steady_state: SteadyStateResults | None,
    steady_state_kwargs: dict | None,
    parameter_updates: dict | None,
) -> SteadyStateResults:
    if parameter_updates is None:
        parameter_updates = {}
    if steady_state is None:
        if model.is_linear:
            return model.f_ss(**model.parameters(**parameter_updates))

        return model.steady_state(**model.parameters(**parameter_updates), **steady_state_kwargs)

    param_dict = model.parameters(**parameter_updates)
    ss_resid = model.evaluate_residual(steady_state, param_dict)
    unsatisfied_flags = np.abs(ss_resid) > _FLOAT_ZERO_TOL
    unsatisfied_eqs = [f"Equation {i}" for i, flag in enumerate(unsatisfied_flags) if flag]

    if np.any(unsatisfied_flags):
        raise SteadyStateNotFoundError(unsatisfied_eqs)
    steady_state.success = True

    return steady_state


def _maybe_linearize_model(
    model: "Model",
    A: np.ndarray | None,
    B: np.ndarray | None,
    C: np.ndarray | None,
    D: np.ndarray | None,
    **linearize_model_kwargs,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Return the linearized system, calling ``model.linearize_model`` only when any of A, B, C, and D is missing.

    Parameters
    ----------
    model : Model
        DSGE model.
    A : ndarray, optional
        Jacobian of the system with respect to variables at t-1.
    B : ndarray, optional
        Jacobian of the system with respect to variables at t.
    C : ndarray, optional
        Jacobian of the system with respect to variables at t+1.
    D : ndarray, optional
        Jacobian of the system with respect to exogenous shocks.
    **linearize_model_kwargs
        Arguments forwarded to ``model.linearize_model``. Ignored when all of A, B, C, and D are provided.

    Returns
    -------
    linear_system : tuple of ndarray
        The four Jacobians A, B, C, and D.
    """
    verbose = linearize_model_kwargs.get("verbose", True)
    n_matrices = sum(x is not None for x in [A, B, C, D])

    if 0 < n_matrices < 4:
        if verbose:
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
    model: "Model", T: np.ndarray | None, R: np.ndarray | None, **solve_model_kwargs
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return the policy matrices, calling ``model.solve_model`` only when either of T and R is missing.

    Parameters
    ----------
    model : Model
        DSGE model whose solution is T and R.
    T : ndarray, optional
        Transition matrix.
    R : ndarray, optional
        Selection matrix.
    **solve_model_kwargs
        Arguments forwarded to ``model.solve_model``. Ignored when both T and R are provided.

    Returns
    -------
    T : ndarray
        Transition matrix.
    R : ndarray
        Selection matrix.
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
) -> None:
    n_shocks = len(shocks)
    n_provided = sum(x is not None for x in [shock_std_dict, shock_cov_matrix, shock_std])
    if n_provided != 1:
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
        unknown = [x for x in shock_std_dict if x not in shock_names]
        missing = [x for x in shock_names if x not in shock_std_dict]
        if unknown:
            raise ValueError(
                f"Unexpected shocks in shock_std_dict. The following names were not found among the model shocks: "
                f"{', '.join(unknown)}"
            )
        if missing:
            raise ValueError(
                f"If shock_std_dict is specified, it must give values for all shocks. The following shocks were not "
                f"found among the provided keys: {', '.join(missing)}"
            )

    if shock_std is not None:
        if isinstance(shock_std, np.ndarray | list):
            shock_std = np.asarray(shock_std, dtype=float)
            if len(shock_std) != n_shocks:
                raise ValueError(
                    f"Length of shock_std ({len(shock_std)}) does not match the number of shocks ({n_shocks})"
                )
            if not np.all(shock_std > 0):
                raise ValueError("Shock standard deviations must be positive")
        elif isinstance(shock_std, int | float):
            if shock_std < 0:
                raise ValueError("Shock standard deviation must be positive")
