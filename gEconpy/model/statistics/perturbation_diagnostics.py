from __future__ import annotations

import logging

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
import pytensor
import pytensor.tensor as pt
import xarray as xr

from pymc.pytensorf import rewrite_pregrad

from gEconpy.exceptions import PerturbationSolutionNotFoundException
from gEconpy.model.perturbation import (
    check_bk_condition as _check_bk_condition,
)
from gEconpy.model.perturbation import (
    compute_bk_eigenvalues_pt,
)
from gEconpy.model.statistics.validation import _maybe_linearize_model

if TYPE_CHECKING:
    from gEconpy.model.model import Model

_log = logging.getLogger(__name__)


def summarize_perturbation_solution(
    linear_system: Sequence[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    perturbation_solution: Sequence[np.ndarray | None, np.ndarray | None],
    model: Model,
):
    A, B, C, D = linear_system
    T, R = perturbation_solution
    if T is None or R is None:
        raise PerturbationSolutionNotFoundException()

    coords = {
        "equation": np.arange(A.shape[0]).astype(int),
        "variable": [x.base_name for x in model.variables],
        "shock": [x.base_name for x in model.shocks],
    }

    return xr.Dataset(
        data_vars={
            "A": (("equation", "variable"), A),
            "B": (("equation", "variable"), B),
            "C": (("equation", "variable"), C),
            "D": (("equation", "shock"), D),
            "T": (("equation", "variable"), T),
            "R": (("equation", "shock"), R),
        },
        coords=coords,
    )


def check_bk_condition(
    model: Model,
    *,
    A: np.ndarray | None = None,
    B: np.ndarray | None = None,
    C: np.ndarray | None = None,
    D: np.ndarray | None = None,
    tol=1e-8,
    on_failure: Literal["raise", "ignore"] = "ignore",
    return_value: Literal["dataframe", "bool", None] = "dataframe",
    **linearize_model_kwargs,
) -> bool | pd.DataFrame | None:
    """
    Compute the generalized eigenvalues of system in the form presented in [1].

    Per [2], the number of unstable eigenvalues (:math:`|v| > 1`) should not be greater than the number of
    forward-looking variables. Failing this test suggests timing problems in the definition of the model.

    Parameters
    ----------
    model: Model
        DSGE model.
    A, B, C, D : np.ndarray, optional
        Jacobian matrices. If not all provided, ``model.linearize_model`` is called.
    tol : float
        Tolerance for zero.
    on_failure : str
        ``'raise'`` or ``'ignore'``.
    return_value : str or None
        ``'dataframe'``, ``'bool'``, or None.
    **linearize_model_kwargs
        Forwarded to ``model.linearize_model``.

    Returns
    -------
    bk_result : bool, DataFrame, or None
    """
    verbose = linearize_model_kwargs.get("verbose", True)
    A, B, C, D = _maybe_linearize_model(model, A, B, C, D, **linearize_model_kwargs)
    return _check_bk_condition(
        A,
        B,
        C,
        D,
        tol=tol,
        verbose=verbose,
        on_failure=on_failure,
        return_value=return_value,
    )


def eigenvalue_sensitivity(
    model: Model,
    *,
    verbose: bool = True,
    steady_state: dict | None = None,
    steady_state_kwargs: dict | None = None,
    **parameter_updates,
) -> xr.Dataset:
    r"""
    Compute the sensitivity of system eigenvalues to model parameters.

    For each eigenvalue of the linearized DSGE system's augmented form, computes the derivative
    of the real and imaginary parts with respect to every free parameter. This reveals which
    parameters push eigenvalues toward or away from the unit circle, potentially moving the
    model in or out of Blanchard-Kahn stability.

    Parameters
    ----------
    model : Model
        A gEconpy DSGE model.
    verbose : bool, default True
        Forwarded to steady-state and linearization routines.
    steady_state : dict, optional
        Pre-computed steady state. If not provided, solved internally.
    steady_state_kwargs : dict, optional
        Keyword arguments forwarded to ``model.steady_state``.
    **parameter_updates
        Parameter overrides (e.g. ``beta=0.98``).

    Returns
    -------
    xr.Dataset
        Dataset with two data variables:

        - ``eigenvalues`` : shape ``(eigenvalue, component)`` where ``component`` is
          ``[real, imaginary, modulus]``.
        - ``gradients`` : shape ``(eigenvalue, part, parameter)`` where ``part`` is
          ``[real, imaginary]``.

    Notes
    -----
    Eigenvalues are computed from the Sims (2002) augmented system via the ``real_eig`` Op,
    which provides exact reverse-mode gradients.

    Eigenvalues are sorted by modulus (ascending), so zeros appear first and large/infinite
    eigenvalues appear last.

    Examples
    --------
    .. code-block:: python

        from gEconpy.model.build import model_from_gcn
        from gEconpy.model.statistics import eigenvalue_sensitivity

        model = model_from_gcn("rbc.gcn")
        ds = eigenvalue_sensitivity(model, verbose=False)

        # Filter to finite eigenvalues if desired
        mod = ds.eigenvalues.sel(component="modulus")
        finite_mask = (mod > 1e-6) & (mod < 1e6)
        finite_idx = ds.eigenvalue.values[finite_mask.values]
        finite = ds.sel(eigenvalue=finite_idx)
    """
    if steady_state_kwargs is None:
        steady_state_kwargs = {}

    param_dict = model.parameters(**parameter_updates)
    if steady_state is None:
        steady_state = model.steady_state(**param_dict, verbose=verbose, **steady_state_kwargs)

    jacobians, ss_nodes, param_nodes = model.symbolic_linearization(steady_state=steady_state, verbose=False)
    A_sym, B_sym, C_sym, D_sym = jacobians
    param_names = [p.name for p in param_nodes]

    lead_var_idx = model.lead_var_idx
    eigvals_re_pt, eigvals_im_pt = compute_bk_eigenvalues_pt(A_sym, B_sym, C_sym, D_sym, lead_var_idx)
    eigvals_re_pt = rewrite_pregrad(eigvals_re_pt)
    eigvals_im_pt = rewrite_pregrad(eigvals_im_pt)

    n_eig = model.n_variables + model.n_forward

    jac_re = pt.stack(pt.jacobian(eigvals_re_pt, param_nodes), axis=1)
    jac_im = pt.stack(pt.jacobian(eigvals_im_pt, param_nodes), axis=1)

    ss_values = {k.removesuffix("_ss"): v for k, v in steady_state.items()}

    all_inputs = list(ss_nodes) + list(param_nodes)
    input_vals = [float(ss_values[v.base_name]) for v in model.variables]
    input_vals += [float(param_dict[n.name]) for n in param_nodes]

    f = pytensor.function(
        all_inputs, [eigvals_re_pt, eigvals_im_pt, jac_re, jac_im], on_unused_input="ignore", mode=model._mode
    )
    re_vals, im_vals, jac_re_vals, jac_im_vals = f(*input_vals)
    mod_vals = np.sqrt(re_vals**2 + im_vals**2)

    eigenvalue_coords = np.arange(n_eig)
    eigenvalues_data = np.stack([re_vals, im_vals, mod_vals], axis=1)
    gradients_data = np.stack([jac_re_vals, jac_im_vals], axis=1)

    return xr.Dataset(
        {
            "eigenvalues": (["eigenvalue", "component"], eigenvalues_data),
            "gradients": (["eigenvalue", "part", "parameter"], gradients_data),
        },
        coords={
            "eigenvalue": eigenvalue_coords,
            "component": ["real", "imaginary", "modulus"],
            "part": ["real", "imaginary"],
            "parameter": param_names,
        },
    )
