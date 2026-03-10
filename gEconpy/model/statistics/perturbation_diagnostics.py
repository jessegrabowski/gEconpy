from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
import xarray as xr

from gEconpy.exceptions import PerturbationSolutionNotFoundException
from gEconpy.model.perturbation import check_bk_condition as _check_bk_condition
from gEconpy.model.statistics.validation import _maybe_linearize_model

if TYPE_CHECKING:
    from gEconpy.model.model import Model


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
