from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from gEconpy.model.model import Model


def matrix_to_dataframe(
    matrix: np.ndarray,
    model: "Model",
    dim1: str | None = None,
    dim2: str | None = None,
    round: int | None = None,
) -> pd.DataFrame:
    """
    Label the rows and columns of a model matrix with variable, shock, or equation names.

    Parameters
    ----------
    matrix : ndarray
        Two-dimensional matrix whose axes have length ``n_variables`` or ``n_shocks``.
    model : Model
        Model whose variables, shocks, and equations label the axes.
    dim1 : str, optional
        Label set for the rows, one of ``'variable'``, ``'equation'``, or ``'shock'``. Inferred from the row count
        when None, with ``'variable'`` preferred over ``'shock'``. Defaults to None.
    dim2 : str, optional
        Label set for the columns, one of ``'variable'``, ``'equation'``, or ``'shock'``. Inferred from the column
        count when None, with ``'variable'`` preferred over ``'shock'``. Defaults to None.
    round : int, optional
        Number of decimal places to round to. No rounding when None. Defaults to None.

    Returns
    -------
    labeled : DataFrame
        ``matrix`` with named index and columns.

    Examples
    --------
    Label the policy matrices of a solved model. Both axes of ``T`` are variables, so the defaults apply. The rows of a
    Jacobian such as ``B`` are equations, which the row count cannot tell apart from variables, so name the row
    dimension explicitly:

    .. code-block:: python

        from gEconpy import matrix_to_dataframe, model_from_gcn
        from gEconpy.data import get_example_gcn

        model = model_from_gcn(get_example_gcn("RBC"), verbose=False)
        A, B, C, D = model.linearize_model(verbose=False)
        T, R = model.solve_model(verbose=False)

        T_df = matrix_to_dataframe(T, model, round=3)
        B_df = matrix_to_dataframe(B, model, dim1="equation", round=3)
    """
    var_names = [x.base_name for x in model.variables]
    shock_names = [x.base_name for x in model.shocks]
    equation_names = [f"Equation {i}" for i in range(len(model.equations))]

    coords = {"variable": var_names, "shock": shock_names, "equation": equation_names}

    n_variables = len(var_names)
    n_shocks = len(shock_names)

    if matrix.ndim != 2:
        raise ValueError(f"matrix must be 2-dimensional, but has {matrix.ndim} dimensions.")

    for i, ordinal in enumerate(["First", "Second"]):
        if matrix.shape[i] not in [n_variables, n_shocks]:
            raise ValueError(
                f"{ordinal} dimension of the matrix has length {matrix.shape[i]}, which matches neither the number "
                f"of variables ({n_variables}) nor the number of shocks ({n_shocks}) in the model."
            )

    if dim1 is None:
        dim1 = "variable" if matrix.shape[0] == n_variables else "shock"
    if dim2 is None:
        dim2 = "variable" if matrix.shape[1] == n_variables else "shock"

    df = pd.DataFrame(
        matrix,
        index=coords[dim1],
        columns=coords[dim2],
    )

    if round is not None:
        return df.round(round)

    return df
