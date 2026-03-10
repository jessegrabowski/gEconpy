import pandas as pd


def matrix_to_dataframe(
    matrix,
    model,
    dim1: str | None = None,
    dim2: str | None = None,
    round: None | int = None,
) -> pd.DataFrame:
    """
    Convert a matrix to a DataFrame with variable names as columns and rows.

    Parameters
    ----------
    matrix: np.ndarray
        DSGE matrix to convert to a DataFrame. Each dimension should have shape n_variables or n_shocks.
    model: Model
        DSGE model object.
    dim1: str, optional
        Name of the first dimension. One of ``'variable'``, ``'equation'``, or ``'shock'``.
    dim2: str, optional
        Name of the second dimension.
    round: int, optional
        Decimal places.

    Returns
    -------
    pd.DataFrame
    """
    var_names = [x.base_name for x in model.variables]
    shock_names = [x.base_name for x in model.shocks]
    equation_names = [f"Equation {i}" for i in range(len(model.equations))]

    coords = {"variable": var_names, "shock": shock_names, "equation": equation_names}

    n_variables = len(var_names)
    n_shocks = len(shock_names)

    if matrix.ndim != 2:
        raise ValueError("Matrix must be 2-dimensional")

    for i, ordinal in enumerate(["First", "Second"]):
        if matrix.shape[i] not in [n_variables, n_shocks]:
            raise ValueError(
                f"{ordinal} dimension of the matrix must match the number of variables or shocks in the model"
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
