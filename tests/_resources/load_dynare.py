from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import scipy.io as sio


def squeeze_record(x):
    if hasattr(x, "__len__") and len(x) == 1:
        try:
            return squeeze_record(x[0])
        except (IndexError, TypeError):
            pass
    return x


def record_to_dict(x):
    if x.dtype.names is not None:
        return dict(zip(x.dtype.names, x, strict=False))
    return x


def get_available_models():
    dynare_output_dir = Path(__file__).parent / "dynare_outputs"
    dynare_output_paths = list(dynare_output_dir.iterdir())
    model_names = [x.name.replace("_results.mat", "") for x in dynare_output_paths]
    return {name: str(mat_path) for name, mat_path in zip(model_names, dynare_output_paths, strict=False)}


def read_dynare_output(
    model_name,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    models = get_available_models()
    path = models[model_name]

    dynare_data = sio.loadmat(path)

    oo = record_to_dict(squeeze_record(dynare_data["oo_"]))
    for key, value in oo.items():
        oo[key] = record_to_dict(squeeze_record(value))

    M = record_to_dict(squeeze_record(dynare_data["M_"]))
    for k, v in M.items():
        M[k] = squeeze_record(v)

    return oo, M


def extract_policy_matrices(oo, M) -> tuple[pd.DataFrame, pd.DataFrame]:
    var_names = np.concatenate([x.item() for x in M["endo_names"]])
    shock_names = np.concatenate([np.atleast_1d(x.item()) for x in np.atleast_1d(M["exo_names"])])
    state_idx = oo["dr"]["state_var"].ravel().astype(int) - 1
    dynare_order = oo["dr"]["order_var"].ravel().astype(int) - 1

    dr_state_idx = np.array([x for x in dynare_order if x in state_idx])

    dynare_T = pd.DataFrame(oo["dr"]["ghx"], index=var_names[dynare_order], columns=var_names[dr_state_idx])
    dynare_R = pd.DataFrame(oo["dr"]["ghu"], index=var_names[dynare_order], columns=shock_names)

    return dynare_T, dynare_R


class DynarePartition(NamedTuple):
    names: list[str]
    counts: tuple[int, int, int, int]
    current_positions: np.ndarray


def extract_dr_partition(oo, M) -> DynarePartition:
    var_names = np.concatenate([x.item() for x in M["endo_names"]])
    dynare_order = oo["dr"]["order_var"].ravel().astype(int) - 1
    current_incidence = np.asarray(M["lead_lag_incidence"])[1, dynare_order]

    return DynarePartition(
        names=var_names[dynare_order].tolist(),
        counts=(int(M["nstatic"]), int(M["npred"]), int(M["nboth"]), int(M["nfwrd"])),
        current_positions=np.flatnonzero(current_incidence),
    )


class SecondOrderMatrices(NamedTuple):
    """Dynare's second-order decision rule, rows in decision-rule order and Kronecker columns folded into axes."""

    names: list[str]
    state_names: list[str]
    ghxx: np.ndarray
    ghxu: np.ndarray
    ghuu: np.ndarray
    ghs2: np.ndarray
    shock_cov: np.ndarray


def extract_second_order_matrices(oo, M) -> SecondOrderMatrices:
    var_names = np.concatenate([x.item() for x in M["endo_names"]])
    dynare_order = oo["dr"]["order_var"].ravel().astype(int) - 1
    state_idx = oo["dr"]["state_var"].ravel().astype(int) - 1
    dr_states = [i for i in dynare_order if i in state_idx]

    n_variables, n_states, n_shocks = len(dynare_order), len(dr_states), oo["dr"]["ghu"].shape[1]

    # MATLAB's kron(P, Q) puts entry (i, j) of a column pair at column i * Q.shape[1] + j, which is the row-major
    # reshape of the folded axes.
    return SecondOrderMatrices(
        names=var_names[dynare_order].tolist(),
        state_names=var_names[dr_states].tolist(),
        ghxx=oo["dr"]["ghxx"].reshape(n_variables, n_states, n_states),
        ghxu=oo["dr"]["ghxu"].reshape(n_variables, n_states, n_shocks),
        ghuu=oo["dr"]["ghuu"].reshape(n_variables, n_shocks, n_shocks),
        ghs2=oo["dr"]["ghs2"].ravel(),
        shock_cov=np.atleast_2d(M["Sigma_e"]),
    )


def load_dynare_outputs(model_name) -> dict[str, pd.DataFrame]:
    models = get_available_models()
    if model_name not in models:
        raise ValueError(f"Model {model_name} not found. Available models are {models.keys()}")

    oo, M = read_dynare_output(model_name)
    T, R = extract_policy_matrices(oo, M)

    return {"T": T, "R": R}
