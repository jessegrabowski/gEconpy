import os

from collections.abc import Sequence

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
        return dict(zip(x.dtype.names, x))
    return x


def get_available_models():
    dynare_output_dir = "tests/dynare_outputs"
    mat_files = os.listdir(dynare_output_dir)
    models = [x.replace("_results.mat", "") for x in mat_files]
    return {
        model: os.path.join(dynare_output_dir, fname)
        for model, fname in zip(models, mat_files)
    }


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
    shock_names = np.concatenate(
        [np.atleast_1d(x.item()) for x in np.atleast_1d(M["exo_names"])]
    )
    state_idx = M["state_var"] - 1
    dynare_order = oo["dr"]["order_var"].ravel() - 1

    dr_state_idx = np.array([x for x in dynare_order if x in state_idx])

    dynare_T = pd.DataFrame(
        oo["dr"]["ghx"], index=var_names[dynare_order], columns=var_names[dr_state_idx]
    )
    dynare_R = pd.DataFrame(
        oo["dr"]["ghu"], index=var_names[dynare_order], columns=shock_names
    )

    return dynare_T, dynare_R


def load_dynare_outputs(model_name) -> dict[str, pd.DataFrame]:
    models = get_available_models()
    if model_name not in models:
        raise ValueError(
            f"Model {model_name} not found. Available models are {models.keys()}"
        )

    oo, M = read_dynare_output(model_name)
    T, R = extract_policy_matrices(oo, M)

    return {"T": T, "R": R}
