import os

import numpy as np
import pymc as pm
import pytensor
import pytest

from gEconpy.model.build import model_from_gcn, statespace_from_gcn


@pytest.mark.parametrize(
    "gcn_file",
    [
        "One_Block_Simple_1_w_Steady_State.gcn",
        "Open_RBC.gcn",
        "Full_New_Keynesian.gcn",
        "RBC_Linearized.gcn",
    ],
)
def test_statespace_matrices_agree_with_model(gcn_file):
    file_path = os.path.join("tests", "Test GCNs", gcn_file)

    ss_mod = statespace_from_gcn(file_path, verbose=False)
    model = model_from_gcn(file_path, verbose=False)

    inputs = pm.inputvars(ss_mod.linearized_system)
    input_names = [x.name for x in inputs]
    f = pytensor.function(inputs, ss_mod.linearized_system, on_unused_input="ignore")
    mod_matrices = model.linearize_model()

    param_dict = model.parameters()
    ss_matrices = f(**{k: param_dict[k] for k in input_names})

    for mod_matrix, ss_matrix in zip(mod_matrices, ss_matrices):
        np.testing.assert_allclose(mod_matrix, ss_matrix, atol=1e-8, rtol=1e-8)
