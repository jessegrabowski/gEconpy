import os

from pathlib import Path
from unittest import mock

import pytest

from gEconpy import model_from_gcn
from gEconpy.exceptions import ExtraParameterError, OrphanParameterError
from gEconpy.model.model import Model

expected_warnings = [
    "Simplification via a tryreduce block was requested but not possible because the system is not well defined.",
    "Removal of constant variables was requested but not possible because the system is not well defined.",
    "The model does not appear correctly specified, there are 8 equations but 12 variables. It will not be possible to "
    "solve this model. Please check the specification using available diagnostic tools, and check the GCN file for "
    "typos.",
]


@pytest.mark.parametrize(
    ["simplify_tryreduce", "simplify_constants", "expected_warning"],
    [
        (True, False, expected_warnings[0]),
        (False, True, expected_warnings[1]),
        (False, False, expected_warnings[2]),
    ],
    ids=["tryreduce", "constants", "no_simplify"],
)
def test_build_warns_if_model_not_defined(gcn_file_1, simplify_tryreduce, simplify_constants, expected_warning):
    with (
        mock.patch(
            "pathlib.Path.open",
            new=mock.mock_open(read_data=gcn_file_1),
            create=True,
        ),
        pytest.warns(UserWarning, match=expected_warning),
    ):
        model_from_gcn(
            gcn_file_1,
            simplify_constants=simplify_constants,
            simplify_tryreduce=simplify_tryreduce,
            verbose=not (simplify_tryreduce or simplify_constants),
        )


def test_missing_parameters_raises():
    GCN_file = """
                block HOUSEHOLD
                {
                    definitions
                    {
                        u[] = log(C[]);
                    };

                    objective
                    {
                        U[] = u[] + beta * E[][U[1]];
                    };

                    controls
                    {
                        C[], K[], K[-1], Y[];
                    };

                    constraints
                    {
                        Y[] = K[-1] ^ alpha;
                        Y[] = r[] * K[-1];
                        K[] = (1 - delta) * K[-1];

                    };

                    calibration
                    {
                        K[ss] / Y[ss] = 0.33 -> alpha;
                        delta = 0.035;
                    };
                };
                """

    with (
        mock.patch(
            "pathlib.Path.open",
            new=mock.mock_open(read_data=GCN_file),
            create=True,
        ),
        pytest.raises(
            OrphanParameterError,
            match=r"The following parameter was found among model equations but did not appear in "
            r"any calibration block: beta",
        ),
    ):
        model_from_gcn(
            GCN_file,
            verbose=False,
            simplify_tryreduce=False,
            simplify_constants=False,
        )


simple_vars = ["L", "K", "A", "Y", "I", "C", "q", "U", "lambda", "q"]
simple_params = ["alpha", "theta", "beta", "delta", "tau", "rho"]
simple_shocks = ["epsilon"]
open_vars = [
    "A",
    "IIP",
    "r",
    "r_given",
    "KtoN",
    "N",
    "K",
    "C",
    "U",
    "Y",
    "I",
    "TB",
    "TBtoY",
    "CA",
    "lambda",
]
open_params = [
    "beta",
    "delta",
    "gamma",
    "omega",
    "gamma_rv",
    "omega_rv",
    "psi2",
    "psi",
    "alpha",
    "rstar",
    "IIPbar",
    "rho_A",
]
open_shocks = ["epsilon_A"]
nk_vars = [
    "shock_technology",
    "shock_preference",
    "pi",
    "pi_star",
    "pi_obj",
    "B",
    "r",
    "r_G",
    "mc",
    "w",
    "w_star",
    "Y",
    "C",
    "lambda",
    "q",
    "I",
    "K",
    "L",
    "U",
    "TC",
    "Div",
    "LHS",
    "RHS",
    "LHS_w",
    "RHS_w",
]
nk_params = [
    "delta",
    "beta",
    "sigma_C",
    "sigma_L",
    "gamma_I",
    "phi_H",
    "psi_w",
    "eta_w",
    "alpha",
    "rho_technology",
    "rho_preference",
    "psi_p",
    "eta_p",
    "gamma_R",
    "gamma_pi",
    "gamma_Y",
    "phi_pi_obj",
    "rho_pi_dot",
]
nk_shocks = ["epsilon_R", "epsilon_pi", "epsilon_Y", "epsilon_preference"]


@pytest.mark.parametrize(
    "gcn_path, expected_variables, expected_params, expected_shocks",
    [
        (
            "one_block_1_ss.gcn",
            simple_vars,
            simple_params,
            simple_shocks,
        ),
        ("open_rbc.gcn", open_vars, open_params, open_shocks),
        pytest.param("full_nk.gcn", nk_vars, nk_params, nk_shocks, marks=pytest.mark.include_nk),
    ],
)
def test_variables_parsed(gcn_path, expected_variables, expected_params, expected_shocks):
    file_path = Path("tests") / "_resources" / "test_gcns" / gcn_path
    model = model_from_gcn(
        file_path,
        verbose=False,
        backend="numpy",
        mode="FAST_COMPILE",
        simplify_constants=False,
        simplify_tryreduce=False,
    )

    model_vars = [v.base_name for v in model.variables]
    model_params = [p.name for p in model.params + model.calibrated_params + model.deterministic_params]
    model_shocks = [s.base_name for s in model.shocks]

    assert set(model_vars) - set(expected_variables) == set() and set(expected_variables) - set(model_vars) == set()
    assert set(model_params) - set(expected_params) == set() and set(expected_params) - set(model_params) == set()
    assert set(model_shocks) - set(expected_shocks) == set() and set(expected_shocks) - set(model_shocks) == set()


@pytest.mark.parametrize(
    "gcn_file",
    [
        "one_block_1_ss.gcn",
        "open_rbc.gcn",
        pytest.param("full_nk.gcn", marks=pytest.mark.include_nk),
    ],
    ids=["one_block_simple", "open_rbc", "full_nk"],
)
def test_load_gcn(gcn_file):
    mod = model_from_gcn(
        Path("tests") / "_resources" / "test_gcns" / gcn_file,
        simplify_blocks=True,
        verbose=False,
    )
    assert isinstance(mod, Model)
    assert len(mod.shocks) > 0
    assert len(mod.variables) > 0
    assert len(mod.equations) > 0

    assert mod.f_params is not None

    assert mod.f_ss is not None
    assert mod.f_ss_jac is not None

    assert mod.f_ss_resid is not None
    assert mod.f_ss_error_grad is not None
    assert mod.f_ss_error_hess is not None


def test_loading_fails_if_orphan_parameters():
    with pytest.raises(OrphanParameterError):
        model_from_gcn(Path("tests") / "_resources" / "test_gcns" / "open_rbc_orphan_params.gcn")


def test_loading_fails_if_extra_parameters():
    with pytest.raises(ExtraParameterError):
        model_from_gcn(Path("tests") / "_resources" / "test_gcns" / "open_rbc_extra_params.gcn")


def test_build_report(caplog):
    model_from_gcn(
        "tests/_resources/test_gcns/rbc_2_block.gcn",
        verbose=True,
        simplify_tryreduce=True,
        simplify_constants=True,
        simplify_blocks=True,
    )

    expected_report = r"""
                Model Building Complete.
                Found:
                    12 equations
                    12 variables
                    The following "variables" were defined as constants and have been substituted away:
                        P_t
                    1 stochastic shock
                        0 / 1 has a defined prior.
                    6 parameters
                        0 / 6 parameters has a defined prior.
                    0 parameters to calibrate.
                    Model appears well defined and ready to proceed to solving."""

    expected_lines = [x.strip() for x in expected_report.strip().split("\n")]
    found_lines = [x.strip() for x in caplog.messages[-1].strip().split("\n")]

    for line1, line2 in zip(expected_lines, found_lines, strict=True):
        assert line1 == line2
