import re
import warnings

import pytest

from gEconpy import model_from_gcn
from gEconpy.exceptions import ExtraParameterError, ExtraParameterWarning, OrphanParameterError
from tests.conftest import TEST_GCNS


@pytest.mark.parametrize(
    ["simplify_tryreduce", "simplify_constants", "expected_warning"],
    [
        (
            True,
            False,
            "Simplification via a tryreduce block was requested but not possible because the system is not well "
            "defined.",
        ),
        (
            False,
            True,
            "Removal of constant variables was requested but not possible because the system is not well defined.",
        ),
        (
            False,
            False,
            "The model does not appear correctly specified, there are 8 equations but 12 variables, so the model "
            "cannot be solved. Add or remove equations until the counts match, and check the GCN file for a "
            "misspelled variable name that the parser reads as a new variable.",
        ),
    ],
    ids=["tryreduce", "constants", "no_simplify"],
)
def test_build_warns_if_model_not_defined(
    gcn_file_1, simplify_tryreduce, simplify_constants, expected_warning, tmp_path
):
    gcn_path = tmp_path / "test_model.gcn"
    gcn_path.write_text(gcn_file_1)

    with pytest.warns(UserWarning, match=expected_warning):
        model_from_gcn(
            gcn_path,
            simplify_constants=simplify_constants,
            simplify_tryreduce=simplify_tryreduce,
            verbose=not (simplify_tryreduce or simplify_constants),
        )


def test_missing_parameters_raises(tmp_path):
    gcn_source = """
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

    gcn_path = tmp_path / "missing_params.gcn"
    gcn_path.write_text(gcn_source)

    with pytest.raises(
        OrphanParameterError,
        match=r"The following parameter was found among model equations but did not appear in "
        r"any calibration block: beta",
    ):
        model_from_gcn(
            gcn_path,
            verbose=False,
            simplify_tryreduce=False,
            simplify_constants=False,
        )


simple_vars = ["L", "K", "A", "Y", "I", "C", "q", "U", "lambda"]
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
    model = model_from_gcn(
        TEST_GCNS / gcn_path,
        verbose=False,
        mode="FAST_COMPILE",
        simplify_constants=False,
        simplify_tryreduce=False,
    )

    model_vars = [v.base_name for v in model.variables]
    model_params = [p.name for p in model.params + model.calibrated_params + model.deterministic_params]
    model_shocks = [s.base_name for s in model.shocks]

    assert set(model_vars) == set(expected_variables)
    assert set(model_params) == set(expected_params)
    assert set(model_shocks) == set(expected_shocks)


@pytest.mark.parametrize(
    "gcn_file",
    [
        "one_block_1_ss.gcn",
        "open_rbc.gcn",
        pytest.param("full_nk.gcn", marks=pytest.mark.include_nk),
    ],
    ids=["one_block_simple", "open_rbc", "full_nk"],
)
def test_load_gcn_with_block_simplification_yields_square_solvable_model(gcn_file):
    mod = model_from_gcn(TEST_GCNS / gcn_file, simplify_blocks=True, verbose=False)
    assert len(mod.equations) == len(mod.variables)

    ss = mod.steady_state(verbose=False, progressbar=False)
    assert ss.success


def test_loading_fails_if_orphan_parameters():
    with pytest.raises(OrphanParameterError, match="any calibration block: orphan"):
        model_from_gcn(TEST_GCNS / "open_rbc_orphan_params.gcn")


EXTRA_PARAMETER_MESSAGE = (
    "The following parameters were given initial values in calibration blocks but were not used in model equations: "
    "extra_param, sigma_epsilon_A. Delete them from the calibration block, or fix the equation that should use them."
)


def test_loading_fails_if_extra_parameters():
    with pytest.raises(ExtraParameterError, match=re.escape(EXTRA_PARAMETER_MESSAGE)):
        model_from_gcn(TEST_GCNS / "open_rbc_extra_params.gcn")


def test_extra_parameters_warn_when_requested():
    with pytest.warns(ExtraParameterWarning, match=re.escape(EXTRA_PARAMETER_MESSAGE)):
        model = model_from_gcn(TEST_GCNS / "open_rbc_extra_params.gcn", verbose=False, on_unused_parameters="warn")

    assert "extra_param" in model.parameters()


def test_extra_parameters_ignored_when_requested():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model = model_from_gcn(TEST_GCNS / "open_rbc_extra_params.gcn", verbose=False, on_unused_parameters="ignore")

    assert "extra_param" in model.parameters()


@pytest.mark.parametrize("backend, expected_mode", [("numpy", "FAST_COMPILE"), ("pytensor", None)], ids=str)
def test_deprecated_backend_maps_to_mode_and_logs_deprecation(backend, expected_mode, caplog):
    with caplog.at_level("WARNING"):
        model = model_from_gcn(TEST_GCNS / "one_block_1_ss.gcn", verbose=False, backend=backend)

    assert model._mode == expected_mode
    assert "The `backend` argument is deprecated" in caplog.text


def test_invalid_backend_raises():
    with pytest.raises(ValueError, match="Invalid backend='jax'"):
        model_from_gcn(TEST_GCNS / "one_block_1_ss.gcn", verbose=False, backend="jax")


def test_build_report(caplog):
    model_from_gcn(
        TEST_GCNS / "rbc_2_block.gcn",
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
                         0 / 1 have a defined prior.
                    6 parameters
                         0 / 6 have a defined prior.
                    0 parameters to calibrate.
                    1 / 12 variables have analytical steady-state values.
                        1 inferred: A_ss
                    Model appears well defined and ready to proceed to solving."""

    expected_lines = [x.strip() for x in expected_report.strip().split("\n")]
    found_lines = [x.strip() for x in caplog.messages[-1].strip().split("\n")]

    for line1, line2 in zip(expected_lines, found_lines, strict=True):
        assert line1 == line2
