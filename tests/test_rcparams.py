import numpy as np
import pytest
import scipy.sparse as sp

from gEconpy import model_from_gcn, rc_context, rcParams, statespace_from_gcn
from gEconpy.model.statespace import VALID_SOLVERS
from gEconpy.rcparams import (
    _ADJOINT_METHOD_OPTIONS,
    _STATESPACE_SOLVER_OPTIONS,
    RcParams,
    _load_rc_params,
    find_rc_file,
    read_rc_file,
)
from gEconpy.solvers.sparse_root.sparse_root import sparse_root
from gEconpy.solvers.stein import ADJOINT_METHODS
from tests._resources.cache_compiled_models import load_and_cache_model
from tests.conftest import TEST_GCNS


def test_an_unknown_key_raises_with_a_suggestion():
    with pytest.raises(KeyError, match=r"Did you mean 'steady_state\.rtol'"):
        RcParams()["steady_state.rtl"] = 1e-8


@pytest.mark.parametrize(
    "key, value",
    [
        ("steady_state.tol", -1e-8),
        ("steady_state.grad_tol", 0.0),
        ("sparse_root.maxiter", 1.5),
        ("sparse_root.maxiter", True),
        ("perturbation.tol", "tight"),
        ("statespace.adjoint_method", "newton"),
        ("statespace.use_direct_lyapunov", "yes"),
    ],
    ids=[
        "negative_tol",
        "zero_positive_tol",
        "fractional_int",
        "bool_int",
        "non_numeric",
        "unknown_choice",
        "non_bool",
    ],
)
def test_an_invalid_value_raises_naming_the_key(key, value):
    with pytest.raises(ValueError, match=rf"rcParams\['{key}'\]"):
        RcParams()[key] = value


def test_string_values_are_converted_to_the_key_type():
    params = RcParams(
        {
            "sparse_root.maxiter": "1e3",
            "steady_state.rtol": "1e-8",
            "statespace.use_adjoint_gradients": "False",
            "build.mode": "None",
        }
    )

    assert params["sparse_root.maxiter"] == 1000
    assert isinstance(params["sparse_root.maxiter"], int)
    assert params["steady_state.rtol"] == 1e-8
    assert params["statespace.use_adjoint_gradients"] is False
    assert params["build.mode"] is None


def test_choice_keys_accept_exactly_the_options_the_solvers_do():
    assert _STATESPACE_SOLVER_OPTIONS == VALID_SOLVERS
    assert _ADJOINT_METHOD_OPTIONS == ADJOINT_METHODS


def test_rc_context_restores_every_key_after_an_error(tmp_path):
    rc_file = tmp_path / "geconpyrc"
    rc_file.write_text("sparse_root.maxiter : 7\n")
    before = dict(rcParams)

    with pytest.raises(RuntimeError), rc_context({"steady_state.tol": 1e-3}, fname=rc_file):
        assert rcParams["steady_state.tol"] == 1e-3
        assert rcParams["sparse_root.maxiter"] == 7
        raise RuntimeError

    assert dict(rcParams) == before


def test_rc_file_skips_comments_and_blank_lines(tmp_path):
    rc_file = tmp_path / "geconpyrc"
    rc_file.write_text("# tolerances\n\nsteady_state.rtol : 1e-8  # loosened\nsparse_root.maxiter: 50\n")

    assert read_rc_file(rc_file) == {"steady_state.rtol": 1e-8, "sparse_root.maxiter": 50}


@pytest.mark.parametrize(
    "line, message",
    [
        ("steady_state.rtl : 1e-8", "Did you mean"),
        ("steady_state.rtol : -1", "expected a float >= 0"),
        ("steady_state.rtol 1e-8", "key : value"),
    ],
    ids=["unknown_key", "invalid_value", "missing_separator"],
)
def test_a_bad_rc_file_line_raises_naming_the_file_and_line(tmp_path, line, message):
    rc_file = tmp_path / "geconpyrc"
    rc_file.write_text(f"# header\n{line}\n")

    with pytest.raises(ValueError, match=rf"geconpyrc, line 2: .*{message}"):
        read_rc_file(rc_file)


def test_the_working_directory_rc_file_wins_over_the_config_directory(tmp_path, monkeypatch):
    config_file = tmp_path / "config" / "geconpy" / "geconpyrc"
    config_file.parent.mkdir(parents=True)
    config_file.write_text("")
    working_directory = tmp_path / "work"
    working_directory.mkdir()
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.chdir(working_directory)

    assert find_rc_file() == config_file

    (working_directory / "geconpyrc").write_text("")
    assert find_rc_file() == working_directory / "geconpyrc"


def test_the_rc_file_found_at_import_sets_the_initial_values(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "empty"))
    monkeypatch.chdir(tmp_path)
    (tmp_path / "geconpyrc").write_text("build.verbose : False\n")

    loaded = _load_rc_params()

    assert loaded["build.verbose"] is False
    assert loaded["steady_state.tol"] == RcParams()["steady_state.tol"]


def test_model_from_gcn_reads_its_build_settings(caplog):
    settings = {"build.verbose": False, "build.mode": "FAST_COMPILE", "build.on_unused_parameters": "ignore"}

    with caplog.at_level("INFO"), rc_context(settings):
        model = model_from_gcn(TEST_GCNS / "open_rbc_extra_params.gcn")

    assert "extra_param" in model.parameters()
    assert model._mode == "FAST_COMPILE"
    assert "Model Building Complete" not in caplog.text


def test_steady_state_tolerances_decide_whether_the_analytic_steady_state_is_accepted():
    model = load_and_cache_model("one_block_1.gcn")

    with rc_context({"steady_state.tol": 0.0, "steady_state.rtol": 0.0}):
        exact = model.steady_state(verbose=False)

    assert model.steady_state(verbose=False).success
    assert not exact.success


def test_sparse_root_reads_its_iteration_cap():
    def fun(x):
        return np.array([x[0] ** 3 - 8.0]), sp.csc_array([[3 * x[0] ** 2]])

    with rc_context({"sparse_root.maxiter": 2}):
        capped = sparse_root(fun, np.array([10.0]), progressbar=False)
    explicit = sparse_root(fun, np.array([10.0]), maxiter=2, progressbar=False)

    assert capped.nit == explicit.nit == 2
    assert not capped.success


def test_statespace_configure_reads_its_solver_settings_unless_given():
    with rc_context({"build.mode": "FAST_COMPILE"}):
        ss_mod = statespace_from_gcn(TEST_GCNS / "rbc_linearized.gcn", verbose=False)
    settings = {
        "statespace.solver": "scan_cycle_reduction",
        "statespace.max_iter": 7,
        "statespace.tol": 1e-4,
        "statespace.use_adjoint_gradients": False,
        "statespace.adjoint_method": "qz",
        "statespace.use_direct_lyapunov": True,
    }

    with rc_context(settings):
        ss_mod.configure(observed_states=["Y"], tol=1e-3, verbose=False)

    assert ss_mod._solver == "scan_cycle_reduction"
    assert ss_mod._solver_kwargs == {
        "tol": 1e-3,
        "max_iter": 7,
        "use_adjoint_gradients": False,
        "adjoint_method": "qz",
    }
    assert ss_mod.use_direct_lyapunov
