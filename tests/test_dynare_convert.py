import re

import numpy as np
import pytest
import sympy as sp

from gEconpy import model_from_gcn
from gEconpy.classes.time_aware_symbol import TimeAwareSymbol
from gEconpy.dynare_convert import (
    DynareCodePrinter,
    find_ss_variables,
    make_mod_file,
    write_model_equations,
    write_param_names,
    write_parameter_declarations,
    write_shock_declarations,
    write_shock_std,
    write_steady_state,
    write_variable_declarations,
)
from gEconpy.parser.constants import LOCAL_DICT


@pytest.mark.parametrize("op", ["*", "/"], ids=["multiplication", "division"])
def test_print_multiplication(op):
    printer = DynareCodePrinter()
    expr = sp.parse_expr(f"a {op} x - 4", transformations="all")
    out = printer.doprint(expr)

    assert out == f"a {op} x - 4"

    expr = sp.parse_expr(
        f"alpha {op} (beta {op} gamma + 1) - sigma",
        transformations="all",
        local_dict=LOCAL_DICT,
    )
    out = printer.doprint(expr)
    assert out == f"alpha {op} (beta {op} gamma + 1) - sigma"


def test_print_power():
    printer = DynareCodePrinter()
    expr = sp.parse_expr("a ** 2 - 4", transformations="all")
    out = printer.doprint(expr)

    assert out == "a ^ 2 - 4"

    expr = sp.parse_expr("alpha ** (beta ** gamma) - sigma", transformations="all", local_dict=LOCAL_DICT)
    out = printer.doprint(expr)
    assert out == "alpha ^ (beta ^ gamma) - sigma"

    expr = sp.parse_expr("x ** 0.5")
    out = printer.doprint(expr)
    assert out == "sqrt(x)"

    expr = sp.parse_expr("zeta ** (-1)", local_dict=LOCAL_DICT)
    out = printer.doprint(expr)
    assert out == "1 / zeta"

    expr = sp.parse_expr("(omega * eta) ** (-0.5)", local_dict=LOCAL_DICT)
    out = printer.doprint(expr)

    # It alphabetizes?
    assert out == "1 / sqrt(eta * omega)"


@pytest.mark.parametrize("name", ["a", "alpha", "x", "beta", "a_name_with_underscores"])
@pytest.mark.parametrize("time_index", [0, 1, -1, "ss"])
def test_print_time_aware_symbol(name, time_index):
    printer = DynareCodePrinter()
    expr = TimeAwareSymbol(name, time_index)
    out = printer.doprint(expr)

    if time_index == 0:
        assert out == name
    elif time_index == -1:
        assert out == f"{name}({time_index})"
    elif time_index == 1:
        assert out == f"{name}(+{time_index})"
    elif time_index == "ss":
        assert out == f"{name}_ss"


@pytest.fixture()
def model():
    return model_from_gcn("tests/_resources/test_gcns/one_block_1_dist.gcn", verbose=False)


@pytest.fixture()
def ss_model():
    return model_from_gcn("tests/_resources/test_gcns/one_block_1_ss.gcn", verbose=False)


@pytest.fixture()
def nk_model():
    return model_from_gcn("tests/_resources/test_gcns/full_nk.gcn", verbose=False)


def test_write_variable_declarations(model):
    out = write_variable_declarations(model)
    assert out.startswith("var")

    tokens = out.replace("\n", " ").replace(";", " ").replace(",", " ")
    tokens = re.sub(" +", " ", tokens).split(" ")

    assert all(x.base_name in tokens for x in model.variables)


def test_write_shock_declarations(model):
    out = write_shock_declarations(model)
    assert out.startswith("varexo")

    tokens = out.replace("\n", " ").replace(";", " ").replace(",", " ")
    tokens = re.sub(" +", " ", tokens).split(" ")

    assert all(x.base_name in tokens for x in model.shocks)


def test_write_param_names(model):
    out = write_param_names(model)

    assert out.startswith("parameters")

    tokens = out.replace("\n", " ").replace(";", " ").replace(",", " ")
    tokens = re.sub(" +", " ", tokens).split(" ")

    assert all(x.name in tokens for x in model.params)


def test_write_parameter_declarations(model):
    out = write_parameter_declarations(model)
    lines = [line for line in out.split("\n") if not line.startswith("parameters") and len(line) > 0]
    for line in lines:
        name, value = line.replace(" ", "").replace(";", "").split("=")
        assert model.parameters()[name] == float(value)


def test_find_ss_variables(nk_model):
    ss_vars = [x.name for x in find_ss_variables(nk_model)]
    assert all(x in ss_vars for x in ["pi_ss", "r_G_ss"])


def test_write_model_equations(nk_model):
    out = write_model_equations(nk_model)

    assert out.startswith("model;")
    assert out.endswith("end;")

    lines = [line for line in out.split("\n") if line not in ["model;", "end;"] and len(line) > 0]
    count = 0
    expect_ss_definition = True

    for line in lines:
        clean_line = line.replace(" ", "").replace(";", "")
        if clean_line.startswith("#"):
            assert "=" in line
            assert expect_ss_definition  # All the ss definitions should be at the beginning and all together
            name, value = clean_line.split("=")
            assert name.endswith("_ss")

        else:
            expect_ss_definition = False
            count += 1

    assert len(nk_model.equations) == count


def test_write_steady_state(model):
    out = write_steady_state(model)
    assert out.startswith("initval;")
    assert out.endswith("end;\n\nsteady;\nresid;")
    lines = [
        line for line in out.split("\n") if line not in ["initval;", "end;", "steady;", "resid;"] and len(line) > 0
    ]
    ss_dict = {}
    for line in lines:
        name, value = line.replace(";", "").replace(" ", "").split("=")
        ss_dict[f"{name}_ss"] = float(value)

    np.testing.assert_allclose(
        model.f_ss_resid(**ss_dict, **model.parameters()),
        np.zeros(len(ss_dict)),
        atol=1e-3,
        rtol=1e-3,
    )


def test_write_analytical_steady_state(ss_model):
    out = write_steady_state(ss_model)
    assert out.startswith("steady_state_model;")
    lines = [line.replace(" ", "").replace(";", "") for line in out.split("\n") if "=" in line and len(line) > 0]
    names, exprs = zip(*[line.split("=") for line in lines], strict=False)
    n_vars = len(ss_model.variables)

    assert all(x.base_name in names[-n_vars:] for x in ss_model.variables)


def test_write_shock_std(model):
    out = write_shock_std(model)
    assert out.startswith("shocks;")
    assert out.endswith("end;")
    lines = [line for line in out.split("\n") if line not in ["shocks;", "end;"] and len(line) > 0]
    assert all(line.startswith("var") for line in lines[::2])
    assert all(line.startswith("stderr") for line in lines[1::2])


@pytest.mark.parametrize("linewidth", [100, 50], ids=["long_lines", "short_lines"])
def test_make_mod_file(linewidth, nk_model):
    out = make_mod_file(nk_model, linewidth=linewidth)
    assert isinstance(out, str)

    lines = out.split("\n")

    # Model equations don't respect the line length -- filter them out
    eq_start_idx = lines.index("model;")
    eq_end_idx = lines.index("end;", eq_start_idx)

    lines = lines[:eq_start_idx] + lines[eq_end_idx:]
    lines = [line for line in lines if "=" not in line]

    assert max([len(x) for x in lines]) <= linewidth
