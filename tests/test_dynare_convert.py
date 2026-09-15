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
    write_lines_from_list,
    write_model_equations,
    write_param_names,
    write_parameter_declarations,
    write_shock_declarations,
    write_shock_std,
    write_steady_state,
    write_variable_declarations,
)
from gEconpy.parser.constants import LOCAL_DICT
from tests._resources.cache_compiled_models import load_and_cache_model
from tests.conftest import TEST_GCNS


@pytest.mark.parametrize(
    "expression, expected",
    [
        ("a * x - 4", "a * x - 4"),
        ("a / x - 4", "a / x - 4"),
        ("alpha * (beta * gamma + 1) - sigma", "alpha * (beta * gamma + 1) - sigma"),
        ("alpha / (beta / gamma + 1) - sigma", "alpha / (beta / gamma + 1) - sigma"),
        ("a ** 2 - 4", "a ^ 2 - 4"),
        ("alpha ** (beta ** gamma) - sigma", "alpha ^ (beta ^ gamma) - sigma"),
        ("x ** 0.5", "sqrt(x)"),
        ("zeta ** (-1)", "1 / zeta"),
        ("(omega * eta) ** (-0.5)", "1 / sqrt(eta * omega)"),
        ("x ** (-2)", "x ^ (-2)"),
        ("x / (y * z)", "x / (y * z)"),
        ("x ** (-1) * y ** (-1)", "1 / (x * y)"),
        ("x / (a + b)", "x / (a + b)"),
        ("-2 * x / (y * z)", "-2 * x / (y * z)"),
    ],
    ids=[
        "mul",
        "div",
        "nested_mul",
        "nested_div",
        "pow",
        "nested_pow",
        "sqrt",
        "reciprocal",
        "reciprocal_sqrt_of_product",
        "negative_integer_pow",
        "product_denominator",
        "reciprocals_merge_into_one_denominator",
        "sum_denominator",
        "negative_coefficient",
    ],
)
def test_printer_writes_dynare_syntax(expression, expected):
    # Sympy orders the factors of a product alphabetically, so the expected strings follow that order.
    printer = DynareCodePrinter()
    expr = sp.parse_expr(expression, transformations="all", local_dict=LOCAL_DICT)

    assert printer.doprint(expr) == expected


@pytest.mark.parametrize("name", ["a", "alpha", "x", "beta", "a_name_with_underscores"])
@pytest.mark.parametrize(
    "time_index, expected_suffix",
    [(0, ""), (1, "(+1)"), (-1, "(-1)"), ("ss", "_ss")],
    ids=["t", "t+1", "t-1", "ss"],
)
def test_print_time_aware_symbol(name, time_index, expected_suffix):
    printer = DynareCodePrinter()
    out = printer.doprint(TimeAwareSymbol(name, time_index))

    assert out == f"{name}{expected_suffix}"


@pytest.fixture(scope="module")
def model():
    return load_and_cache_model("one_block_1_dist.gcn")


@pytest.fixture(scope="module")
def ss_model():
    return load_and_cache_model("one_block_1_ss.gcn")


@pytest.fixture(scope="module")
def nk_model():
    return load_and_cache_model("full_nk.gcn")


def _declared_names(block: str, keyword: str) -> list[str]:
    lines = block.split("\n")
    assert all(line.startswith(f"{keyword} ") and line.endswith(";") for line in lines)

    return [name.strip() for line in lines for name in line.removeprefix(keyword).removesuffix(";").split(",")]


@pytest.mark.parametrize("linewidth", [100, 12], ids=["one_line", "wrapped"])
def test_write_lines_from_list_wraps_and_keeps_every_item(linewidth):
    items = ["alpha", "beta", "delta", "gamma", "rho"]
    out = write_lines_from_list(items, linewidth=linewidth, line_start="var")

    assert _declared_names(out, "var") == items
    assert max(len(line) for line in out.split("\n")) <= linewidth
    assert (len(out.split("\n")) > 1) == (linewidth == 12)


def test_write_lines_from_list_never_emits_empty_declaration():
    out = write_lines_from_list(["a_very_long_name", "b"], linewidth=10, line_start="var")
    assert _declared_names(out, "var") == ["a_very_long_name", "b"]
    assert "var;" not in out.split("\n")


def test_write_variable_declarations(model):
    out = write_variable_declarations(model)
    assert _declared_names(out, "var") == [x.base_name for x in model.variables]


def test_write_shock_declarations(model):
    out = write_shock_declarations(model)
    assert _declared_names(out, "varexo") == [x.base_name for x in model.shocks]


def test_write_param_names(model):
    out = write_param_names(model)
    assert _declared_names(out, "parameters") == [x.name for x in model.params]


def test_write_parameter_declarations_assigns_calibrated_values(model):
    out = write_parameter_declarations(model)
    names_block, values_block = out.split("\n\n")

    assignments = dict(re.findall(r"^(\w+) = ([-\d.]+);$", values_block, flags=re.MULTILINE))
    assert _declared_names(names_block, "parameters") == list(assignments)
    assert {name: float(value) for name, value in assignments.items()} == pytest.approx(
        dict(model.parameters()), abs=5e-4
    )


def test_find_ss_variables_sorted_by_base_name(nk_model):
    assert [x.name for x in find_ss_variables(nk_model)] == ["pi_ss", "r_G_ss"]


def test_write_model_equations_defines_analytic_ss_values_before_equations(nk_model):
    out = write_model_equations(nk_model)
    assert out.startswith("model;")
    assert out.endswith("end;")

    body = [line for line in out.removeprefix("model;").removesuffix("end;").split("\n") if line]
    ss_definitions = [line for line in body if line.startswith("#")]
    equations = [line for line in body if not line.startswith("#")]

    assert body == ss_definitions + equations
    assert all(line.endswith(";") for line in body)
    assert [re.match(r"#(\w+) = ", line).group(1) for line in ss_definitions] == ["pi_ss", "r_G_ss"]
    assert len(equations) == len(nk_model.equations)


def test_write_model_equations_falls_back_to_numeric_ss_values(tmp_path):
    # The added identity references C[ss], which the model has no analytic expression for.
    source = (TEST_GCNS / "one_block_1.gcn").read_text()
    identity = "        log(A[]) = rho * log(A[-1]) + epsilon[];\n"
    source = source.replace(identity, identity + "        C_hat[] = log(C[]) - log(C[ss]);\n")
    gcn_path = tmp_path / "one_block_numeric_ss.gcn"
    gcn_path.write_text(source)
    numeric_ss_model = model_from_gcn(gcn_path, verbose=False)

    out = write_model_equations(numeric_ss_model)

    ss_definitions = re.findall(r"^#(\w+) = ([-\d.e]+);$", out, flags=re.MULTILINE)
    expected_C_ss = numeric_ss_model.steady_state(verbose=False, progressbar=False)["C_ss"]
    assert [name for name, _ in ss_definitions] == ["C_ss"]
    assert float(ss_definitions[0][1]) == pytest.approx(expected_C_ss)


def test_write_steady_state_initval_solves_the_model(model):
    out = write_steady_state(model)
    assert out.startswith("initval;")
    assert out.endswith("end;\n\nsteady;\nresid;")

    assignments = re.findall(r"^(\w+) = ([-\d.]+);$", out, flags=re.MULTILINE)
    ss_dict = {f"{name}_ss": float(value) for name, value in assignments}
    assert len(ss_dict) == len(model.variables)

    np.testing.assert_allclose(
        model.evaluate_residual(ss_dict, model.parameters()),
        np.zeros(len(ss_dict)),
        atol=1e-3,
        rtol=1e-3,
    )


@pytest.mark.parametrize("use_cse", [True, False], ids=["cse", "no_cse"])
def test_write_analytical_steady_state(ss_model, use_cse):
    out = write_steady_state(ss_model, use_cse=use_cse)
    assert out.startswith("steady_state_model;")
    assert out.endswith("end;\n\nsteady;\nresid;")

    names = re.findall(r"^(\w+) = .*;$", out, flags=re.MULTILINE)
    variable_names = [eq.lhs.base_name for eq in ss_model.steady_state_relationships]
    cse_names = names[: len(names) - len(variable_names)]

    assert names[-len(variable_names) :] == variable_names
    assert all(re.fullmatch(r"x\d+", name) for name in cse_names)
    assert (len(cse_names) > 0) == use_cse


def test_write_shock_std(model):
    out = write_shock_std(model)
    assert out.startswith("shocks;")
    assert out.endswith("end;")

    lines = [line for line in out.split("\n") if line not in ["shocks;", "end;"] and len(line) > 0]
    assert lines[::2] == [f"var {shock.base_name};" for shock in model.shocks]
    assert lines[1::2] == ["stderr 0.01;"] * len(model.shocks)


@pytest.mark.parametrize("linewidth", [100, 50], ids=["long_lines", "short_lines"])
def test_make_mod_file_wraps_declarations_and_orders_blocks(linewidth, nk_model):
    out = make_mod_file(nk_model, linewidth=linewidth)
    lines = out.split("\n")

    # Model equations are not wrapped, so only the declaration blocks are checked.
    eq_start_idx = lines.index("model;")
    eq_end_idx = lines.index("end;", eq_start_idx)
    declaration_lines = [line for line in lines[:eq_start_idx] + lines[eq_end_idx:] if "=" not in line]
    assert max(len(line) for line in declaration_lines) <= linewidth

    block_starts = ["var ", "varexo ", "parameters ", "model;", "steady;", "check(", "shocks;", "stoch_simul("]
    block_positions = [out.index(start) for start in block_starts]
    assert block_positions == sorted(block_positions)


def test_make_mod_file_out_path_writes_file_and_returns_none(model, tmp_path):
    out_path = tmp_path / "model.mod"
    returned = make_mod_file(model, out_path=out_path)

    assert returned is None
    assert out_path.read_text() == make_mod_file(model)
