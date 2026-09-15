import numpy as np
import pytest
import sympy as sp

from gEconpy import model_from_gcn
from gEconpy.data import get_example_gcn
from gEconpy.model.block import Block
from gEconpy.model.block import registry as registry_mod
from gEconpy.model.block.cobb_douglas import CobbDouglasBlock, _match_cobb_douglas_constraint
from gEconpy.parser.loader import load_gcn_file, load_gcn_string
from gEconpy.solvers.cycle_reduction import solve_policy_function_with_cycle_reduction
from tests.conftest import parsed_var

RBC_PATH = get_example_gcn("RBC")


class TestDispatchOnRBC:
    def test_firm_is_dispatched(self):
        primitives = load_gcn_file(RBC_PATH, simplify_blocks=True)
        assert isinstance(primitives.block_dict["FIRM"], CobbDouglasBlock)

    @pytest.mark.parametrize("name", ["HOUSEHOLD", "TECHNOLOGY_SHOCKS"])
    def test_non_firm_blocks_fall_back(self, name):
        primitives = load_gcn_file(RBC_PATH, simplify_blocks=True)
        assert type(primitives.block_dict[name]) is Block


def test_rbc_policy_equivalence(monkeypatch):
    """The dispatched block must give the same policy function as the general Block, to machine epsilon."""
    model = model_from_gcn(RBC_PATH, verbose=False)
    A, B, C, D = model.linearize_model(verbose=False)
    T, R, _, _ = solve_policy_function_with_cycle_reduction(A, B, C, D, 1000, 1e-12, False)

    monkeypatch.setattr(registry_mod, "_REGISTRY", [])
    model_base = model_from_gcn(RBC_PATH, verbose=False)
    A_b, B_b, C_b, D_b = model_base.linearize_model(verbose=False)
    T_b, R_b, _, _ = solve_policy_function_with_cycle_reduction(A_b, B_b, C_b, D_b, 1000, 1e-12, False)

    assert np.max(np.abs(T - T_b)) < 1e-12
    assert np.max(np.abs(R - R_b)) < 1e-12


class TestDetectionConservatism:
    """A false positive silently drops terms from the user's equations, so each near miss here must be rejected."""

    def test_rejects_no_constraint(self):
        assert _match_cobb_douglas_constraint(None) is None
        assert _match_cobb_douglas_constraint({}) is None

    def test_rejects_two_constraints(self):
        Y, A, x1, x2, a1 = sp.symbols("Y A x1 x2 a1")
        constraint = sp.Eq(Y, A * x1**a1 * x2 ** (1 - a1))
        assert _match_cobb_douglas_constraint({0: constraint, 1: constraint}) is None

    @pytest.mark.parametrize(
        "rhs",
        [
            "A * (x1 + x2)",
            "A * x1**0 * x2**1",
            "A * x1",
            "2 * A * x1**a1 * x2**a2",
            "A * x1**a1 * x1**a2",
            "A * x1**a1 * x2**a2 + a1 * x1**2",
        ],
        ids=[
            "sum-not-monomial",
            "exponent-zero",
            "bare-input-ambiguous-with-A",
            "extra-constant-factor",
            "duplicate-input",
            "extra-additive-term",
        ],
    )
    def test_rejects_non_cobb_douglas_constraint(self, rhs):
        Y, A, x1, x2, a1, a2 = sp.symbols("Y A x1 x2 a1 a2")
        constraint = sp.Eq(Y, sp.sympify(rhs, locals={"A": A, "x1": x1, "x2": x2, "a1": a1, "a2": a2}))
        assert _match_cobb_douglas_constraint({0: constraint}) is None

    def test_accepts_crs_form(self):
        Y, A, x1, x2, a1 = sp.symbols("Y A x1 x2 a1")
        match = _match_cobb_douglas_constraint({3: sp.Eq(Y, A * x1**a1 * x2 ** (1 - a1))})

        assert match is not None
        assert match.output == Y
        assert match.productivity == A
        assert match.idx == 3
        inputs = dict(match.inputs)
        assert set(inputs.keys()) == {x1, x2}
        assert inputs[x1] + inputs[x2] == 1

    def test_accepts_non_crs_form(self):
        Y, A, x1, x2, a1, a2 = sp.symbols("Y A x1 x2 a1 a2")
        match = _match_cobb_douglas_constraint({0: sp.Eq(Y, A * x1**a1 * x2**a2)})

        assert match is not None
        inputs = dict(match.inputs)
        assert inputs == {x1: a1, x2: a2}

    @pytest.mark.parametrize("k", [1, 3, 10])
    def test_accepts_arbitrary_arity(self, k):
        Y, A = sp.symbols("Y A")
        xs = sp.symbols(f"x1:{k + 1}")
        exponents = sp.symbols(f"a1:{k + 1}")
        product = A * sp.Mul(*[x**a for x, a in zip(xs, exponents, strict=True)])
        match = _match_cobb_douglas_constraint({0: sp.Eq(Y, product)})

        assert match is not None
        assert match.output == Y
        assert match.productivity == A
        assert dict(match.inputs) == dict(zip(xs, exponents, strict=True))

    def test_detect_requires_objective(self):
        Y, A, x1, x2, a1 = sp.symbols("Y A x1 x2 a1")
        constraints = {0: sp.Eq(Y, A * x1**a1 * x2 ** (1 - a1))}
        assert CobbDouglasBlock.detect(constraints, objective=None, identities=None) is False

    @pytest.mark.parametrize("k", [2, 3])
    def test_no_leading_productivity(self, k):
        Y = sp.Symbol("Y")
        xs = sp.symbols(f"x1:{k + 1}")
        exponents = sp.symbols(f"a1:{k + 1}")
        product = sp.Mul(*[x**a for x, a in zip(xs, exponents, strict=True)])
        match = _match_cobb_douglas_constraint({0: sp.Eq(Y, product)})

        assert match is not None
        assert match.productivity is None
        assert set(dict(match.inputs).keys()) == set(xs)


FIRM_WITH_NON_INPUT_CONTROL = """
block FIRM
{
    controls { K[], L[], B[]; };
    objective { Pi[] = Y[] - r[] * K[] - w[] * L[] - B[] ^ 2 + q[] * B[]; };
    constraints { Y[] = A[] * K[] ^ alpha * L[] ^ (1 - alpha) : mu[]; };
    calibration { alpha = 0.35; };
};
"""

FIRM_WITH_DEFINITION_AND_GENERATED_MULTIPLIER = """
block FIRM
{
    definitions { cost[] = r[] * K[] + w[] * L[]; };
    controls { K[], L[]; };
    objective { Pi[] = Y[] - cost[]; };
    constraints { Y[] = A[] * K[] ^ alpha * L[] ^ (1 - alpha); };
    calibration { alpha = 0.35; };
};
"""


def test_control_outside_the_production_function_falls_back_to_lagrangian_derivative():
    """A control that is not a production input gets the generic chain-rule FOC, here d/dB of the objective."""
    block = load_gcn_string(FIRM_WITH_NON_INPUT_CONTROL).block_dict["FIRM"]
    assert isinstance(block, CobbDouglasBlock)

    B, q = parsed_var("B", 0), parsed_var("q", 0)
    assert sp.simplify(block.system_equations[-1] - (q - 2 * B)) == 0


def test_constructing_without_matching_constraint_raises():
    Y, r, w, Pi = sp.symbols("Y r w Pi")
    with pytest.raises(RuntimeError, match="constructed without a matching Cobb-Douglas constraint"):
        CobbDouglasBlock(
            name="FIRM",
            objective={0: sp.Eq(Pi, Y - r)},
            constraints={1: sp.Eq(Y, r + w)},
            controls=[Y],
            multipliers={0: None, 1: None},
            equation_flags={0: {}, 1: {}},
        )


def test_definition_with_generated_multiplier_solves():
    block = load_gcn_string(FIRM_WITH_DEFINITION_AND_GENERATED_MULTIPLIER).block_dict["FIRM"]
    assert isinstance(block, CobbDouglasBlock)
    assert len(block.system_equations) == 4
