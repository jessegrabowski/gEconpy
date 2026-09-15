import numpy as np
import pytest
import sympy as sp

from gEconpy import model_from_gcn
from gEconpy.data import get_example_gcn
from gEconpy.model.block import Block
from gEconpy.model.block import registry as registry_mod
from gEconpy.model.block.ces import CESBlock, _match_ces_constraint
from gEconpy.parser.loader import load_gcn_file, load_gcn_string
from gEconpy.solvers.cycle_reduction import solve_policy_function_with_cycle_reduction
from tests.conftest import parsed_var

RBC_CES_PATH = get_example_gcn("RBC_with_CES")


def canonical_ces(Y, A, x1, x2, alpha, psi):
    s = (psi - 1) / psi
    inner = alpha ** (1 / psi) * x1**s + (1 - alpha) ** (1 / psi) * x2**s
    return sp.Eq(Y, A * inner ** (1 / s))


class TestDispatchOnRBCWithCES:
    def test_firm_is_dispatched(self):
        primitives = load_gcn_file(RBC_CES_PATH, simplify_blocks=True)
        assert isinstance(primitives.block_dict["FIRM"], CESBlock)

    @pytest.mark.parametrize("name", ["HOUSEHOLD", "TECHNOLOGY_SHOCKS"])
    def test_non_firm_blocks_fall_back(self, name):
        primitives = load_gcn_file(RBC_CES_PATH, simplify_blocks=True)
        assert type(primitives.block_dict[name]) is Block


def test_policy_equivalence(monkeypatch):
    """The dispatched block must give the same policy function as the general Block, to machine epsilon."""
    model = model_from_gcn(RBC_CES_PATH, verbose=False)
    A, B, C, D = model.linearize_model(verbose=False)
    T, R, _, _ = solve_policy_function_with_cycle_reduction(A, B, C, D, 1000, 1e-12, False)

    monkeypatch.setattr(registry_mod, "_REGISTRY", [])
    model_base = model_from_gcn(RBC_CES_PATH, verbose=False)
    A_b, B_b, C_b, D_b = model_base.linearize_model(verbose=False)
    T_b, R_b, _, _ = solve_policy_function_with_cycle_reduction(A_b, B_b, C_b, D_b, 1000, 1e-12, False)

    assert np.max(np.abs(T - T_b)) < 1e-12
    assert np.max(np.abs(R - R_b)) < 1e-12


class TestDetectionConservatism:
    """A false positive silently drops terms from the user's equations, so each near miss here must be rejected."""

    def test_rejects_no_constraint(self):
        assert _match_ces_constraint(None) is None
        assert _match_ces_constraint({}) is None

    def test_rejects_two_constraints(self):
        constraint = canonical_ces(*sp.symbols("Y A x1 x2 alpha psi"))
        assert _match_ces_constraint({0: constraint, 1: constraint}) is None

    def test_rejects_cobb_douglas_constraint(self):
        Y, A, x1, x2, a1 = sp.symbols("Y A x1 x2 a1")
        assert _match_ces_constraint({0: sp.Eq(Y, A * x1**a1 * x2 ** (1 - a1))}) is None

    def test_rejects_when_outer_exponent_is_not_reciprocal_of_inner(self):
        Y, A, x1, x2, a, b = sp.symbols("Y A x1 x2 a b")
        assert _match_ces_constraint({0: sp.Eq(Y, A * (x1**a + x2**a) ** b)}) is None

    def test_rejects_extra_constant_factor(self):
        Y, A, x1, x2, alpha, psi = sp.symbols("Y A x1 x2 alpha psi")
        canonical = canonical_ces(Y, A, x1, x2, alpha, psi)
        assert _match_ces_constraint({0: sp.Eq(Y, 2 * canonical.rhs)}) is None

    def test_accepts_canonical_two_input_form(self):
        Y, A, x1, x2, alpha, psi = sp.symbols("Y A x1 x2 alpha psi")
        match = _match_ces_constraint({7: canonical_ces(Y, A, x1, x2, alpha, psi)})

        assert match is not None
        assert match.output == Y
        assert match.productivity == A
        assert match.idx == 7
        assert sp.simplify(match.exponent - (psi - 1) / psi) == 0
        assert set(dict(match.inputs).keys()) == {x1, x2}

    @pytest.mark.parametrize("k", [2, 3, 5])
    def test_accepts_arbitrary_arity(self, k):
        Y, A, psi = sp.symbols("Y A psi")
        xs = sp.symbols(f"x1:{k + 1}")
        shares = sp.symbols(f"alpha1:{k + 1}")
        s = (psi - 1) / psi
        inner = sum(share ** (1 / psi) * x**s for share, x in zip(shares, xs, strict=True))
        match = _match_ces_constraint({0: sp.Eq(Y, A * inner ** (1 / s))})

        assert match is not None
        assert match.output == Y
        assert match.productivity == A
        assert set(dict(match.inputs).keys()) == set(xs)

    def test_detect_requires_objective(self):
        constraints = {0: canonical_ces(*sp.symbols("Y A x1 x2 alpha psi"))}
        assert CESBlock.detect(constraints, objective=None, identities=None) is False


class TestParameterizationVariants:
    """The matcher must accept the exponent and share spellings that appear across DSGE practice."""

    def test_direct_exponent_form(self):
        Y, A, x1, x2, alpha, rho = sp.symbols("Y A x1 x2 alpha rho")
        match = _match_ces_constraint({0: sp.Eq(Y, A * (alpha * x1**rho + (1 - alpha) * x2**rho) ** (1 / rho))})

        assert match is not None
        assert sp.simplify(match.exponent - rho) == 0
        assert dict(match.inputs) == {x1: alpha, x2: 1 - alpha}

    def test_acms_negative_exponent_form(self):
        Y, A, x1, x2, alpha, rho = sp.symbols("Y A x1 x2 alpha rho")
        rhs = A * (alpha * x1 ** (-rho) + (1 - alpha) * x2 ** (-rho)) ** (-1 / rho)
        match = _match_ces_constraint({0: sp.Eq(Y, rhs)})

        assert match is not None
        assert sp.simplify(match.exponent + rho) == 0

    def test_sigma_parameterization(self):
        Y, A, x1, x2, alpha, sigma = sp.symbols("Y A x1 x2 alpha sigma")
        match = _match_ces_constraint({0: canonical_ces(Y, A, x1, x2, alpha, sigma)})

        assert match is not None
        assert sp.simplify(match.exponent - (sigma - 1) / sigma) == 0

    def test_ratio_shares(self):
        Y, A, x1, x2, alpha, rho = sp.symbols("Y A x1 x2 alpha rho")
        share1 = alpha / (1 - alpha)
        match = _match_ces_constraint({0: sp.Eq(Y, A * (share1 * x1**rho + x2**rho) ** (1 / rho))})

        assert match is not None
        shares = dict(match.inputs)
        assert sp.simplify(shares[x1] - share1) == 0
        assert sp.simplify(shares[x2] - 1) == 0

    @pytest.mark.parametrize(
        "share1, share2",
        [
            (sp.Symbol("alpha"), sp.Symbol("beta")),
            (sp.Rational(3, 10), sp.Rational(7, 10)),
            (sp.S.One, sp.S.One),
        ],
        ids=["two-parameters", "numeric", "implicit-unit"],
    )
    def test_share_spellings(self, share1, share2):
        Y, A, x1, x2, rho = sp.symbols("Y A x1 x2 rho")
        match = _match_ces_constraint({0: sp.Eq(Y, A * (share1 * x1**rho + share2 * x2**rho) ** (1 / rho))})

        assert match is not None
        assert dict(match.inputs) == {x1: share1, x2: share2}

    def test_mixed_share_spellings(self):
        Y, A, x1, x2, alpha, psi = sp.symbols("Y A x1 x2 alpha psi")
        s = (psi - 1) / psi
        inner = alpha ** (1 / psi) * x1**s + (1 - alpha) * x2**s
        match = _match_ces_constraint({0: sp.Eq(Y, A * inner ** (1 / s))})

        assert match is not None
        assert dict(match.inputs) == {x1: alpha ** (1 / psi), x2: 1 - alpha}

    def test_no_leading_productivity(self):
        Y, x1, x2, alpha, rho = sp.symbols("Y x1 x2 alpha rho")
        match = _match_ces_constraint({0: sp.Eq(Y, (alpha * x1**rho + (1 - alpha) * x2**rho) ** (1 / rho))})

        assert match is not None
        assert match.productivity is None
        assert sp.simplify(match.exponent - rho) == 0
        assert dict(match.inputs) == {x1: alpha, x2: 1 - alpha}

    def test_no_leading_productivity_canonical_form(self):
        Y, x1, x2, alpha, psi = sp.symbols("Y x1 x2 alpha psi")
        canonical = canonical_ces(Y, sp.S.One, x1, x2, alpha, psi)
        match = _match_ces_constraint({0: canonical})

        assert match is not None
        assert match.productivity is None


CES_CONSTRAINT = (
    "Y[] = A[] * (alpha ^ (1 / psi) * K[] ^ ((psi - 1) / psi) + (1 - alpha) ^ (1 / psi) * L[] ^ ((psi - 1) / psi))"
    " ^ (psi / (psi - 1))"
)

FIRM_WITH_NON_INPUT_CONTROL = f"""
block FIRM
{{
    controls {{ K[], L[], B[]; }};
    objective {{ Pi[] = Y[] - r[] * K[] - w[] * L[] - B[] ^ 2 + q[] * B[]; }};
    constraints {{ {CES_CONSTRAINT} : mu[]; }};
    calibration {{ alpha = 0.35; psi = 0.8; }};
}};
"""

FIRM_WITH_DEFINITION_AND_GENERATED_MULTIPLIER = f"""
block FIRM
{{
    definitions {{ cost[] = r[] * K[] + w[] * L[]; }};
    controls {{ K[], L[]; }};
    objective {{ Pi[] = Y[] - cost[]; }};
    constraints {{ {CES_CONSTRAINT}; }};
    calibration {{ alpha = 0.35; psi = 0.8; }};
}};
"""


def test_control_outside_the_production_function_falls_back_to_lagrangian_derivative():
    """A control that is not a production input gets the generic chain-rule FOC, here d/dB of the objective."""
    block = load_gcn_string(FIRM_WITH_NON_INPUT_CONTROL).block_dict["FIRM"]
    assert isinstance(block, CESBlock)

    B, q = parsed_var("B", 0), parsed_var("q", 0)
    assert sp.simplify(block.system_equations[-1] - (q - 2 * B)) == 0


def test_constructing_without_matching_constraint_raises():
    Y, r, w, Pi = sp.symbols("Y r w Pi")
    with pytest.raises(RuntimeError, match="constructed without a matching CES constraint"):
        CESBlock(
            name="FIRM",
            objective={0: sp.Eq(Pi, Y - r)},
            constraints={1: sp.Eq(Y, r + w)},
            controls=[Y],
            multipliers={0: None, 1: None},
            equation_flags={0: {}, 1: {}},
        )


def test_definition_with_generated_multiplier_solves():
    block = load_gcn_string(FIRM_WITH_DEFINITION_AND_GENERATED_MULTIPLIER).block_dict["FIRM"]
    assert isinstance(block, CESBlock)
    assert len(block.system_equations) == 4
