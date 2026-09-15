import re

import numpy as np
import pytest
import sympy as sp

from gEconpy.classes.time_aware_symbol import TimeAwareSymbol
from gEconpy.exceptions import (
    ControlVariableNotFoundException,
    DynamicCalibratingEquationException,
    MultipleObjectiveFunctionsException,
    OptimizationProblemNotDefinedException,
)
from gEconpy.model.block import Block
from gEconpy.model.block.cobb_douglas import CobbDouglasBlock
from gEconpy.parser import constants
from gEconpy.parser.loader import load_gcn_file, load_gcn_string
from gEconpy.parser.preprocessor import preprocess
from gEconpy.parser.transform.to_block import ast_block_to_block
from gEconpy.utilities import set_equality_equals_zero, unpack_keys_and_values
from tests.conftest import TEST_GCNS, parsed_symbol, parsed_symbols, parsed_var


def get_block_from_string(gcn_string: str, block_name: str = "HOUSEHOLD") -> Block:
    result = load_gcn_string(gcn_string)
    return result.block_dict[block_name]


def get_unsolved_block_from_string(gcn_string: str, block_name: str = "HOUSEHOLD") -> Block:
    result = preprocess(gcn_string, validate=True)
    for ast_block in result.ast.blocks:
        if ast_block.name == block_name:
            return ast_block_to_block(ast_block, result.assumptions)
    raise KeyError(f"Block {block_name} not found")


@pytest.fixture
def rng():
    return np.random.default_rng(0)


_MISSING_CONTROLS = """
    block HOUSEHOLD
    {
        objective
        {
            U[] = u[] + beta * E[][U[1]];
        };
    };
"""

_MISSING_OBJECTIVE = """
    block HOUSEHOLD
    {
        controls
        {
            K[], I[], C[], L[];
        };
    };
"""

_MULTIPLE_OBJECTIVES = """
    block HOUSEHOLD
    {
        objective
        {
            U[] = u[] + beta * E[][U[1]];
            C[] = a[] + b[];
        };
        controls
        {
            K[], I[], C[], L[];
        };
    };
"""

_CONTROL_NOT_FOUND = """
    block HOUSEHOLD
    {
        objective
        {
            U[] = u[] + beta * E[][U[1]];
        };
        controls
        {
            Z[];
        };
    };
"""

_DYNAMIC_CALIBRATION = """
    block HOUSEHOLD
    {
        calibration
        {
            Y[ss] / K[] = 0.33 -> alpha;
        };
    };
"""

_VARIABLE_IN_DETERMINISTIC_PARAMETER = """
    block HOUSEHOLD
    {
        calibration
        {
            beta = 0.99;
            alpha = beta * Y[];
        };
    };
"""


_NO_CONTINUATION_VALUE = """
    block HOUSEHOLD
    {
        controls { C[]; };
        objective { U[] = log(C[]) + beta * V[1]; };
        constraints { C[] = w[]; };
    };
"""

_SS_VARIABLE_WITHOUT_STEADY_STATE_BLOCK = """
    block HOUSEHOLD
    {
        identities { Y[] = A[] * L[]; };
        calibration
        {
            alpha = 0.5;
            phi = Y[ss] ^ 2 + alpha;
        };
    };
"""

_SS_VARIABLE_WITHOUT_ANALYTIC_VALUE = """
    block STEADY_STATE
    {
        identities { A[ss] = 1; };
    };

    block HOUSEHOLD
    {
        identities { Y[] = A[] * L[]; };
        calibration
        {
            alpha = 0.5;
            phi = Y[ss] ^ 2 + alpha;
        };
    };
"""


@pytest.mark.parametrize(
    "gcn_string, exception, match",
    [
        pytest.param(
            _MISSING_CONTROLS,
            OptimizationProblemNotDefinedException,
            "has an objective component but no controls component",
            marks=pytest.mark.xfail(strict=True, reason="The message names the present component as the missing one"),
        ),
        pytest.param(
            _MISSING_OBJECTIVE,
            OptimizationProblemNotDefinedException,
            "has a controls component but no objective component",
            marks=pytest.mark.xfail(strict=True, reason="The message names the present component as the missing one"),
        ),
        (_MULTIPLE_OBJECTIVES, MultipleObjectiveFunctionsException, "declares 2 objectives"),
        (_CONTROL_NOT_FOUND, ControlVariableNotFoundException, "Control variable 'Z_t' in block HOUSEHOLD"),
        (_DYNAMIC_CALIBRATION, DynamicCalibratingEquationException, "uses non-steady-state variables"),
        (_VARIABLE_IN_DETERMINISTIC_PARAMETER, ValueError, "cannot be functions of variables"),
        (_NO_CONTINUATION_VALUE, ValueError, "did not find the continuation value"),
        (_SS_VARIABLE_WITHOUT_STEADY_STATE_BLOCK, ValueError, "no STEADY_STATE block with analytic solutions"),
        (_SS_VARIABLE_WITHOUT_ANALYTIC_VALUE, ValueError, "without analytic solutions: Y_ss"),
    ],
    ids=[
        "missing-controls",
        "missing-objective",
        "multiple-objectives",
        "control-not-found",
        "dynamic-calibrating-equation",
        "variable-in-deterministic-parameter",
        "objective-without-continuation-value",
        "ss-variable-without-steady-state-block",
        "ss-variable-without-analytic-value",
    ],
)
def test_malformed_block_raises(gcn_string, exception, match):
    with pytest.raises(exception, match=match):
        get_block_from_string(gcn_string)


def test_block_parser_handles_empty_block():
    test_file = """
        block HOUSEHOLD
        {
            definitions
            {

            };
            identities
            {
                Y[] = C[] + I[];
            };
        };
        """
    block = get_block_from_string(test_file)
    assert not block.definitions


def test_lagrange_multiplier_in_objective_raises():
    test_file = """
        block HOUSEHOLD
        {
            definitions
            {
                u[] = log(C[]);
            };

            objective
            {
                U[] = u[] + beta * E[][U[1]] : lambda[];
            };

            controls
            {
                C[], K[];
            };

            constraints
            {
                Y[] = K[-1] ^ alpha;
                K[] = (1 - delta) * K[-1];
                C[] = r[] * K[-1];
            };

            calibration
            {
                alpha = 0.33;
                delta = 0.035;
                beta = 0.99;
            };
        };
        """

    block = get_unsolved_block_from_string(test_file)

    with pytest.raises(NotImplementedError):
        block.solve_optimization()


def test_invalid_decorator_raises():
    test_file = """
        block HOUSEHOLD
        {
            objective
            {
                @exclude
                U[] = u[] + beta * E[][U[1]] : lambda[];
            };

            controls
            {
                u[];
            };
        };
        """

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Equation Eq(U_t, beta*U_t+1 + u_t) in objective block of HOUSEHOLD has an invalid decorator: exclude."
        ),
    ):
        get_block_from_string(test_file)


@pytest.fixture
def block():
    result = load_gcn_file(TEST_GCNS / "one_block_2.gcn")
    return result.block_dict["HOUSEHOLD"]


class TestBlockCases:
    def test_string_repr(self, block):
        assert (
            str(block) == f"{block.name} Block of {block.n_equations} equations, initialized: "
            f"{block.initialized}, "
            f"solved: {block.system_equations is not None}"
        )

    def test_html_repr(self, block):
        html_string = block.__html_repr__()
        assert "Block: HOUSEHOLD" in html_string
        assert "<summary>Definitions</summary>" in html_string
        assert "<summary>Identities</summary>" in html_string
        assert "<summary>Objective</summary>" in html_string
        assert "<summary>Controls</summary>" in html_string
        assert "<summary>Calibration</summary>" in html_string
        assert "class='block-info'" in html_string

    def test_attributes_present(self, block):
        for component in constants.BLOCK_COMPONENTS:
            assert getattr(block, component.lower()) is not None

    def test_eq_number(self, block):
        assert block.n_equations == 14

    def test_variable_list_parsing(self, block):
        for variable in block.controls:
            assert isinstance(variable, TimeAwareSymbol)
        assert len(block.controls) == 5

        for variable in block.shocks:
            assert isinstance(variable, TimeAwareSymbol)
        assert len(block.shocks) == 1

    def test_lagrange_parsing(self, block):
        n_named_multipliers = sum(x is not None for x in block.multipliers.values())
        assert n_named_multipliers == 2
        assert block.multipliers[3] == parsed_var("lambda", 0)
        assert block.multipliers[4] == parsed_var("q", 0)

    def test_extract_discount_factor_on_Bellman_eq(self, block):
        df = block._get_discount_factor()
        assert df.name == "beta"

    def test_extract_discount_factor_on_static_eq(self, block):
        PI = parsed_var("Pi", 0)
        P = parsed_var("P", 0)
        Y = parsed_var("Y", 0)
        r = parsed_var("r", 0)
        w = parsed_var("w", 0)
        L = parsed_var("L", 0)
        K = parsed_var("K", 0)

        block.objective = {0: sp.Eq(PI, P * Y - r * K - w * L)}
        df = block._get_discount_factor()
        assert np.allclose(float(df), 1.0)

    def test_extract_discount_factor_on_lagged_eq(self, block):
        PI = parsed_var("Pi", 0)
        P = parsed_var("P", 0)
        Y = parsed_var("Y", 0)
        r = parsed_var("r", 0)
        w = parsed_var("w", 0)
        L = parsed_var("L", 0)
        K = parsed_var("K", -1)

        block.objective = {0: sp.Eq(PI, P * Y - r * K - w * L)}
        df = block._get_discount_factor()
        assert np.allclose(float(df), 1)

    def test_household_lagrangian_function(self, block):
        U = parsed_var("U", 1)
        Y = parsed_var("Y", 0, positive=True)
        C = parsed_var("C", 0, positive=True)
        I = parsed_var("I", 0, positive=True)
        K = parsed_var("K", 0, positive=True)
        L = parsed_var("L", 0, positive=True)
        A = parsed_var("A", 0, positive=True)
        lamb = parsed_var("lambda", 0)
        lamb_H_1 = parsed_var("lambda__H_1", 0)
        q = parsed_var("q", 0)

        alpha, beta, delta, theta, tau = parsed_symbols(["alpha", "beta", "delta", "theta", "tau"], positive=True)
        Theta, zeta = parsed_symbols(["Theta", "zeta"])

        utility = (C**theta * (1 - L) ** (1 - theta)) ** (1 - tau) / (1 - tau)
        mkt_clearing = C + I - Y
        production = Y - A * K**alpha * L ** (1 - alpha) - (Theta + zeta)
        law_motion_K = K - (1 - delta) * K.step_backward() - I

        answer = beta * U + utility - lamb * mkt_clearing - q * law_motion_K - lamb_H_1 * production

        lagrangian = block._build_lagrangian()
        assert (lagrangian - answer).simplify().evalf() == 0

    def test_Household_FOC(self, block, rng):
        block.solve_optimization(try_simplify=False)
        _, identities = unpack_keys_and_values(block.identities)
        _, objective = unpack_keys_and_values(block.objective)
        _, definitions = unpack_keys_and_values(block.definitions)
        sub_dict = {eq.lhs: eq.rhs for eq in definitions}
        objective = set_equality_equals_zero(objective[0].subs(sub_dict))

        assert all(set_equality_equals_zero(eq) in block.system_equations for eq in identities)
        assert objective in block.system_equations

        U = parsed_var("U", 1)
        Y = parsed_var("Y", 0, positive=True)
        C = parsed_var("C", 0, positive=True)
        I = parsed_var("I", 0, positive=True)
        K = parsed_var("K", 0, positive=True)
        L = parsed_var("L", 0, positive=True)
        A = parsed_var("A", 0, positive=True)
        lamb = parsed_var("lambda", 0)
        lamb_H_1 = parsed_var("lambda__H_1", 0)
        q = parsed_var("q", 0)
        eps = parsed_var("epsilon", 0)

        alpha, beta, delta, theta, tau, rho = parsed_symbols(
            ["alpha", "beta", "delta", "theta", "tau", "rho"], positive=True
        )
        Theta, zeta = parsed_symbols("Theta, zeta")

        all_variables = [
            U,
            U.step_backward(),
            Y,
            C,
            I,
            K,
            K.step_backward(),
            L,
            A,
            A.step_backward(),
            lamb,
            lamb_H_1,
            q,
            q.step_forward(),
            alpha,
            beta,
            delta,
            theta,
            tau,
            rho,
            eps,
            L.to_ss(),
            K.to_ss(),
        ]

        sub_dict = dict(zip(all_variables, rng.uniform(0, 1, size=len(all_variables)), strict=True))
        sub_dict[Theta] = 0
        sub_dict[zeta] = 0

        dL_dC = (C**theta * (1 - L) ** (1 - theta)) ** (-tau) * C ** (theta - 1) * (1 - L) ** (1 - theta) * theta - lamb

        dL_dL = (C**theta * (1 - L) ** (1 - theta)) ** (-tau) * C**theta * (1 - L) ** (-theta) * (
            1 - theta
        ) * -1 + lamb_H_1 * (1 - alpha) * A * K**alpha * L ** (-alpha)
        dL_dK = lamb_H_1 * A * alpha * K ** (alpha - 1) * L ** (1 - alpha) - q + beta * (1 - delta) * q.step_forward()
        dL_dI = -lamb + q

        subbed_system = [np.float32(eq.subs(sub_dict)) for eq in block.system_equations]

        for solution in [dL_dC, dL_dL, dL_dK, dL_dI]:
            assert np.float32(solution.subs(sub_dict)) in subbed_system

    def test_firm_block_lagrange_parsing(self):
        result = load_gcn_file(TEST_GCNS / "rbc_2_block.gcn")
        block = result.block_dict["FIRM"]

        Y = parsed_var("Y", 0)
        K = parsed_var("K", -1)
        L = parsed_var("L", 0)
        A = parsed_var("A", 0)
        r = parsed_var("r", 0)
        w = parsed_var("w", 0)
        P = parsed_var("P", 0)
        alpha, _rho = parsed_symbols(["alpha", "rho"])

        tc = -(r * K + w * L)
        prod = Y - A * K**alpha * L ** (1 - alpha)
        L = tc - P * prod

        assert (block._build_lagrangian() - L).simplify() == 0

    def test_get_param_dict_and_calibrating_equations(self, block):
        block.solve_optimization(try_simplify=False)

        _alpha, theta, beta, delta, tau, rho = parsed_symbols(
            ["alpha", "theta", "beta", "delta", "tau", "rho"], positive=True
        )
        K = parsed_var("K", 0, positive=True).to_ss()
        L = parsed_var("L", 0, positive=True).to_ss()

        answer = {theta: 0.357, beta: 1 / 1.01, delta: 0.02, tau: 2, rho: 0.95}
        assert all(key in block.param_dict for key in answer)

        for key in block.param_dict:
            np.testing.assert_allclose(answer[key], block.param_dict.values_to_float()[key])

        assert [str(p) for p in block.params_to_calibrate] == ["alpha"]

        actual_alpha = block.params_to_calibrate[0]
        actual_L_ss = next(s for s in block.calibrating_equations[0].free_symbols if str(s) == "L_ss")
        actual_K_ss = next(s for s in block.calibrating_equations[0].free_symbols if str(s) == "K_ss")

        # The GCN line ``L[ss] / K[ss] = 0.36 -> alpha`` stores alpha = 0.36 - L_ss / K_ss.
        calibrating_eqs = [actual_alpha - 0.36 + actual_L_ss / actual_K_ss]

        for i, eq in enumerate(calibrating_eqs):
            assert eq.simplify() == (block.params_to_calibrate[i] - block.calibrating_equations[i]).simplify()

    def test_deterministic_relationships(self, block):
        assert len(block.deterministic_relationships) == 2
        assert len(block.deterministic_params) == 2

        assert [x.name for x in block.deterministic_params] == ["Theta", "zeta"]
        answers = [3 + 1 / 1.01 * 0.95, -np.log(0.357)]
        for eq, answer in zip(block.deterministic_relationships, answers, strict=True):
            np.testing.assert_allclose(float(eq.subs(block.param_dict).evalf()), answer)

    def test_variable_list(self, block):
        block.solve_optimization(try_simplify=False)
        assert {x.base_name for x in block.variables} == {
            "A",
            "C",
            "I",
            "K",
            "L",
            "U",
            "Y",
            "lambda",
            "q",
            "lambda__H_1",
        }
        assert {x.base_name for x in block.shocks} == {"epsilon"}


def test_block_with_excluded_equation():
    result = load_gcn_file(TEST_GCNS / "rbc_with_excluded.gcn")
    block = result.block_dict["HOUSEHOLD"]
    block.solve_optimization()

    n_controls, n_objective, n_kept_constraints = 4, 1, 1
    assert len(block.system_equations) == n_controls + n_objective + n_kept_constraints


class TestBlockFromSympy:
    def test_from_sympy_creates_valid_block(self):
        C = parsed_var("C", 0)
        Y = parsed_var("Y", 0)

        identities = {0: sp.Eq(Y, C)}
        equation_flags = {0: {}}

        block = Block(
            name="TEST",
            identities=identities,
            equation_flags=equation_flags,
        )

        assert block.name == "TEST"
        assert block.initialized is True
        assert block.identities == identities

    def test_from_sympy_matches_dict_constructor(self):
        test_file = """
            block HOUSEHOLD
            {
                identities
                {
                    Y[] = C[] + I[];
                    K[] = I[] + (1 - delta) * K[-1];
                };

                calibration
                {
                    delta = 0.02;
                };
            };
            """

        loaded_block = get_block_from_string(test_file)
        loaded_block.solve_optimization()

        Y = parsed_var("Y", 0)
        C = parsed_var("C", 0)
        I = parsed_var("I", 0)
        K = parsed_var("K", 0)
        K_lag = parsed_var("K", -1)
        delta = parsed_symbol("delta")

        identities = {
            0: sp.Eq(Y, C + I),
            1: sp.Eq(K, I + (1 - delta) * K_lag),
        }
        calibration = {
            2: sp.Eq(delta, sp.Float(0.02)),
        }
        equation_flags = {0: {}, 1: {}, 2: {"is_calibrating": False}}

        new_block = Block(
            name="HOUSEHOLD",
            identities=identities,
            calibration=calibration,
            equation_flags=equation_flags,
        )
        new_block.solve_optimization()

        assert loaded_block.name == new_block.name
        assert len(loaded_block.system_equations) == len(new_block.system_equations)
        assert {v.base_name for v in loaded_block.variables} == {v.base_name for v in new_block.variables}

        assert set(loaded_block.param_dict.keys()) == set(new_block.param_dict.keys())
        for key in loaded_block.param_dict:
            assert float(loaded_block.param_dict[key]) == float(new_block.param_dict[key])

    def test_from_sympy_with_optimization_problem(self):
        U = parsed_var("U", 0)
        U_next = parsed_var("U", 1)
        C = parsed_var("C", 0)
        L = parsed_var("L", 0)
        w = parsed_var("w", 0)
        lambda_ = parsed_var("lambda", 0)
        beta = parsed_symbol("beta")

        objective = {0: sp.Eq(U, sp.log(C) - L + beta * U_next)}
        constraints = {1: sp.Eq(C, w * L)}
        controls = [C, L]
        multipliers = {0: None, 1: lambda_}
        equation_flags = {0: {}, 1: {}}

        block = Block(
            name="HOUSEHOLD",
            objective=objective,
            constraints=constraints,
            controls=controls,
            multipliers=multipliers,
            equation_flags=equation_flags,
        )

        assert block.initialized is True
        assert block.controls == controls
        assert block.objective == objective
        assert block.constraints == constraints

        block.solve_optimization()

        foc_C = 1 / C - lambda_
        foc_L = -1 + lambda_ * w
        assert len(block.system_equations) == 4
        assert all(
            sp.simplify(foc - expected) == 0
            for foc, expected in zip(block.system_equations[-2:], [foc_C, foc_L], strict=True)
        )

    def test_generated_multiplier_is_eliminated_by_simplification(self):
        """An unnamed multiplier that a two-term FOC pins down is solved out and its slot reset to None."""
        U = parsed_var("U", 0)
        U_next = parsed_var("U", 1)
        C = parsed_var("C", 0)
        L = parsed_var("L", 0)
        w = parsed_var("w", 0)
        beta = parsed_symbol("beta")

        block = Block(
            name="HOUSEHOLD",
            objective={0: sp.Eq(U, sp.log(C) - L + beta * U_next)},
            constraints={1: sp.Eq(C, w * L)},
            controls=[C, L],
            multipliers={0: None, 1: None},
            equation_flags={0: {}, 1: {}},
        )
        block.solve_optimization(try_simplify=False)
        generated = block.multipliers[1]
        assert generated.base_name == "lambda__H_1"
        assert any(generated in eq.atoms() for eq in block.system_equations)

        block.simplify_system_equations()

        assert block.multipliers[1] is None
        assert generated in block.eliminated_variables
        assert not any(generated in eq.atoms() for eq in block.system_equations)
        assert len(block.system_equations) == 3
        assert sp.simplify(block.system_equations[-1] - (w / C - 1)) == 0


def test_lagged_definition_produces_derivative_in_foc():
    """The bond Euler equation must carry the derivative of the lagged risk-premium definition."""
    result = load_gcn_file(TEST_GCNS / "debt_elastic_premium.gcn")
    block = result.block_dict["HOUSEHOLD"]

    all_atoms = set()
    for eq in block.system_equations:
        all_atoms |= eq.atoms()

    variables = sorted([a for a in all_atoms if isinstance(a, (TimeAwareSymbol, sp.Symbol))], key=str)
    ns = {str(a): a for a in variables}

    B = ns["B_t"]
    Y = ns["Y_t"]
    R_star = ns["R_star_t"]
    lam = ns["lambda_t"]
    lam_lead = ns["lambda_t+1"]
    beta_s = ns["beta"]
    phi_B = ns["phi_B"]
    B_bar = ns["B_bar"]

    Phi_B = sp.exp(-phi_B * (-B_bar + B / Y))

    expected_bond_foc = -beta_s * lam_lead * R_star * Phi_B * (phi_B * B / Y - 1) - lam
    bond_foc = block.system_equations[-1]
    bond_foc = bond_foc.collect(R_star).collect(Phi_B)
    assert bond_foc == expected_bond_foc


def test_ss_variable_in_calibration_resolves_to_deterministic_param():
    gcn = """
    block STEADY_STATE
    {
        identities
        {
            Y[ss] = Y_bar;
        };
    };

    block HOUSEHOLD
    {
        identities
        {
            Y[] = A[] * L[];
        };

        calibration
        {
            alpha = 0.5;
            phi = Y[ss] ^ 2 + alpha;
            Y_bar = 0.8;
        };
    };
    """
    result = load_gcn_string(gcn)
    block = result.block_dict["HOUSEHOLD"]

    alpha = parsed_symbol("alpha")
    Y_bar = parsed_symbol("Y_bar")
    phi = parsed_symbol("phi")

    assert phi in block.deterministic_dict
    assert block.deterministic_dict[phi] == Y_bar**2 + alpha


@pytest.mark.parametrize(
    "gcn_file",
    ["rbc_2_block.gcn", "rbc_2_block_minimize.gcn"],
    ids=["maximize-profit", "minimize-cost"],
)
def test_firm_focs_match_closed_form(gcn_file, rng):
    result = load_gcn_file(TEST_GCNS / gcn_file)
    firm_block = result.block_dict["FIRM"]

    Y = parsed_var("Y", 0)
    TC = parsed_var("TC", 0)
    K = parsed_var("K", -1)
    L = parsed_var("L", 0)
    A = parsed_var("A", 0)
    r = parsed_var("r", 0)
    w = parsed_var("w", 0)
    P = parsed_var("P", 0)
    epsilon = parsed_var("epsilon_A", 0)
    alpha, rho = parsed_symbols(["alpha", "rho_A"])

    # The chain-rule FOCs of the general Block and the closed-form FOCs of CobbDouglasBlock only agree at points
    # where the production residual is zero, so Y is computed from the constraint while the rest are sampled.
    rest_vars = [TC, K, L, A, A.step_backward(), P, r, w, alpha, rho, epsilon]
    sub_dict = dict(zip(rest_vars, rng.uniform(0.1, 1, size=len(rest_vars)), strict=True))
    sub_dict[Y] = float(sub_dict[A] * sub_dict[K] ** sub_dict[alpha] * sub_dict[L] ** (1 - sub_dict[alpha]))

    expected_dL_dK = -r + P * A * alpha * K ** (alpha - 1) * L ** (1 - alpha)
    expected_dL_dL = -w + P * A * (1 - alpha) * K**alpha * L ** (-alpha)

    subbed_system = [float(eq.subs(sub_dict)) for eq in firm_block.system_equations]

    for expected_foc in [float(expected_dL_dK.subs(sub_dict)), float(expected_dL_dL.subs(sub_dict))]:
        assert any(abs(actual - expected_foc) < 1e-10 for actual in subbed_system), (
            f"Expected FOC value {expected_foc} not found in system_equations (values: {subbed_system})"
        )


FIRM_WITH_COST_DEFINITION = """
block FIRM
{
    definitions
    {
        cost[] = r[] * K[] + w[] * L[];
    };

    controls
    {
        K[], L[];
    };

    objective
    {
        Pi[] = Y[] - cost[];
    };

    constraints
    {
        Y[] = A[] * K[] ^ alpha * L[] ^ (1 - alpha) : mu[];
    };

    calibration
    {
        alpha = 0.35;
    };
};
"""


def test_closed_form_focs_substitute_definitions():
    """The closed-form FOC must see the cost written in a definition, as the generic Lagrangian does."""
    closed_form = get_unsolved_block_from_string(FIRM_WITH_COST_DEFINITION, block_name="FIRM")
    assert isinstance(closed_form, CobbDouglasBlock)

    generic = Block(
        name=closed_form.name,
        definitions=closed_form.definitions,
        controls=closed_form.controls,
        objective=closed_form.objective,
        constraints=closed_form.constraints,
        calibration=closed_form.calibration,
        multipliers=dict(closed_form.multipliers),
        equation_flags=closed_form.equation_flags,
    )
    closed_form.solve_optimization(try_simplify=False)
    generic.solve_optimization(try_simplify=False)

    Y, A, K, L = parsed_var("Y", 0), parsed_var("A", 0), parsed_var("K", 0), parsed_var("L", 0)
    alpha = parsed_symbol("alpha")
    on_constraint = {Y: A * K**alpha * L ** (1 - alpha)}

    for closed_form_foc, generic_foc in zip(
        closed_form.system_equations[-2:], generic.system_equations[-2:], strict=True
    ):
        assert sp.simplify((closed_form_foc - generic_foc).subs(on_constraint)) == 0


def test_minimize_and_maximize_on_same_equation_raises():
    gcn = """
    block FIRM
    {
        controls { L[]; };
        objective
        {
            @minimize
            @maximize
            TC[] = w[] * L[];
        };
        constraints { Y[] = A[] * L[] : mc[]; };
    };
    """
    with pytest.raises(ValueError, match="both @minimize and @maximize"):
        load_gcn_string(gcn)


_CONSTRAINT_BLOCK = """
    block B
    {{
        controls {{ C[]; }};
        objective {{ U[] = C[]; }};
        constraints {{ @{tag} C[] = 1 : lambda[]; }};
    }};
"""

_IDENTITY_BLOCK = """
    block B
    {{
        identities {{ @{tag} Y[] = C[]; }};
    }};
"""


@pytest.mark.parametrize(
    "tag, gcn_template",
    [
        ("minimize", _CONSTRAINT_BLOCK),
        ("minimize", _IDENTITY_BLOCK),
        ("maximize", _CONSTRAINT_BLOCK),
        ("maximize", _IDENTITY_BLOCK),
    ],
    ids=[
        "minimize-on-constraint",
        "minimize-on-identity",
        "maximize-on-constraint",
        "maximize-on-identity",
    ],
)
def test_optimization_tag_on_wrong_component_raises(tag, gcn_template):
    with pytest.raises(ValueError, match=f"invalid decorator: {tag}"):
        load_gcn_string(gcn_template.format(tag=tag))
