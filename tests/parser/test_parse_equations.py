import pytest
import sympy as sp

from gEconpy.classes.time_aware_symbol import TimeAwareSymbol
from gEconpy.parser import parse_equations, gEcon_parser

classification_cases = [
    "Y[] = C[] + I[]",
    "Y[] = A[] * C[] ^ alpha * L[] ^ ( 1 - alpha ) : mc[]",
    "K[ss] / L[ss] = 3 -> alpha",
]
classification_answers = [
    ["variable", "operator", "variable", "operator", "variable"],
    [
        "variable",
        "operator",
        "variable",
        "operator",
        "variable",
        "operator",
        "parameter",
        "operator",
        "variable",
        "operator",
        "operator",
        "number",
        "operator",
        "parameter",
        "operator",
        "lagrange_definition",
        "variable",
    ],
    [
        "variable",
        "operator",
        "variable",
        "operator",
        "number",
        "calibration_definition",
        "parameter",
    ],
]


@pytest.mark.parametrize(
    "case, expected_result",
    zip(classification_cases, classification_answers),
    ids=["simple", "complex", "calibration"],
)
def test_token_classification(case, expected_result):
    result = [parse_equations.token_classifier(token) for token in case.split()]
    assert result == expected_result


@pytest.mark.parametrize(
    "case, expected_result",
    [
        ("A[1]", "t1"),
        ("A[2]", "t2"),
        ("Happy[10]", "t10"),
        ("A[-1]", "tL1"),
        ("A[-2]", "tL2"),
        ("HAPPY[-10]", "tL10"),
        ("alpha_1[-1]", "tL1"),
        ("h1[2]", "t2"),
        ("A[ss]", "ss"),
    ],
    ids=[
        "t+1",
        "t+2",
        "t+10",
        "t-1",
        "t-2",
        "t-10",
        "numerical_suffix",
        "numerica_suffix_no_underscore",
        "steady_state",
    ],
)
def test_time_index_extraction(case, expected_result):
    result = parse_equations.extract_time_index(case)
    assert result == expected_result


@pytest.mark.parametrize(
    "case, expected_result",
    [
        ("A[]", TimeAwareSymbol("A", 0)),
        ("A[1]", TimeAwareSymbol("A", 1)),
        ("Happy[10]", TimeAwareSymbol("Happy", 10)),
        ("A[-1]", TimeAwareSymbol("A", -1)),
        ("A[-2]", TimeAwareSymbol("A", -2)),
        ("HAPPY[-10]", TimeAwareSymbol("HAPPY", -10)),
        ("alpha_1[-1]", TimeAwareSymbol("alpha_1", -1)),
        ("A[ss]", TimeAwareSymbol("A", 0).to_ss()),
        ("pi", sp.Symbol("pi")),
    ],
    ids=[
        "t",
        "t+1",
        "t+10",
        "t-1",
        "t-2",
        "t-10",
        "numerical_suffix",
        "steady_state",
        "parameter",
    ],
)
def test_single_symbol_to_sympy(case, expected_result):
    result = parse_equations.single_symbol_to_sympy(case)
    assert expected_result == result


@pytest.mark.parametrize(
    "case, expected_symbol, expected_name, expected_t",
    [
        (sp.Eq(sp.Symbol("x_t"), 0), sp.Symbol("x_t"), "x", 0),
        (sp.Eq(sp.Symbol("x_t1"), 0), sp.Symbol("x_{t+1}"), "x", 1),
        (sp.Eq(sp.Symbol("x_tL1"), 0), sp.Symbol("x_{t-1}"), "x", -1),
        (sp.Eq(sp.Symbol("x_t10"), 0), sp.Symbol("x_{t+10}"), "x", 10),
        (sp.Eq(sp.Symbol("x_tL10"), 0), sp.Symbol("x_{t-10}"), "x", -10),
        (sp.Eq(sp.Symbol("x_ss"), 0), sp.Symbol("x_ss"), "x", "ss"),
        (
            sp.Eq(sp.Symbol("This_is_a_variable_with_a_super_long_name_t10000"), 0),
            sp.Symbol("This_is_a_variable_with_a_super_long_name_{t+10000}"),
            "This_is_a_variable_with_a_super_long_name",
            10000,
        ),
        (
            sp.Eq(sp.Symbol("alpha_1_t10"), 0),
            sp.Symbol("alpha_1_{t+10}"),
            "alpha_1",
            10,
        ),
    ],
    ids=[
        "t",
        "t+1",
        "t-1",
        "t+10",
        "t-10",
        "steady_state",
        "long_name",
        "name_with_num",
    ],
)
def test_sympy_to_time_aware(case, expected_symbol, expected_name, expected_t):
    result = parse_equations.rename_time_indexes(case)
    result = next(iter([x for x in result.atoms() if isinstance(x, sp.Symbol)]))
    assert result == expected_symbol

    result = parse_equations.convert_symbols_to_time_symbols(case)
    result = next(iter([x for x in result.atoms() if isinstance(x, sp.Symbol)]))
    assert isinstance(result, TimeAwareSymbol)
    assert result.base_name == expected_name
    assert result.time_index == expected_t


def test_parameters_parsed_with_time_subscripts():
    test_file = """block SYSTEM_EQUATIONS
    {
        identities
        {
            #1. Labor supply
            W[] = sigma * C[] + phi * L[];

            #2. Euler Equation
            sigma / beta * (E[][C[1]] - C[]) = R_ss * E[][R[1]];

            #3. Law of motion of capital -- Timings have been changed to cause Gensys to fail
            K[] = (1 - delta) * K[] + delta * I[];

            #4. Production Function -- Timings have been changed to cause Gensys to fail
            Y[] = A[] + alpha * E[][K[1]] + (1 - alpha) * L[];

            #5. Demand for capital
            R[] = Y[] - K[-1];

            #6. Demand for labor
            W[] = Y[] - L[];

            #7. Equlibrium Condition
            Y_ss * Y[] = C_ss * C[] + I_ss * I[];

            #8. Productivity Shock
            A[] = rho_A * A[-1] + epsilon_A[];

        };

        shocks
        {
            epsilon_A[];
        };

        calibration
        {
            sigma = 2;
            phi = 1.5;
            alpha = 0.35;
            beta = 0.985;
            delta = 0.025;
            rho_A = 0.95;

            #P_ss = 1;
            R_ss = (1 / beta - (1 - delta));
            W_ss = (1 - alpha) ^ (1 / (1 - alpha)) * (alpha / R_ss) ^ (alpha / (1 - alpha));
            Y_ss = (R_ss / (R_ss - delta * alpha)) ^ (sigma / (sigma + phi)) *
                   ((1 - alpha) ^ (-phi) * (W_ss) ^ (1 + phi)) ^ (1 / (sigma + phi));
            K_ss = alpha * Y_ss / R_ss;

            I_tp1 = delta * K_ss;
            C_tm1 = Y_ss - I_ss;
            L_t = (1 - alpha) * Y_ss / W_ss;
        };
    };
    """

    parser_output, prior_dict = gEcon_parser.preprocess_gcn(test_file)
    block_dict, options, tryreduce, assumptions = (
        gEcon_parser.split_gcn_into_dictionaries(parser_output)
    )
    system = gEcon_parser.parsed_block_to_dict(block_dict["SYSTEM_EQUATIONS"])
    parser_output = parse_equations.build_sympy_equations(
        system["calibration"], assumptions
    )

    for eq, attrs in parser_output:
        assert not any(isinstance(x, TimeAwareSymbol) for x in eq.atoms())


def test_parse_equations_to_sympy():
    test_eq = "{definitions { u[] = log ( C[] ) + log ( L[] ) ; }; objective { U[] = u[] + beta * E[] [ U[1] ] ; };"
    test_eq += "calibration { L[ss] / K[ss] = 0.36 -> alpha ; }; };"

    answers = [
        sp.Eq(
            TimeAwareSymbol("u", 0),
            sp.log(TimeAwareSymbol("C", 0)) + sp.log(TimeAwareSymbol("L", 0)),
        ),
        sp.Eq(
            TimeAwareSymbol("U", 0),
            sp.Symbol("beta") * TimeAwareSymbol("U", 1) + TimeAwareSymbol("u", 0),
        ),
        sp.Eq(
            sp.Symbol("alpha"),
            TimeAwareSymbol("L", 0).to_ss() / TimeAwareSymbol("K", 0).to_ss() - 0.36,
        ),
    ]

    block_dict = gEcon_parser.parsed_block_to_dict(test_eq)

    for i, (component, equations) in enumerate(block_dict.items()):
        block_dict[component], flags = list(
            zip(*parse_equations.build_sympy_equations(equations))
        )
        eq1 = block_dict[component][0]
        eq2 = answers[i]

        assert ((eq1.lhs - eq1.rhs) - (eq2.lhs - eq2.rhs)).simplify() == 0
        assert not flags[0]["is_calibrating"] if i < 2 else flags[0]["is_calibrating"]
