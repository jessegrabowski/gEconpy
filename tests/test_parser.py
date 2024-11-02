import os
import unittest

from pathlib import Path

import pyparsing
import pytest
import sympy as sp

from scipy.stats import invgamma, norm

from gEconpy.classes.time_aware_symbol import TimeAwareSymbol
from gEconpy.parser import file_loaders, gEcon_parser, parse_equations, parse_plaintext
from gEconpy.parser.constants import DEFAULT_ASSUMPTIONS
from gEconpy.parser.parse_distributions import CompositeDistribution

ROOT = Path(__file__).parent.absolute()


@pytest.fixture
def model():
    return file_loaders.load_gcn(os.path.join(ROOT, "Test GCNs/one_block_1_dist.gcn"))


def test_distribution_extraction_simple(model):
    test_str = "alpha ~ Normal(0, 1) = 0.5;"
    line, prior_dict = parse_plaintext.extract_distributions(test_str)
    assert line == "alpha = 0.5;"
    assert list(prior_dict.keys()) == ["alpha"]
    assert list(prior_dict.values()) == ["Normal(0, 1) = 0.5"]


def test_remove_distributions_and_normal_parse(model):
    parser_output, prior_dict = gEcon_parser.preprocess_gcn(model)

    assert list(prior_dict.keys()) == [
        "epsilon[]",
        "alpha",
        "rho",
        "gamma",
        "sigma_epsilon",
    ]
    assert list(prior_dict.values()) == (
        [
            "N(mean=0, sd=sigma_epsilon)",
            "Beta(mean=0.5, sd=0.1) = 0.4",
            "Beta(mean=0.95, sd=0.04) = 0.95",
            "HalfNormal(sigma=1) = 1.5",
            "Inv_Gamma(mean=0.1, sd=0.01) = 0.01",
        ]
    )


def test_remove_comment_line():
    test_string = """#This is a comment
                      Y[] = A[] + B[] + C[];"""
    expected_result = "Y[] = A[] + B[] + C[];"

    parsed_string = parse_plaintext.remove_comments(test_string)
    assert parsed_string.strip() == expected_result


def test_remove_end_of_line_comment():
    test_string = "Y[] = A[] + B[] + C[]; #here is a comment at the end"
    expected_result = "Y[] = A[] + B[] + C[]; "

    parsed_string = parse_plaintext.remove_comments(test_string)
    assert parsed_string == expected_result


@pytest.mark.parametrize(
    "test_string, expected_result",
    [
        (
            "Y[]=K[]^alpha*L[]^(1-alpha):P[];",
            "Y[] = K[] ^ alpha * L[] ^ ( 1 - alpha ) : P[] ;",
        ),
        ("K[ss]/L[ss]=3->alpha", "K[ss] / L[ss] = 3 -> alpha"),
    ],
    ids=["equation", "calibration"],
)
def test_add_space_to_equations(test_string, expected_result):
    result = parse_plaintext.add_spaces_around_operators(test_string)
    assert result == expected_result


parse_expectation_tests = [
    "E[][u[] + beta * U[1]];",
    "AMAZE[-1] + WILDE[] = E[][AMAZE[] + WILDE[1]];",
    "E[][A[] + 21];",
    "E[][21 + A[]];",
    "E[][A[1] + alpha];",
    "E[][A[1] + alpha] + sigma",
    "U[] = E[][u[] + beta * U[1]]",
]

parse_expectation_expected = [
    "E[] [ u[] + beta * U[1] ] ;",
    "AMAZE[-1] + WILDE[] = E[] [ AMAZE[] + WILDE[1] ] ;",
    "E[] [ A[] + 21 ] ;",
    "E[] [ 21 + A[] ] ;",
    "E[] [ A[1] + alpha ] ;",
    "E[] [ A[1] + alpha ] + sigma",
    "U[] = E[] [ u[] + beta * U[1] ]",
]


@pytest.mark.parametrize(
    "test_string, expected_result",
    zip(parse_expectation_tests, parse_expectation_expected),
    ids=[
        "bellman_rhs",
        "equation",
        "constant_on_right",
        "constant_on_left",
        "variable_in_expectation",
        "addition",
        "bellman",
    ],
)
def test_add_space_to_expectation_operator(test_string, expected_result):
    result = parse_plaintext.add_spaces_around_expectations(test_string)
    result = parse_plaintext.remove_extra_spaces(result)
    result = parse_plaintext.repair_special_tokens(result)

    assert result == expected_result


def test_parse_gcn():
    test_file = """block HOUSEHOLD
    {
        definitions
        {
            u[] = log(C[]) - log(L[]);
        };

        objective
        {
            U[] = u[] + beta * E[][U[1]];
        };

        controls
        {
            C[], L[];
        };

        constraints
        {
            C[] = w[] * L[];
        };

        calibration
        {
            beta = 0.99;
        };
    };
    """

    parser_output, _ = gEcon_parser.preprocess_gcn(test_file)
    with open(os.path.join(ROOT, "Test Answer Strings/test_parse_gcn.txt")) as file:
        expected_result = file.read()

    assert parser_output.strip() == expected_result.strip()


def test_block_extraction():
    test_file = """options
                    {
                        output logfile = TRUE;
                        output LaTeX = TRUE;
                        output LaTeX landscape = TRUE;
                    };

                    tryreduce
                    {
                        Div[], TC[];
                    };
                    """
    parser_output, _ = gEcon_parser.preprocess_gcn(test_file)

    options = gEcon_parser.extract_special_block(parser_output, "options")
    tryreduce = gEcon_parser.extract_special_block(parser_output, "tryreduce")

    assert isinstance(options, dict)
    assert list(options.keys()) == [
        "output logfile",
        "output LaTeX",
        "output LaTeX landscape",
    ]

    assert list(options.values()) == [True, True, True]
    assert tryreduce == ["Div[]", "TC[]"]


def test_block_deletion():
    test_file = """options
                    {
                        output logfile = TRUE;
                        output LaTeX = TRUE;
                        output LaTeX landscape = TRUE;
                    };

                    tryreduce
                    {
                        Div[], TC[];
                    };
                    """

    parser_output, _ = gEcon_parser.preprocess_gcn(test_file)
    result = parse_plaintext.delete_block(parser_output, "options")

    assert result.strip() == "tryreduce { Div[], TC[] ; };"

    result = parse_plaintext.delete_block(parser_output, "tryreduce")
    with open(
        os.path.join(ROOT, "Test Answer Strings/test_block_deletion.txt")
    ) as file:
        expected_result = file.read()

    assert result.strip() == expected_result.strip()


def test_split_gcn_by_blocks():
    test_file = file_loaders.load_gcn(os.path.join(ROOT, "Test GCNs/one_block_1.gcn"))
    parser_output, _ = gEcon_parser.preprocess_gcn(test_file)

    with open(
        os.path.join(ROOT, "Test Answer Strings/test_split_gcn_by_blocks.txt")
    ) as file:
        expected_result = file.read()

    block_dict, options, tryreduce, assumptions = (
        gEcon_parser.split_gcn_into_dictionaries(parser_output)
    )

    assert list(block_dict.keys()) == ["HOUSEHOLD"]

    assert options == {}
    assert tryreduce == []
    assert isinstance(assumptions, dict)

    assert block_dict["HOUSEHOLD"].strip() == expected_result.strip()


def test_equation_rebuilding():
    test_eq = "{Y[] = C[] + I[] + G[]; A[] ^ ( ( alpha + 1 ) / alpha ) - B[] / C[] * exp ( L[] ); };"

    parser_output, _ = gEcon_parser.preprocess_gcn(test_eq)
    parsed_block = (
        pyparsing.nestedExpr("{", "};").parseString(parser_output).asList()[0]
    )
    eqs = gEcon_parser.rebuild_eqs_from_parser_output(parsed_block)

    assert len(eqs) == 2
    assert " ".join(eqs[0]).strip() == test_eq.split(";")[0].replace("{", "").strip()
    assert " ".join(eqs[1]).strip() == test_eq.split(";")[1].replace("};", "").strip()


def test_parse_block_to_dict():
    test_eq = "{definitions { u[] = log ( C[] ) + log ( L[] ) ; };"
    test_eq += "objective { U[] = u[] + beta * E[] [ U[1] ] ; }; };"

    block_dict = gEcon_parser.parsed_block_to_dict(test_eq)

    assert list(block_dict.keys()) == ["definitions", "objective"]
    assert block_dict["definitions"] == [
        ["u[]", "=", "log", "(", "C[]", ")", "+", "log", "(", "L[]", ")"]
    ]
    assert block_dict["objective"] == [
        ["U[]", "=", "u[]", "+", "beta", "*", "E[]", "[", "U[1]", "]"]
    ]

    test_file = file_loaders.load_gcn(os.path.join(ROOT, "Test GCNs/rbc_2_block.gcn"))

    parser_output, _ = gEcon_parser.preprocess_gcn(test_file)
    block_dict, *_ = gEcon_parser.split_gcn_into_dictionaries(parser_output)
    household = gEcon_parser.parsed_block_to_dict(block_dict["HOUSEHOLD"])
    firm = gEcon_parser.parsed_block_to_dict(block_dict["FIRM"])

    assert household["controls"] == [["K[]", "C[]", "L[]", "I[]"]]
    assert firm["controls"] == [["K[-1]", "L[]"]]


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


def test_extract_assumption_blocks():
    test_file = """assumptions
                    {
                        positive
                        {
                            C[], K[], L[], A[], lambda[], w[], r[], mc[],
                            beta, delta, sigma_C, sigma_L, alpha;
                        };
                    };
                """

    parser_output, _ = gEcon_parser.preprocess_gcn(test_file)

    assumptions = gEcon_parser.extract_special_block(parser_output, "assumptions")
    assert all(v == {"real": True, "positive": True} for v in assumptions.values())


def test_invalid_assumptions_raise_error():
    test_file = """assumptions
        {
            random_words
            {
                L[], M[], P[];
            };
        };
        """
    parser_output, _ = gEcon_parser.preprocess_gcn(test_file)
    with pytest.raises(
        ValueError, match='Assumption "random_words" is not a valid Sympy assumption.'
    ):
        gEcon_parser.extract_special_block(parser_output, "assumptions")


def test_typo_in_assumptions_gives_suggestion():
    test_file = """assumptions
    {
        possitive
        {
            L[], M[], P[];
        };
    };
    """
    parser_output, _ = gEcon_parser.preprocess_gcn(test_file)
    with pytest.raises(
        ValueError,
        match='Assumption "possitive" is not a valid Sympy assumption. '
        'Did you mean "positive"?',
    ):
        gEcon_parser.extract_special_block(parser_output, "assumptions")


def test_default_assumptions_set_if_no_assumption_block():
    test_file = """
        block HOUSEHOLD
        {
            identities
            {
                C[] = 1;
            };
        };
    """

    parser_output, _ = gEcon_parser.preprocess_gcn(test_file)
    assumptions = gEcon_parser.extract_special_block(parser_output, "assumptions")

    assert assumptions["C"] == DEFAULT_ASSUMPTIONS


def test_defaults_removed_if_conflicting_with_user_spec():
    test_file = """
        assumptions
        {
            imaginary
            {
                C[];
            };
        };

        block HOUSEHOLD
        {
            identities
            {
                C[] = 1;
            };
        };
    """

    parser_output, _ = gEcon_parser.preprocess_gcn(test_file)
    assumptions = gEcon_parser.extract_special_block(parser_output, "assumptions")

    assert "real" not in assumptions["C"].keys()


def test_defaults_given_when_variable_subset_defined():
    test_file = """
         assumptions
         {
             negative
             {
                 C[];
             };
         };

         block HOUSEHOLD
         {
             identities
             {
                 C[] = 1;
                 L[] = 1;
             };
         };
     """

    parser_output, _ = gEcon_parser.preprocess_gcn(test_file)
    results = gEcon_parser.extract_special_block(parser_output, "assumptions")

    assert results["C"] == {"real": True, "negative": True}
    assert results["L"] == DEFAULT_ASSUMPTIONS


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


def test_composite_distribution():
    sigma_epsilon = invgamma(a=20)
    mu_epsilon = norm(loc=1, scale=0.1)

    d = CompositeDistribution(norm, loc=mu_epsilon, scale=sigma_epsilon)
    assert d.rv_params["loc"].mean() == mu_epsilon.mean()
    assert d.rv_params["loc"].std() == mu_epsilon.std()
    assert d.rv_params["scale"].mean() == sigma_epsilon.mean()
    assert d.rv_params["scale"].std() == sigma_epsilon.std()

    point_dict = {"loc": 0.1, "scale": 1, "epsilon": 1}
    assert d.logpdf(point_dict) == mu_epsilon.logpdf(0.1) + sigma_epsilon.logpdf(
        1
    ) + norm(loc=0.1, scale=1).logpdf(1)


def test_shock_block_with_multiple_distributions():
    test_file = """block TEST_BLOCK
                    {
                        shocks
                        {
                            epsilon_1[] ~ Normal(mu=0, sd=sigma_1);
                            epsilon_2[] ~ Normal(mu=0, sd=sigma_2);
                        };
                        calibration
                        {
                            sigma_1 ~ Invgamma(a=0.1, b=0.2) = 0.1;
                            sigma_2 ~ Invgamma(a=0.1, b=0.2) = 0.2;
                        };
                    };
                    """

    parser_output, prior_dict = gEcon_parser.preprocess_gcn(test_file)

    assert len(prior_dict) == 4
    assert list(prior_dict.keys()) == [
        "epsilon_1[]",
        "epsilon_2[]",
        "sigma_1",
        "sigma_2",
    ]

    dists = [
        "Normal(mu=0, sd=sigma_1)",
        "Normal(mu=0, sd=sigma_2)",
        "Invgamma(a=0.1, b=0.2) = 0.1",
        "Invgamma(a=0.1, b=0.2) = 0.2",
    ]

    for value, d in zip(prior_dict.values(), dists):
        assert value == d


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
