import os
import pyparsing
import pytest

from gEconpy.parser import gEcon_parser
from gEconpy.parser.constants import DEFAULT_ASSUMPTIONS
from gEconpy.parser.gEcon_parser import (
    preprocess_gcn,
    extract_special_block,
    split_gcn_into_dictionaries,
    rebuild_eqs_from_parser_output,
    parsed_block_to_dict,
)
from gEconpy.parser.file_loaders import load_gcn
from pathlib import Path

ROOT = Path(__file__).parent.parent.absolute()


def test_remove_distributions_and_normal_parse(model):
    parser_output, prior_dict = preprocess_gcn(model)
    assert list(prior_dict.keys()) == [
        "epsilon[]",
        "alpha",
        "rho",
        "gamma",
        "sigma_epsilon",
    ]
    assert list(prior_dict.values()) == (
        [
            "Normal(mu=0, sigma=sigma_epsilon)",
            "Beta(mu=0.5, sigma=0.1) = 0.4",
            "Beta(mu=0.95, sigma=0.04) = 0.95",
            "HalfNormal(sigma=1) = 1.5",
            "InverseGamma(mu=0.1, sigma=0.01) = 0.01",
        ]
    )


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
    parser_output, _ = preprocess_gcn(test_file)
    with open(
        ROOT / "_resources" / "test_answer_strings" / "test_parse_gcn.txt"
    ) as file:
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
    parser_output, _ = preprocess_gcn(test_file)
    options = extract_special_block(parser_output, "options")
    tryreduce = extract_special_block(parser_output, "tryreduce")
    assert isinstance(options, dict)
    assert list(options.keys()) == [
        "output logfile",
        "output LaTeX",
        "output LaTeX landscape",
    ]
    assert list(options.values()) == [True, True, True]
    assert tryreduce == ["Div[]", "TC[]"]


def test_split_gcn_by_blocks():
    test_file = load_gcn(ROOT / "_resources" / "test_gcns" / "one_block_1.gcn")
    parser_output, _ = preprocess_gcn(test_file)
    with open(
        ROOT / "_resources" / "test_answer_strings" / "test_split_gcn_by_blocks.txt"
    ) as file:
        expected_result = file.read()
    block_dict, options, tryreduce, assumptions = split_gcn_into_dictionaries(
        parser_output
    )
    assert list(block_dict.keys()) == ["HOUSEHOLD"]
    assert options == {}
    assert tryreduce == []
    assert isinstance(assumptions, dict)
    assert block_dict["HOUSEHOLD"].strip() == expected_result.strip()


def test_equation_rebuilding():
    test_eq = "{Y[] = C[] + I[] + G[]; A[] ^ ( ( alpha + 1 ) / alpha ) - B[] / C[] * exp ( L[] ); };"
    parser_output, _ = preprocess_gcn(test_eq)
    parsed_block = (
        pyparsing.nestedExpr("{", "};").parseString(parser_output).asList()[0]
    )
    eqs = rebuild_eqs_from_parser_output(parsed_block)
    assert len(eqs) == 2
    assert " ".join(eqs[0]).strip() == test_eq.split(";")[0].replace("{", "").strip()
    assert " ".join(eqs[1]).strip() == test_eq.split(";")[1].replace("};", "").strip()


def test_parse_block_to_dict():
    test_eq = "{definitions { u[] = log ( C[] ) + log ( L[] ) ; };"
    test_eq += "objective { U[] = u[] + beta * E[] [ U[1] ] ; }; };"
    block_dict = parsed_block_to_dict(test_eq)
    assert list(block_dict.keys()) == ["definitions", "objective"]
    assert block_dict["definitions"] == [
        ["u[]", "=", "log", "(", "C[]", ")", "+", "log", "(", "L[]", ")"]
    ]
    assert block_dict["objective"] == [
        ["U[]", "=", "u[]", "+", "beta", "*", "E[]", "[", "U[1]", "]"]
    ]


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
