from pathlib import Path

import pytest

from gEconpy.parser._legacy import gEcon_parser, parse_plaintext


def test_distribution_extraction_simple():
    test_str = "alpha ~ Normal(0, 1) = 0.5;"
    line, prior_dict = parse_plaintext.extract_distributions(test_str)
    assert line == "alpha = 0.5;"
    assert list(prior_dict.keys()) == ["alpha"]
    assert list(prior_dict.values()) == ["Normal(0, 1) = 0.5"]


def test_remove_comment_line():
    test_string = """#This is a comment\n                      Y[] = A[] + B[] + C[];"""
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
    zip(parse_expectation_tests, parse_expectation_expected, strict=False),
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

    ROOT = Path(__file__).parent.parent.absolute()
    with (ROOT / "_resources" / "test_answer_strings" / "test_block_deletion.txt").open() as file:
        expected_result = file.read()

    assert result.strip() == expected_result.strip()
