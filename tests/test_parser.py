import unittest
from collections import defaultdict

import pyparsing
import sympy as sp
from scipy.stats import invgamma, norm

from gEconpy.classes.time_aware_symbol import TimeAwareSymbol
from gEconpy.parser import file_loaders, gEcon_parser, parse_equations, parse_plaintext
from gEconpy.parser.constants import DEFAULT_ASSUMPTIONS
from gEconpy.parser.parse_distributions import CompositeDistribution


class ParserDistributionCases(unittest.TestCase):
    def setUp(self):
        self.model = file_loaders.load_gcn("Test GCNs/One_Block_Simple_1_w_Distributions.gcn")

    def test_distribution_extraction_simple(self):
        test_str = "alpha ~ Normal(0, 1) = 0.5;"
        line, prior_dict = parse_plaintext.extract_distributions(test_str)
        self.assertEqual(line, "alpha = 0.5;")
        self.assertEqual(list(prior_dict.keys()), ["alpha"])
        self.assertEqual(list(prior_dict.values()), ["Normal(0, 1)"])

    def test_remove_distributions_and_normal_parse(self):
        parser_output, prior_dict = gEcon_parser.preprocess_gcn(self.model)

        self.assertEqual(
            list(prior_dict.keys()),
            ["epsilon[]", "alpha", "rho", "gamma", "sigma_epsilon"],
        )
        self.assertEqual(
            list(prior_dict.values()),
            [
                "N(mean=0, sd=sigma_epsilon)",
                "Beta(mean=0.5, sd=0.1)",
                "Beta(mean=0.95, sd=0.04)",
                "HalfNormal(sigma=1)",
                "Inv_Gamma(mean=0.1, sd=0.01)",
            ],
        )


class ParserTestCases(unittest.TestCase):
    def test_remove_comment_line(self):
        test_string = """#This is a comment
                          Y[] = A[] + B[] + C[];"""
        expected_result = "Y[] = A[] + B[] + C[];"

        parsed_string = parse_plaintext.remove_comments(test_string)
        self.assertEqual(parsed_string.strip(), expected_result)

    def test_remove_end_of_line_comment(self):
        test_string = "Y[] = A[] + B[] + C[]; #here is a comment at the end"
        expected_result = "Y[] = A[] + B[] + C[]; "

        parsed_string = parse_plaintext.remove_comments(test_string)
        self.assertEqual(parsed_string, expected_result)

    def test_add_space_to_equations(self):
        tests = ["Y[]=K[]^alpha*L[]^(1-alpha):P[];", "K[ss]/L[ss]=3->alpha"]
        answers = [
            "Y[] = K[] ^ alpha * L[] ^ ( 1 - alpha ) : P[] ;",
            "K[ss] / L[ss] = 3 -> alpha",
        ]

        for case, expected_result in zip(tests, answers):
            result = parse_plaintext.add_spaces_around_operators(case)
            self.assertEqual(result, expected_result)

    def test_add_space_to_expectation_operator(self):
        test_cases = [
            "E[][u[] + beta * U[1]];",
            "AMAZE[-1] + WILDE[] = E[][AMAZE[] + WILDE[1]];",
            "E[][A[] + 21];",
            "E[][21 + A[]];",
            "E[][A[1] + alpha];",
            "E[][A[1] + alpha] + sigma",
            "U[] = E[][u[] + beta * U[1]]",
        ]

        answers = [
            "E[] [ u[] + beta * U[1] ] ;",
            "AMAZE[-1] + WILDE[] = E[] [ AMAZE[] + WILDE[1] ] ;",
            "E[] [ A[] + 21 ] ;",
            "E[] [ 21 + A[] ] ;",
            "E[] [ A[1] + alpha ] ;",
            "E[] [ A[1] + alpha ] + sigma",
            "U[] = E[] [ u[] + beta * U[1] ]",
        ]
        for case, answer in zip(test_cases, answers):
            result = parse_plaintext.add_spaces_around_expectations(case)
            result = parse_plaintext.remove_extra_spaces(result)
            result = parse_plaintext.repair_special_tokens(result)

            self.assertEqual(result, answer)

    def test_parse_gcn(self):
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
        with open("Test Answer Strings/test_parse_gcn.txt") as file:
            expected_result = file.read()

        self.assertEqual(parser_output, expected_result)

    def test_block_extraction(self):
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

        results = gEcon_parser.extract_special_block(parser_output, "options")
        results.update(gEcon_parser.extract_special_block(parser_output, "tryreduce"))

        self.assertEqual(list(results.keys()), ["options", "tryreduce"])
        self.assertIsInstance(results["options"], dict)
        self.assertEqual(
            list(results["options"].keys()),
            ["output logfile", "output LaTeX", "output LaTeX landscape"],
        )

        self.assertEqual(list(results["options"].values()), [True, True, True])

        self.assertEqual(results["tryreduce"], ["Div[]", "TC[]"])

    def test_block_deletion(self):
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

        self.assertEqual(result.strip(), "tryreduce { Div[], TC[] ; };")

        result = parse_plaintext.delete_block(parser_output, "tryreduce")
        with open("Test Answer Strings/test_block_deletion.txt") as file:
            expected_result = file.read()

        self.assertEqual(result.strip(), expected_result)

    def test_split_gcn_by_blocks(self):
        test_file = file_loaders.load_gcn("Test GCNs/One_Block_Simple_1.gcn")
        parser_output, _ = gEcon_parser.preprocess_gcn(test_file)

        with open("Test Answer Strings/test_split_gcn_by_blocks.txt") as file:
            expected_result = file.read()

        block_dict = gEcon_parser.split_gcn_into_block_dictionary(parser_output)

        self.assertEqual(
            list(block_dict.keys()),
            ["options", "tryreduce", "assumptions", "STEADY_STATE", "HOUSEHOLD"],
        )

        self.assertIs(block_dict["options"], None)
        self.assertIs(block_dict["tryreduce"], None)
        self.assertTrue(isinstance(block_dict["assumptions"], defaultdict))

        self.assertEqual(block_dict["HOUSEHOLD"].strip(), expected_result)

    def test_equation_rebuilding(self):
        test_eq = (
            "{Y[] = C[] + I[] + G[]; A[] ^ ( ( alpha + 1 ) / alpha ) - B[] / C[] * exp ( L[] ); };"
        )

        parser_output, _ = gEcon_parser.preprocess_gcn(test_eq)
        parsed_block = pyparsing.nestedExpr("{", "};").parseString(parser_output).asList()[0]
        eqs = gEcon_parser.rebuild_eqs_from_parser_output(parsed_block)

        self.assertEqual(len(eqs), 2)
        self.assertEqual(" ".join(eqs[0]).strip(), test_eq.split(";")[0].replace("{", "").strip())
        self.assertEqual(" ".join(eqs[1]).strip(), test_eq.split(";")[1].replace("};", "").strip())

    def test_parse_block_to_dict(self):
        test_eq = "{definitions { u[] = log ( C[] ) + log ( L[] ) ; };"
        test_eq += "objective { U[] = u[] + beta * E[] [ U[1] ] ; }; };"

        block_dict = gEcon_parser.parsed_block_to_dict(test_eq)
        self.assertEqual(list(block_dict.keys()), ["definitions", "objective"])
        self.assertEqual(
            block_dict["definitions"],
            [["u[]", "=", "log", "(", "C[]", ")", "+", "log", "(", "L[]", ")"]],
        )
        self.assertEqual(
            block_dict["objective"],
            [["U[]", "=", "u[]", "+", "beta", "*", "E[]", "[", "U[1]", "]"]],
        )

        test_file = file_loaders.load_gcn("Test GCNs/Two_Block_RBC_1.gcn")
        parser_output, _ = gEcon_parser.preprocess_gcn(test_file)
        block_dict = gEcon_parser.split_gcn_into_block_dictionary(parser_output)
        household = gEcon_parser.parsed_block_to_dict(block_dict["HOUSEHOLD"])
        firm = gEcon_parser.parsed_block_to_dict(block_dict["FIRM"])

        self.assertEqual(household["controls"], [["K[]", "C[]", "L[]", "I[]"]])
        self.assertEqual(firm["controls"], [["K[-1]", "L[]"]])

    def test_token_classification(self):
        tests = [
            "Y[] = C[] + I[]",
            "Y[] = A[] * C[] ^ alpha * L[] ^ ( 1 - alpha ) : mc[]",
            "K[ss] / L[ss] = 3 -> alpha",
        ]

        results = [
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

        for case, expected_result in zip(tests, results):
            result = [parse_equations.token_classifier(token) for token in case.split()]
            self.assertEqual(result, expected_result)

    def test_time_index_extraction(self):
        tests = [
            "A[1]",
            "A[2]",
            "Happy[10]",
            "A[-1]",
            "A[-2]",
            "HAPPY[-10]",
            "alpha_1[-1]",
            "A[ss]",
        ]

        results = ["t1", "t2", "t10", "tL1", "tL2", "tL10", "tL1", "ss"]

        for case, expected_result in zip(tests, results):
            result = parse_equations.extract_time_index(case)
            self.assertEqual(result, expected_result)

    def test_single_symbol_to_sympy(self):
        tests = [
            "A[]",
            "A[1]",
            "Happy[10]",
            "A[-1]",
            "A[-2]",
            "HAPPY[-10]",
            "alpha_1[-1]",
            "A[ss]",
            "pi",
        ]
        results = [
            TimeAwareSymbol("A", 0),
            TimeAwareSymbol("A", 1),
            TimeAwareSymbol("Happy", 10),
            TimeAwareSymbol("A", -1),
            TimeAwareSymbol("A", -2),
            TimeAwareSymbol("HAPPY", -10),
            TimeAwareSymbol("alpha_1", -1),
            TimeAwareSymbol("A", 0).to_ss(),
            sp.Symbol("pi"),
        ]

        for case, expected_result in zip(tests, results):
            result = parse_equations.single_symbol_to_sympy(case)
            self.assertEqual(expected_result, result)

    def test_sympy_rename_time_index(self):
        x_t, x_t1, x_tL1, x_10t, x_tL10, x_ss = sp.symbols(
            ["x_t", "x_t1", "x_tL1", "x_t10", "x_tL10", "x_ss"]
        )
        long_name_t, name_with_num = sp.symbols(
            ["This_is_a_variable_with_a_super_long_name_t10000", "alpha_1_t10"]
        )

        tests = [
            sp.Eq(x_t, 0),
            sp.Eq(x_t1, 0),
            sp.Eq(x_tL1, 0),
            sp.Eq(x_10t, 0),
            sp.Eq(x_tL10, 0),
            sp.Eq(x_ss, 0),
            sp.Eq(long_name_t, 0),
            sp.Eq(name_with_num, 0),
        ]

        answers = [
            sp.Symbol("x_t"),
            sp.Symbol("x_{t+1}"),
            sp.Symbol("x_{t-1}"),
            sp.Symbol("x_{t+10}"),
            sp.Symbol("x_{t-10}"),
            sp.Symbol("x_ss"),
            sp.Symbol("This_is_a_variable_with_a_super_long_name_{t+10000}"),
            sp.Symbol("alpha_1_{t+10}"),
        ]

        for case, expected_result in zip(tests, answers):
            result = parse_equations.rename_time_indexes(case)
            result = [x for x in result.atoms() if isinstance(x, sp.Symbol)][0]
            self.assertEqual(result, expected_result)

        eq_test = sp.Eq(x_t + x_t1 - x_tL1 * x_10t**x_tL10, x_ss - long_name_t / name_with_num)
        eq_answer = sp.Eq(
            answers[0] + answers[1] - answers[2] * answers[3] ** answers[4],
            answers[5] - answers[6] / answers[7],
        )

        self.assertEqual(eq_test, eq_answer)

    def test_convert_to_time_aware_equation(self):
        x_t, x_t1, x_tL1, x_10t, x_tL10, x_ss = sp.symbols(
            ["x_{t}", "x_{t+1}", "x_{t-1}", "x_{t+10}", "x_{t-10}", "x_ss"]
        )
        long_name_t, name_with_num = sp.symbols(
            ["This_is_a_variable_with_a_super_long_name_{t+10000}", "alpha_1_{t+10}"]
        )

        tests = [
            sp.Eq(x_t, 0),
            sp.Eq(x_t1, 0),
            sp.Eq(x_tL1, 0),
            sp.Eq(x_10t, 0),
            sp.Eq(x_tL10, 0),
            sp.Eq(x_ss, 0),
            sp.Eq(long_name_t, 0),
            sp.Eq(name_with_num, 0),
        ]

        answers = [
            ("x", 0),
            ("x", 1),
            ("x", -1),
            ("x", 10),
            ("x", -10),
            ("x", "ss"),
            ("This_is_a_variable_with_a_super_long_name", 10000),
            ("alpha_1", 10),
        ]

        for case, expected_results in zip(tests, answers):
            result = parse_equations.convert_symbols_to_time_symbols(case)
            result = [x for x in result.atoms() if isinstance(x, sp.Symbol)][0]
            self.assertIsInstance(result, TimeAwareSymbol)
            self.assertEqual(result.base_name, expected_results[0])
            self.assertEqual(result.time_index, expected_results[1])

    def test_extract_assumption_blocks(self):
        test_file = """positive
                        {
                            C[], K[], L[], A[], lambda[], w[], r[], mc[],
                            beta, delta, sigma_C, sigma_L, alpha;
                        };
                    """

        parser_output, _ = gEcon_parser.preprocess_gcn(test_file)

        results = gEcon_parser.extract_special_block(parser_output, "assumptions")
        self.assertTrue(list(results.keys()), ["positive"])

    def test_invalid_assumptions_raise_error(self):
        test_file = """assumptions
        {
            random_words
            {
                L[], M[], P[];
            };
        };
        """
        parser_output, _ = gEcon_parser.preprocess_gcn(test_file)
        self.assertRaises(
            ValueError, gEcon_parser.extract_special_block, parser_output, "assumptions"
        )

    def test_typo_in_assumptions_gives_suggestion(self):
        test_file = """assumptions
        {
            possitive
            {
                L[], M[], P[];
            };
        };
        """
        parser_output, _ = gEcon_parser.preprocess_gcn(test_file)
        try:
            gEcon_parser.extract_special_block(parser_output, "assumptions")
        except ValueError as e:
            self.assertEqual(
                str(e),
                'Assumption "possitive" is not a valid Sympy assumption. Did you mean "positive"?',
            )

    def test_default_assumptions_set_if_no_assumption_block(self):
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
        results = gEcon_parser.extract_special_block(parser_output, "assumptions")

        self.assertEqual(results["assumptions"]["C"], DEFAULT_ASSUMPTIONS)

    def test_defaults_removed_if_conflicting_with_user_spec(self):
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
        results = gEcon_parser.extract_special_block(parser_output, "assumptions")

        self.assertTrue("real" not in results["assumptions"]["C"].keys())

    def test_defaults_given_when_variable_subset_defined(self):
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

        self.assertEqual(results["assumptions"]["C"], {"real": True, "negative": True})
        self.assertEqual(results["assumptions"]["L"], DEFAULT_ASSUMPTIONS)

    def test_parse_equations_to_sympy(self):
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
            block_dict[component] = parse_equations.build_sympy_equations(equations)
            eq1 = block_dict[component][0]
            eq2 = answers[i]

            self.assertEqual(((eq1.lhs - eq1.rhs) - (eq2.lhs - eq2.rhs)).simplify(), 0)

    def test_composite_distribution(self):
        sigma_epsilon = invgamma(a=20)
        mu_epsilon = norm(loc=1, scale=0.1)

        d = CompositeDistribution(norm, loc=mu_epsilon, scale=sigma_epsilon)
        self.assertEqual(d.rv_params["loc"].mean(), mu_epsilon.mean())
        self.assertEqual(d.rv_params["loc"].std(), mu_epsilon.std())
        self.assertEqual(d.rv_params["scale"].mean(), sigma_epsilon.mean())
        self.assertEqual(d.rv_params["scale"].std(), sigma_epsilon.std())

        point_dict = {"loc": 0.1, "scale": 1, "epsilon": 1}
        self.assertEqual(
            d.logpdf(point_dict),
            mu_epsilon.logpdf(0.1) + sigma_epsilon.logpdf(1) + norm(loc=0.1, scale=1).logpdf(1),
        )

    def test_shock_block_with_multiple_distributions(self):
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

        self.assertEqual(len(prior_dict), 4)
        self.assertEqual(
            list(prior_dict.keys()),
            ["epsilon_1[]", "epsilon_2[]", "sigma_1", "sigma_2"],
        )
        dists = [
            "Normal(mu=0, sd=sigma_1)",
            "Normal(mu=0, sd=sigma_2)",
            "Invgamma(a=0.1, b=0.2)",
            "Invgamma(a=0.1, b=0.2)",
        ]

        for value, d in zip(prior_dict.values(), dists):
            self.assertEqual(value, d)


if __name__ == "__main__":
    unittest.main()
