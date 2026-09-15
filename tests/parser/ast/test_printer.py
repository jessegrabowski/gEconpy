import pytest

from gEconpy.parser.ast import (
    STEADY_STATE,
    T_MINUS_1,
    BinaryOp,
    Expectation,
    FunctionCall,
    GCNBlock,
    GCNDistribution,
    GCNEquation,
    GCNModel,
    Number,
    Operator,
    Parameter,
    T,
    TimeIndex,
    UnaryOp,
    Variable,
)
from gEconpy.parser.ast.printer import (
    print_block,
    print_distribution,
    print_equation,
    print_expression,
    print_model,
)
from gEconpy.parser.grammar.expressions import parse_expression
from gEconpy.parser.preprocessor import quick_parse


class TestPrintExpression:
    @pytest.mark.parametrize(
        ("node", "expected"),
        [
            (Number(value=42.0), "42"),
            (Number(value=3.14), "3.14"),
            (Parameter(name="alpha"), "alpha"),
            (Variable(name="C", time_index=T), "C[]"),
            (Variable(name="K", time_index=T_MINUS_1), "K[-1]"),
            (Variable(name="U", time_index=TimeIndex(1)), "U[1]"),
            (Variable(name="Y", time_index=STEADY_STATE), "Y[ss]"),
            (UnaryOp(op=Operator.NEG, operand=Parameter(name="x")), "-x"),
            (FunctionCall(func_name="log", args=(Variable(name="C"),)), "log(C[])"),
            (Expectation(expr=Variable(name="U", time_index=TimeIndex(1))), "E[][U[1]]"),
        ],
    )
    def test_atoms_and_wrappers(self, node, expected):
        assert print_expression(node) == expected

    @pytest.mark.parametrize(
        ("op", "symbol"),
        [
            (Operator.ADD, "+"),
            (Operator.SUB, "-"),
            (Operator.MUL, "*"),
            (Operator.DIV, "/"),
            (Operator.POW, "^"),
        ],
    )
    def test_binary_operators(self, op, symbol):
        node = BinaryOp(left=Parameter(name="a"), op=op, right=Parameter(name="b"))
        assert print_expression(node) == f"a {symbol} b"

    def test_lower_precedence_left_operand_is_parenthesized(self):
        node = BinaryOp(
            left=BinaryOp(left=Parameter(name="a"), op=Operator.ADD, right=Parameter(name="b")),
            op=Operator.MUL,
            right=Parameter(name="c"),
        )
        assert print_expression(node) == "(a + b) * c"

    def test_higher_precedence_left_operand_is_bare(self):
        node = BinaryOp(
            left=BinaryOp(left=Parameter(name="a"), op=Operator.MUL, right=Parameter(name="b")),
            op=Operator.ADD,
            right=Parameter(name="c"),
        )
        assert print_expression(node) == "a * b + c"


class TestPrintEquation:
    def test_simple_equation(self):
        eq = GCNEquation(lhs=Variable(name="Y"), rhs=Variable(name="C"))
        assert print_equation(eq) == "Y[] = C[]"

    def test_equation_with_lagrange(self):
        eq = GCNEquation(lhs=Variable(name="C"), rhs=Variable(name="Y"), lagrange_multiplier="lambda")
        assert print_equation(eq) == "C[] = Y[] : lambda[]"

    def test_calibrating_equation(self):
        eq = GCNEquation(lhs=Parameter(name="beta"), rhs=Number(value=0.99), calibrating_parameter="beta")
        assert print_equation(eq) == "beta = 0.99 -> beta"


class TestPrintDistribution:
    def test_simple_distribution(self):
        dist = GCNDistribution(parameter_name="alpha", dist_name="Beta", dist_kwargs={"alpha": 2, "beta": 5})
        assert print_distribution(dist) == "alpha ~ Beta(alpha=2, beta=5)"

    def test_distribution_with_initial_value(self):
        dist = GCNDistribution(
            parameter_name="alpha",
            dist_name="Beta",
            dist_kwargs={"alpha": 2, "beta": 5},
            initial_value=0.35,
        )
        assert print_distribution(dist) == "alpha ~ Beta(alpha=2, beta=5) = 0.35"

    def test_wrapped_distribution(self):
        dist = GCNDistribution(
            parameter_name="beta",
            dist_name="Beta",
            dist_kwargs={},
            wrapper_name="maxent",
            wrapper_kwargs={"lower": 0.95, "upper": 0.999},
            initial_value=0.99,
        )
        assert print_distribution(dist) == "beta ~ maxent(Beta(), lower=0.95, upper=0.999) = 0.99"


class TestPrintBlock:
    def test_simple_block(self):
        block = GCNBlock(name="TEST", identities=[GCNEquation(lhs=Variable(name="Y"), rhs=Variable(name="C"))])
        expected = "\n".join(
            [
                "block TEST",
                "{",
                "    identities",
                "    {",
                "        Y[] = C[];",
                "    };",
                "",
                "};",
            ]
        )
        assert print_block(block) == expected

    def test_block_with_controls_and_shocks_on_one_line_each(self):
        block = GCNBlock(
            name="HOUSEHOLD",
            controls=[Variable(name="C"), Variable(name="K", time_index=T_MINUS_1)],
            shocks=[Variable(name="epsilon")],
        )
        assert print_block(block, indent="  ") == "\n".join(
            [
                "block HOUSEHOLD",
                "{",
                "  controls",
                "  {",
                "    C[], K[-1];",
                "  };",
                "",
                "  shocks",
                "  {",
                "    epsilon[];",
                "  };",
                "",
                "};",
            ]
        )

    def test_block_with_calibration(self):
        block = GCNBlock(
            name="TEST",
            calibration=[
                GCNEquation(lhs=Parameter(name="alpha"), rhs=Number(value=0.35)),
                GCNDistribution(parameter_name="beta", dist_name="Beta", dist_kwargs={"alpha": 2, "beta": 5}),
                GCNEquation(
                    lhs=Variable(name="Y", time_index=STEADY_STATE),
                    rhs=Number(value=1.0),
                    calibrating_parameter="delta",
                ),
            ],
        )
        assert print_block(block) == "\n".join(
            [
                "block TEST",
                "{",
                "    calibration",
                "    {",
                "        alpha = 0.35;",
                "        beta ~ Beta(alpha=2, beta=5);",
                "        Y[ss] = 1 -> delta;",
                "    };",
                "",
                "};",
            ]
        )


class TestPrintModel:
    def test_model_with_options(self):
        model = GCNModel(options={"output logfile": True, "output LaTeX": False, "solver": "gensys"})
        assert print_model(model) == "\n".join(
            ["options", "{", "    output logfile = TRUE;", "    output LaTeX = FALSE;", "    solver = gensys;", "};"]
        )

    def test_model_with_tryreduce(self):
        model = GCNModel(tryreduce=["U[]", "TC[]"])
        assert print_model(model) == "\n".join(["tryreduce", "{", "    U[], TC[];", "};"])

    def test_model_with_assumptions_groups_by_assumption(self):
        model = GCNModel(assumptions={"K": {"positive": True}, "C": {"positive": True, "real": False}})
        expected = "\n".join(
            [
                "assumptions",
                "{",
                "    positive",
                "    {",
                "        C, K;",
                "    };",
                "};",
            ]
        )
        assert print_model(model) == expected

    def test_full_model(self):
        block = GCNBlock(
            name="HOUSEHOLD",
            controls=[Variable(name="C")],
            objective=[GCNEquation(lhs=Variable(name="U"), rhs=Variable(name="u"))],
            constraints=[GCNEquation(lhs=Variable(name="C"), rhs=Variable(name="Y"), lagrange_multiplier="lambda")],
        )
        model = GCNModel(blocks=[block], options={"output logfile": True}, tryreduce=["U[]"])

        assert print_model(model) == "\n\n".join(
            [
                "options\n{\n    output logfile = TRUE;\n};",
                "tryreduce\n{\n    U[];\n};",
                print_block(block),
            ]
        )

    def test_printed_model_reparses_to_same_ast(self):
        source = """
        options { output logfile = TRUE; solver = gensys; };
        assumptions { positive { C[], K[], alpha; }; };

        block HOUSEHOLD
        {
            definitions { u[] = log(C[]); };
            controls { C[], K[]; };
            objective { U[] = u[] + beta * E[][U[1]]; };
            constraints { C[] + K[] = Y[] : lambda[]; };
            identities { Y[] = A[] * K[-1] ^ alpha; };
            shocks { epsilon[]; };
            calibration
            {
                alpha ~ maxent(Beta(alpha=2, beta=5), lower=0.2, upper=0.5) = 0.35;
                beta = 0.99;
                Y[ss] / K[ss] = 0.36 -> delta;
            };
        };
        """
        model = quick_parse(source)
        assert quick_parse(print_model(model)) == model


class TestRoundTrip:
    @pytest.mark.parametrize(
        "expr_str",
        [
            "alpha",
            "C[]",
            "K[-1]",
            "Y[1]",
            "a + b",
            "a - b",
            "a * b",
            "a / b",
            "K[] ^ alpha",
            "log(C[])",
            "exp(x)",
            "A[] * K[-1] ^ alpha * L[] ^ (1 - alpha)",
            "u[] + beta * E[][U[1]]",
            "a - (b - c)",
            "a - (b + c)",
            "a / (b * c)",
            "a + (b + c)",
            "(a ^ b) ^ c",
            "a ^ b ^ c",
            "a ^ (b * c)",
            "-a ^ b",
            "a ^ -b",
            "a * -b",
            "-(a + b)",
            "log(a + b) / (c - d) ^ 2",
        ],
    )
    def test_printed_expression_reparses_to_same_tree(self, expr_str):
        node = parse_expression(expr_str)
        assert parse_expression(print_expression(node)) == node
