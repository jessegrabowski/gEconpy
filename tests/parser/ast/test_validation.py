from collections.abc import Iterable

from gEconpy.parser.ast import (
    STEADY_STATE,
    T_MINUS_1,
    BinaryOp,
    GCNBlock,
    GCNDistribution,
    GCNEquation,
    GCNModel,
    Number,
    Operator,
    Parameter,
    Variable,
)
from gEconpy.parser.ast.validation import (
    check_undefined_parameters,
    check_undefined_variables,
    full_validation,
    validate_block,
    validate_equation,
    validate_model,
)
from gEconpy.parser.errors import GCNParseError


def _messages(errors: Iterable[GCNParseError]) -> list[str]:
    return [str(e) for e in errors]


class TestValidateBlock:
    def test_empty_block_no_errors(self):
        errors = validate_block(GCNBlock(name="TEST"))
        assert not errors.has_errors

    def test_duplicate_control_variable(self):
        block = GCNBlock(name="TEST", controls=[Variable(name="C"), Variable(name="L"), Variable(name="C")])
        errors = validate_block(block)
        assert errors.has_errors
        assert _messages(errors) == [
            "[E101] Duplicate control variable 'C' in block TEST",
            "[W002] Block TEST has controls but no objective function",
        ]

    def test_duplicate_shock_variable(self):
        block = GCNBlock(name="TEST", shocks=[Variable(name="epsilon_A"), Variable(name="epsilon_A")])
        errors = validate_block(block)
        assert errors.has_errors
        assert _messages(errors) == ["[E101] Duplicate shock variable 'epsilon_A' in block TEST"]

    def test_name_repeated_three_times_is_reported_twice(self):
        block = GCNBlock(name="TEST", shocks=[Variable(name="e"), Variable(name="e"), Variable(name="e")])
        assert _messages(validate_block(block)) == ["[E101] Duplicate shock variable 'e' in block TEST"] * 2

    def test_duplicate_calibration_parameter(self):
        block = GCNBlock(
            name="TEST",
            calibration=[
                GCNEquation(lhs=Parameter(name="alpha"), rhs=Number(value=0.3)),
                GCNEquation(lhs=Parameter(name="alpha"), rhs=Number(value=0.4)),
            ],
        )
        errors = validate_block(block)
        assert errors.has_errors
        assert _messages(errors) == ["[E101] Duplicate calibration for parameter 'alpha' in block TEST"]

    def test_controls_without_objective_warning(self):
        block = GCNBlock(name="TEST", controls=[Variable(name="C")])
        errors = validate_block(block)
        assert not errors.has_errors
        assert _messages(errors.warnings) == ["[W002] Block TEST has controls but no objective function"]

    def test_objective_without_constraints_warning(self):
        block = GCNBlock(name="TEST", objective=[GCNEquation(lhs=Variable(name="U"), rhs=Variable(name="u"))])
        errors = validate_block(block)
        assert not errors.has_errors
        assert _messages(errors.warnings) == ["[W003] Block TEST has objective but no constraints"]

    def test_valid_optimization_block_no_errors(self):
        block = GCNBlock(
            name="HOUSEHOLD",
            controls=[Variable(name="C")],
            objective=[GCNEquation(lhs=Variable(name="U"), rhs=Variable(name="u"))],
            constraints=[GCNEquation(lhs=Variable(name="C"), rhs=Variable(name="Y"))],
        )
        errors = validate_block(block)
        assert not list(errors)


class TestValidateModel:
    def test_empty_model_no_errors(self):
        errors = validate_model(GCNModel())
        assert not errors.has_errors

    def test_duplicate_block_names(self):
        model = GCNModel(blocks=[GCNBlock(name="HOUSEHOLD"), GCNBlock(name="HOUSEHOLD")])
        errors = validate_model(model)
        assert errors.has_errors
        assert _messages(errors) == ["[E100] Duplicate block name: HOUSEHOLD"]

    def test_parameter_defined_in_multiple_blocks(self):
        model = GCNModel(
            blocks=[
                GCNBlock(
                    name="HOUSEHOLD",
                    calibration=[GCNEquation(lhs=Parameter(name="alpha"), rhs=Number(value=0.3))],
                ),
                GCNBlock(
                    name="FIRM",
                    calibration=[GCNEquation(lhs=Parameter(name="alpha"), rhs=Number(value=0.35))],
                ),
            ]
        )
        errors = validate_model(model)
        assert errors.has_errors
        assert _messages(errors) == ["[E101] Parameter 'alpha' defined in multiple blocks: HOUSEHOLD, FIRM"]

    def test_valid_multi_block_model(self):
        model = GCNModel(
            blocks=[
                GCNBlock(
                    name="HOUSEHOLD",
                    calibration=[GCNEquation(lhs=Parameter(name="beta"), rhs=Number(value=0.99))],
                ),
                GCNBlock(
                    name="FIRM",
                    calibration=[GCNEquation(lhs=Parameter(name="alpha"), rhs=Number(value=0.35))],
                ),
            ]
        )
        errors = validate_model(model)
        assert not errors.has_errors


class TestValidateEquation:
    def test_returns_empty_collector(self):
        eq = GCNEquation(lhs=Variable(name="Y"), rhs=Variable(name="C"), calibrating_parameter="alpha")
        assert not list(validate_equation(eq))


class TestCheckUndefinedVariables:
    def test_all_variables_defined(self):
        block = GCNBlock(
            name="TEST",
            controls=[Variable(name="C"), Variable(name="Y")],
            identities=[GCNEquation(lhs=Variable(name="Y"), rhs=Variable(name="C"))],
        )
        errors = check_undefined_variables(GCNModel(blocks=[block]))
        assert not list(errors)

    def test_undefined_variable_warning(self):
        block = GCNBlock(
            name="TEST",
            controls=[Variable(name="C")],
            identities=[
                GCNEquation(
                    lhs=Variable(name="Y"),
                    rhs=BinaryOp(left=Variable(name="C"), op=Operator.ADD, right=Variable(name="I")),
                )
            ],
        )
        errors = check_undefined_variables(GCNModel(blocks=[block]))
        assert _messages(errors) == ["Variable 'I' is used but not defined"]

    def test_external_variables_not_flagged(self):
        block = GCNBlock(name="TEST", identities=[GCNEquation(lhs=Variable(name="Y"), rhs=Variable(name="X"))])
        errors = check_undefined_variables(GCNModel(blocks=[block]), external_variables={"X"})
        assert not list(errors)


class TestCheckUndefinedParameters:
    def test_all_parameters_calibrated(self):
        block = GCNBlock(
            name="TEST",
            identities=[
                GCNEquation(
                    lhs=Variable(name="Y"),
                    rhs=BinaryOp(left=Parameter(name="alpha"), op=Operator.MUL, right=Variable(name="K")),
                )
            ],
            calibration=[GCNEquation(lhs=Parameter(name="alpha"), rhs=Number(value=0.3))],
        )
        errors = check_undefined_parameters(GCNModel(blocks=[block]))
        assert not list(errors)

    def test_uncalibrated_parameter_warning(self):
        block = GCNBlock(
            name="TEST",
            identities=[
                GCNEquation(
                    lhs=Variable(name="Y"),
                    rhs=BinaryOp(left=Parameter(name="alpha"), op=Operator.MUL, right=Variable(name="K")),
                )
            ],
        )
        errors = check_undefined_parameters(GCNModel(blocks=[block]))
        assert _messages(errors) == ["Parameter 'alpha' is used but not calibrated"]

    def test_external_parameters_not_flagged(self):
        block = GCNBlock(name="TEST", identities=[GCNEquation(lhs=Variable(name="Y"), rhs=Parameter(name="beta"))])
        errors = check_undefined_parameters(GCNModel(blocks=[block]), external_parameters={"beta"})
        assert not list(errors)

    def test_calibrating_equation_counts_as_defined(self):
        block = GCNBlock(
            name="TEST",
            identities=[
                GCNEquation(
                    lhs=Variable(name="Y"),
                    rhs=BinaryOp(left=Parameter(name="alpha"), op=Operator.MUL, right=Variable(name="K")),
                )
            ],
            calibration=[
                GCNEquation(
                    lhs=Variable(name="Y", time_index=STEADY_STATE),
                    rhs=Number(value=0.36),
                    calibrating_parameter="alpha",
                )
            ],
        )
        errors = check_undefined_parameters(GCNModel(blocks=[block]))
        assert not list(errors)

    def test_distribution_calibration_counts_as_defined(self):
        block = GCNBlock(
            name="TEST",
            calibration=[
                GCNDistribution(
                    parameter_name="alpha",
                    dist_name="Beta",
                    dist_kwargs={"alpha": 2, "beta": 5},
                    initial_value=0.35,
                )
            ],
            identities=[GCNEquation(lhs=Variable(name="Y"), rhs=Parameter(name="alpha"))],
        )
        errors = check_undefined_parameters(GCNModel(blocks=[block]))
        assert not list(errors)


class TestFullValidation:
    def test_complete_valid_model(self):
        block = GCNBlock(
            name="HOUSEHOLD",
            controls=[Variable(name="C"), Variable(name="K")],
            objective=[GCNEquation(lhs=Variable(name="U"), rhs=Variable(name="u"))],
            constraints=[
                GCNEquation(
                    lhs=Variable(name="C"),
                    rhs=BinaryOp(left=Variable(name="Y"), op=Operator.SUB, right=Variable(name="I")),
                )
            ],
            definitions=[GCNEquation(lhs=Variable(name="u"), rhs=Variable(name="C"))],
            identities=[
                GCNEquation(
                    lhs=Variable(name="Y"),
                    rhs=BinaryOp(
                        left=Parameter(name="alpha"),
                        op=Operator.MUL,
                        right=Variable(name="K", time_index=T_MINUS_1),
                    ),
                )
            ],
            calibration=[GCNEquation(lhs=Parameter(name="alpha"), rhs=Number(value=0.35))],
        )
        errors = full_validation(GCNModel(blocks=[block]), external_variables={"I"})
        assert not list(errors)

    def test_reports_block_then_model_then_variable_then_parameter_issues(self):
        model = GCNModel(
            blocks=[
                GCNBlock(name="BLOCK_A", controls=[Variable(name="C"), Variable(name="C")]),
                GCNBlock(
                    name="BLOCK_A",
                    identities=[
                        GCNEquation(
                            lhs=Variable(name="Y"),
                            rhs=BinaryOp(left=Parameter(name="beta"), op=Operator.MUL, right=Variable(name="X")),
                        )
                    ],
                    calibration=[GCNEquation(lhs=Parameter(name="alpha"), rhs=Number(value=0.3))],
                ),
            ]
        )
        assert _messages(full_validation(model)) == [
            "[E101] Duplicate control variable 'C' in block BLOCK_A",
            "[W002] Block BLOCK_A has controls but no objective function",
            "[E100] Duplicate block name: BLOCK_A",
            "Variable 'X' is used but not defined",
            "Parameter 'beta' is used but not calibrated",
        ]


class TestDistributionValidation:
    def test_duplicate_distribution_calibration(self):
        block = GCNBlock(
            name="TEST",
            calibration=[
                GCNDistribution(parameter_name="alpha", dist_name="Beta", dist_kwargs={"alpha": 2, "beta": 5}),
                GCNDistribution(parameter_name="alpha", dist_name="Gamma", dist_kwargs={"alpha": 2, "beta": 1}),
            ],
        )
        errors = validate_block(block)
        assert errors.has_errors
        assert _messages(errors) == ["[E101] Duplicate calibration for parameter 'alpha' in block TEST"]
