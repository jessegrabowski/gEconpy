import pytest

from gEconpy.parser.ast import (
    T_MINUS_1,
    BinaryOp,
    GCNBlock,
    GCNDistribution,
    GCNEquation,
    GCNModel,
    Number,
    Operator,
    Parameter,
    T,
    Variable,
)
from gEconpy.parser.ast.validation import (
    check_undefined_parameters,
    check_undefined_variables,
    full_validation,
    validate_block,
    validate_model,
)


class TestValidateBlock:
    def test_empty_block_no_errors(self):
        block = GCNBlock(name="TEST")
        errors = validate_block(block)
        assert not errors.has_errors

    def test_duplicate_control_variable(self):
        block = GCNBlock(name="TEST")
        block.controls = [
            Variable(name="C"),
            Variable(name="L"),
            Variable(name="C"),  # Duplicate
        ]
        errors = validate_block(block)
        assert errors.has_errors
        assert any("Duplicate control" in str(e) for e in errors)

    def test_duplicate_shock_variable(self):
        block = GCNBlock(name="TEST")
        block.shocks = [
            Variable(name="epsilon_A"),
            Variable(name="epsilon_A"),  # Duplicate
        ]
        errors = validate_block(block)
        assert errors.has_errors
        assert any("Duplicate shock" in str(e) for e in errors)

    def test_duplicate_calibration_parameter(self):
        block = GCNBlock(name="TEST")
        block.calibration = [
            GCNEquation(lhs=Parameter(name="alpha"), rhs=Number(value=0.3)),
            GCNEquation(lhs=Parameter(name="alpha"), rhs=Number(value=0.4)),  # Duplicate
        ]
        errors = validate_block(block)
        assert errors.has_errors
        assert any("Duplicate calibration" in str(e) for e in errors)

    def test_controls_without_objective_warning(self):
        block = GCNBlock(name="TEST")
        block.controls = [Variable(name="C")]
        # No objective
        errors = validate_block(block)
        assert any("controls but no objective" in str(e) for e in errors)

    def test_objective_without_constraints_warning(self):
        block = GCNBlock(name="TEST")
        block.objective = GCNEquation(
            lhs=Variable(name="U"),
            rhs=Variable(name="u"),
        )
        # No constraints
        errors = validate_block(block)
        assert any("objective but no constraints" in str(e) for e in errors)

    def test_valid_optimization_block_no_errors(self):
        block = GCNBlock(name="HOUSEHOLD")
        block.controls = [Variable(name="C")]
        block.objective = GCNEquation(
            lhs=Variable(name="U"),
            rhs=Variable(name="u"),
        )
        block.constraints = [
            GCNEquation(
                lhs=Variable(name="C"),
                rhs=Variable(name="Y"),
            )
        ]
        errors = validate_block(block)
        # Should only have warnings about undefined vars, not structural errors
        assert not errors.has_errors


class TestValidateModel:
    def test_empty_model_no_errors(self):
        model = GCNModel(blocks=[], options={}, tryreduce=[], assumptions={})
        errors = validate_model(model)
        assert not errors.has_errors

    def test_duplicate_block_names(self):
        block1 = GCNBlock(name="HOUSEHOLD")
        block2 = GCNBlock(name="HOUSEHOLD")  # Duplicate
        model = GCNModel(
            blocks=[block1, block2],
            options={},
            tryreduce=[],
            assumptions={},
        )
        errors = validate_model(model)
        assert errors.has_errors
        assert any("Duplicate block name" in str(e) for e in errors)

    def test_parameter_defined_in_multiple_blocks(self):
        block1 = GCNBlock(name="HOUSEHOLD")
        block1.calibration = [GCNEquation(lhs=Parameter(name="alpha"), rhs=Number(value=0.3))]

        block2 = GCNBlock(name="FIRM")
        block2.calibration = [
            GCNEquation(lhs=Parameter(name="alpha"), rhs=Number(value=0.35))  # Same param
        ]

        model = GCNModel(
            blocks=[block1, block2],
            options={},
            tryreduce=[],
            assumptions={},
        )
        errors = validate_model(model)
        assert errors.has_errors
        assert any("defined in multiple blocks" in str(e) for e in errors)

    def test_valid_multi_block_model(self):
        block1 = GCNBlock(name="HOUSEHOLD")
        block1.calibration = [GCNEquation(lhs=Parameter(name="beta"), rhs=Number(value=0.99))]

        block2 = GCNBlock(name="FIRM")
        block2.calibration = [GCNEquation(lhs=Parameter(name="alpha"), rhs=Number(value=0.35))]

        model = GCNModel(
            blocks=[block1, block2],
            options={},
            tryreduce=[],
            assumptions={},
        )
        errors = validate_model(model)
        assert not errors.has_errors


class TestCheckUndefinedVariables:
    def test_all_variables_defined(self):
        block = GCNBlock(name="TEST")
        block.controls = [Variable(name="C"), Variable(name="Y")]
        block.identities = [
            GCNEquation(
                lhs=Variable(name="Y"),
                rhs=Variable(name="C"),
            )
        ]

        model = GCNModel(blocks=[block], options={}, tryreduce=[], assumptions={})
        errors = check_undefined_variables(model)
        assert not list(errors)

    def test_undefined_variable_warning(self):
        block = GCNBlock(name="TEST")
        block.controls = [Variable(name="C")]
        block.identities = [
            GCNEquation(
                lhs=Variable(name="Y"),
                rhs=BinaryOp(
                    left=Variable(name="C"),
                    op=Operator.ADD,
                    right=Variable(name="I"),  # I is not defined
                ),
            )
        ]

        model = GCNModel(blocks=[block], options={}, tryreduce=[], assumptions={})
        errors = check_undefined_variables(model)
        assert any("'I' is used but not defined" in str(e) for e in errors)

    def test_external_variables_not_flagged(self):
        block = GCNBlock(name="TEST")
        block.identities = [
            GCNEquation(
                lhs=Variable(name="Y"),
                rhs=Variable(name="X"),  # X is external
            )
        ]

        model = GCNModel(blocks=[block], options={}, tryreduce=[], assumptions={})
        errors = check_undefined_variables(model, external_variables={"X"})
        assert not any("'X'" in str(e) for e in errors)


class TestCheckUndefinedParameters:
    def test_all_parameters_calibrated(self):
        block = GCNBlock(name="TEST")
        block.identities = [
            GCNEquation(
                lhs=Variable(name="Y"),
                rhs=BinaryOp(
                    left=Parameter(name="alpha"),
                    op=Operator.MUL,
                    right=Variable(name="K"),
                ),
            )
        ]
        block.calibration = [GCNEquation(lhs=Parameter(name="alpha"), rhs=Number(value=0.3))]

        model = GCNModel(blocks=[block], options={}, tryreduce=[], assumptions={})
        errors = check_undefined_parameters(model)
        assert not list(errors)

    def test_uncalibrated_parameter_warning(self):
        block = GCNBlock(name="TEST")
        block.identities = [
            GCNEquation(
                lhs=Variable(name="Y"),
                rhs=BinaryOp(
                    left=Parameter(name="alpha"),
                    op=Operator.MUL,
                    right=Variable(name="K"),
                ),
            )
        ]
        # No calibration for alpha

        model = GCNModel(blocks=[block], options={}, tryreduce=[], assumptions={})
        errors = check_undefined_parameters(model)
        assert any("'alpha' is used but not calibrated" in str(e) for e in errors)

    def test_external_parameters_not_flagged(self):
        block = GCNBlock(name="TEST")
        block.identities = [
            GCNEquation(
                lhs=Variable(name="Y"),
                rhs=Parameter(name="beta"),  # External
            )
        ]

        model = GCNModel(blocks=[block], options={}, tryreduce=[], assumptions={})
        errors = check_undefined_parameters(model, external_parameters={"beta"})
        assert not any("'beta'" in str(e) for e in errors)


class TestFullValidation:
    def test_complete_valid_model(self):
        block = GCNBlock(name="HOUSEHOLD")
        block.controls = [Variable(name="C"), Variable(name="K")]
        block.objective = GCNEquation(
            lhs=Variable(name="U"),
            rhs=Variable(name="u"),
        )
        block.constraints = [
            GCNEquation(
                lhs=Variable(name="C"),
                rhs=BinaryOp(
                    left=Variable(name="Y"),
                    op=Operator.SUB,
                    right=Variable(name="I"),
                ),
            )
        ]
        block.definitions = [
            GCNEquation(
                lhs=Variable(name="u"),
                rhs=Variable(name="C"),
            )
        ]
        block.identities = [
            GCNEquation(
                lhs=Variable(name="Y"),
                rhs=BinaryOp(
                    left=Parameter(name="alpha"),
                    op=Operator.MUL,
                    right=Variable(name="K", time_index=T_MINUS_1),
                ),
            )
        ]
        block.calibration = [GCNEquation(lhs=Parameter(name="alpha"), rhs=Number(value=0.35))]

        model = GCNModel(blocks=[block], options={}, tryreduce=[], assumptions={})

        # Mark I as external (defined elsewhere)
        errors = full_validation(model, external_variables={"I"})

        # Should have no errors (maybe some warnings about undefined)
        assert not errors.has_errors

    def test_multiple_issues(self):
        block1 = GCNBlock(name="BLOCK_A")
        block1.controls = [Variable(name="C"), Variable(name="C")]  # Duplicate

        block2 = GCNBlock(name="BLOCK_A")  # Duplicate block name
        block2.calibration = [GCNEquation(lhs=Parameter(name="alpha"), rhs=Number(value=0.3))]

        model = GCNModel(
            blocks=[block1, block2],
            options={},
            tryreduce=[],
            assumptions={},
        )
        errors = full_validation(model)

        # Should have multiple errors
        error_messages = [str(e) for e in errors]
        assert any("Duplicate control" in msg for msg in error_messages)
        assert any("Duplicate block" in msg for msg in error_messages)


class TestDistributionValidation:
    def test_distribution_calibration_tracked(self):
        block = GCNBlock(name="TEST")
        block.calibration = [
            GCNDistribution(
                parameter_name="alpha",
                dist_name="Beta",
                dist_kwargs={"alpha": 2, "beta": 5},
                initial_value=0.35,
            )
        ]
        block.identities = [
            GCNEquation(
                lhs=Variable(name="Y"),
                rhs=Parameter(name="alpha"),
            )
        ]

        model = GCNModel(blocks=[block], options={}, tryreduce=[], assumptions={})
        errors = check_undefined_parameters(model)
        # alpha is defined via distribution, should not be flagged
        assert not any("'alpha'" in str(e) for e in errors)

    def test_duplicate_distribution_calibration(self):
        block = GCNBlock(name="TEST")
        block.calibration = [
            GCNDistribution(
                parameter_name="alpha",
                dist_name="Beta",
                dist_kwargs={"alpha": 2, "beta": 5},
            ),
            GCNDistribution(
                parameter_name="alpha",  # Duplicate
                dist_name="Gamma",
                dist_kwargs={"alpha": 2, "beta": 1},
            ),
        ]

        errors = validate_block(block)
        assert errors.has_errors
        assert any("Duplicate calibration" in str(e) for e in errors)
