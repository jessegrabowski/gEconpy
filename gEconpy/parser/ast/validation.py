from collections import defaultdict

from gEconpy.parser.ast import (
    GCNBlock,
    GCNDistribution,
    GCNEquation,
    GCNModel,
    Parameter,
    Variable,
    collect_parameter_names,
    collect_variable_names,
)
from gEconpy.parser.error_catalog import ErrorCode
from gEconpy.parser.errors import (
    ErrorCollector,
    GCNSemanticError,
    Severity,
)


def validate_block(block: GCNBlock) -> ErrorCollector:  # noqa: PLR0912
    """
    Validate a single block for semantic errors.

    Parameters
    ----------
    block : GCNBlock
        The block to validate.

    Returns
    -------
    errors : ErrorCollector
        Collection of validation errors found.
    """
    errors = ErrorCollector()

    # Check for duplicate control variables
    control_names = [v.name for v in block.controls]
    seen_controls = set()
    for name in control_names:
        if name in seen_controls:
            errors.add(
                GCNSemanticError(f"Duplicate control variable '{name}' in block {block.name}", code=ErrorCode.E101)
            )
        seen_controls.add(name)

    # Check for duplicate shock variables
    shock_names = [v.name for v in block.shocks]
    seen_shocks = set()
    for name in shock_names:
        if name in seen_shocks:
            errors.add(
                GCNSemanticError(f"Duplicate shock variable '{name}' in block {block.name}", code=ErrorCode.E101)
            )
        seen_shocks.add(name)

    # Check for duplicate calibration parameters
    calibration_params = set()
    for item in block.calibration:
        if isinstance(item, GCNDistribution):
            param_name = item.parameter_name
        elif isinstance(item, GCNEquation):
            if isinstance(item.lhs, Parameter):
                param_name = item.lhs.name
            else:
                continue
        else:
            continue

        if param_name in calibration_params:
            errors.add(
                GCNSemanticError(
                    f"Duplicate calibration for parameter '{param_name}' in block {block.name}", code=ErrorCode.E101
                )
            )
        calibration_params.add(param_name)

    # Check that objective exists if controls exist (optimization problem)
    if block.controls and not block.objective:
        errors.add(
            GCNSemanticError(
                f"Block {block.name} has controls but no objective function",
                code=ErrorCode.W002,
                severity=Severity.WARNING,
            )
        )

    # Check that constraints exist if objective exists
    if block.objective and not block.constraints:
        errors.add(
            GCNSemanticError(
                f"Block {block.name} has objective but no constraints", code=ErrorCode.W003, severity=Severity.WARNING
            )
        )

    return errors


def validate_model(model: GCNModel) -> ErrorCollector:  # noqa: PLR0912
    """
    Validate a complete model for semantic errors.

    Parameters
    ----------
    model : GCNModel
        The model to validate.

    Returns
    -------
    errors : ErrorCollector
        Collection of validation errors found.
    """
    errors = ErrorCollector()

    # Validate each block
    for block in model.blocks:
        block_errors = validate_block(block)
        for error in block_errors:
            errors.add(error)

    # Check for duplicate block names
    block_names = [b.name for b in model.blocks]
    seen_blocks = set()
    for name in block_names:
        if name in seen_blocks:
            errors.add(GCNSemanticError(f"Duplicate block name: {name}", code=ErrorCode.E100))
        seen_blocks.add(name)

    # Collect all defined variables and parameters across blocks
    all_defined_vars = set()
    all_defined_params = set()
    all_shocks = set()

    for block in model.blocks:
        # Controls define variables
        for var in block.controls:
            all_defined_vars.add(var.name)

        # Shocks define variables
        for shock in block.shocks:
            all_shocks.add(shock.name)
            all_defined_vars.add(shock.name)

        # LHS of definitions/identities define variables
        for eq in block.definitions + block.identities:
            if isinstance(eq.lhs, Variable):
                all_defined_vars.add(eq.lhs.name)

        # Objective LHS defines a variable
        for eq in block.objective:
            if isinstance(eq.lhs, Variable):
                all_defined_vars.add(eq.lhs.name)

        # Calibration defines parameters
        for item in block.calibration:
            if isinstance(item, GCNDistribution):
                all_defined_params.add(item.parameter_name)
            elif isinstance(item, GCNEquation) and isinstance(item.lhs, Parameter):
                all_defined_params.add(item.lhs.name)

    # Check for duplicate parameter definitions across blocks
    param_definitions = defaultdict(list)
    for block in model.blocks:
        for item in block.calibration:
            if isinstance(item, GCNDistribution):
                param_definitions[item.parameter_name].append(block.name)
            elif isinstance(item, GCNEquation) and isinstance(item.lhs, Parameter):
                param_definitions[item.lhs.name].append(block.name)

    for param_name, blocks in param_definitions.items():
        if len(blocks) > 1:
            errors.add(
                GCNSemanticError(
                    f"Parameter '{param_name}' defined in multiple blocks: {', '.join(blocks)}", code=ErrorCode.E101
                )
            )

    return errors


def validate_equation(eq: GCNEquation) -> ErrorCollector:
    """
    Validate a single equation.

    Parameters
    ----------
    eq : GCNEquation
        The equation to validate.

    Returns
    -------
    errors : ErrorCollector
        Collection of validation errors found.
    """
    errors = ErrorCollector()

    # Check that calibrating equations have a parameter on LHS or RHS
    if eq.is_calibrating and eq.calibrating_parameter is None:
        errors.add(GCNSemanticError("Calibrating equation has no calibrating parameter"))

    return errors


def check_undefined_variables(
    model: GCNModel,
    external_variables: set[str] | None = None,
) -> ErrorCollector:
    """
    Check for variables used but not defined.

    Parameters
    ----------
    model : GCNModel
        The model to check.
    external_variables : set of str, optional
        Set of variable names that are defined externally (e.g., from data).

    Returns
    -------
    errors : ErrorCollector
        Collection of validation errors for undefined variables.
    """
    errors = ErrorCollector()
    external = external_variables or set()

    # Collect all defined variables
    defined_vars = set()
    for block in model.blocks:
        for var in block.controls:
            defined_vars.add(var.name)
        for var in block.shocks:
            defined_vars.add(var.name)
        for eq in block.definitions + block.identities:
            if isinstance(eq.lhs, Variable):
                defined_vars.add(eq.lhs.name)
        for eq in block.objective:
            if isinstance(eq.lhs, Variable):
                defined_vars.add(eq.lhs.name)

    # Collect all used variables
    used_vars = set()
    for block in model.blocks:
        for eq in block.definitions + block.constraints + block.identities:
            used_vars |= collect_variable_names(eq.rhs)
        for eq in block.objective:
            used_vars |= collect_variable_names(eq.rhs)

    # Find undefined
    undefined = used_vars - defined_vars - external
    for var in sorted(undefined):
        errors.add(GCNSemanticError(f"Variable '{var}' is used but not defined", severity=Severity.WARNING))

    return errors


def check_undefined_parameters(
    model: GCNModel,
    external_parameters: set[str] | None = None,
) -> ErrorCollector:
    """
    Check for parameters used but not calibrated.

    Parameters
    ----------
    model : GCNModel
        The model to check.
    external_parameters : set of str, optional
        Set of parameter names that are defined externally.

    Returns
    -------
    errors : ErrorCollector
        Collection of validation errors for undefined parameters.
    """
    errors = ErrorCollector()
    external = external_parameters or set()

    # Collect all calibrated parameters
    calibrated_params = set()
    for block in model.blocks:
        for item in block.calibration:
            if isinstance(item, GCNDistribution):
                calibrated_params.add(item.parameter_name)
            elif isinstance(item, GCNEquation) and isinstance(item.lhs, Parameter):
                calibrated_params.add(item.lhs.name)

    # Collect all used parameters
    used_params = set()
    for block in model.blocks:
        for eq in block.definitions + block.constraints + block.identities:
            used_params |= collect_parameter_names(eq.lhs)
            used_params |= collect_parameter_names(eq.rhs)
        for eq in block.objective:
            used_params |= collect_parameter_names(eq.lhs)
            used_params |= collect_parameter_names(eq.rhs)

    # Find undefined
    undefined = used_params - calibrated_params - external
    for param in sorted(undefined):
        errors.add(GCNSemanticError(f"Parameter '{param}' is used but not calibrated", severity=Severity.WARNING))

    return errors


def full_validation(
    model: GCNModel,
    external_variables: set[str] | None = None,
    external_parameters: set[str] | None = None,
) -> ErrorCollector:
    """
    Run full validation on a model.

    Parameters
    ----------
    model : GCNModel
        The model to validate.
    external_variables : set of str, optional
        Set of externally defined variable names.
    external_parameters : set of str, optional
        Set of externally defined parameter names.

    Returns
    -------
    errors : ErrorCollector
        All validation errors and warnings.
    """
    errors = ErrorCollector()

    model_errors = validate_model(model)
    for e in model_errors:
        errors.add(e)

    var_errors = check_undefined_variables(model, external_variables)
    for e in var_errors:
        errors.add(e)

    param_errors = check_undefined_parameters(model, external_parameters)
    for e in param_errors:
        errors.add(e)

    return errors
