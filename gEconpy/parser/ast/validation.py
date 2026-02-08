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
from gEconpy.parser.errors import (
    ValidationErrorCollection,
)


def validate_block(block: GCNBlock) -> ValidationErrorCollection:  # noqa: PLR0912
    """
    Validate a single block for semantic errors.

    Parameters
    ----------
    block : GCNBlock
        The block to validate.

    Returns
    -------
    ValidationErrorCollection
        Collection of validation errors found.
    """
    errors = ValidationErrorCollection()

    # Check for duplicate control variables
    control_names = [v.name for v in block.controls]
    seen_controls = set()
    for name in control_names:
        if name in seen_controls:
            errors.add_error(f"Duplicate control variable '{name}' in block {block.name}")
        seen_controls.add(name)

    # Check for duplicate shock variables
    shock_names = [v.name for v in block.shocks]
    seen_shocks = set()
    for name in shock_names:
        if name in seen_shocks:
            errors.add_error(f"Duplicate shock variable '{name}' in block {block.name}")
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
            errors.add_error(f"Duplicate calibration for parameter '{param_name}' in block {block.name}")
        calibration_params.add(param_name)

    # Check that objective exists if controls exist (optimization problem)
    if block.controls and not block.objective:
        errors.add_warning(f"Block {block.name} has controls but no objective function")

    # Check that constraints exist if objective exists
    if block.objective and not block.constraints:
        errors.add_warning(f"Block {block.name} has objective but no constraints")

    return errors


def validate_model(model: GCNModel) -> ValidationErrorCollection:  # noqa: PLR0912
    """
    Validate a complete model for semantic errors.

    Parameters
    ----------
    model : GCNModel
        The model to validate.

    Returns
    -------
    ValidationErrorCollection
        Collection of validation errors found.
    """
    errors = ValidationErrorCollection()

    # Validate each block
    for block in model.blocks:
        block_errors = validate_block(block)
        for error in block_errors:
            errors._errors.append(error)

    # Check for duplicate block names
    block_names = [b.name for b in model.blocks]
    seen_blocks = set()
    for name in block_names:
        if name in seen_blocks:
            errors.add_error(f"Duplicate block name: {name}")
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

    for param_name, block_names in param_definitions.items():
        if len(block_names) > 1:
            errors.add_error(f"Parameter '{param_name}' defined in multiple blocks: {', '.join(block_names)}")

    return errors


def validate_equation(eq: GCNEquation) -> ValidationErrorCollection:
    """
    Validate a single equation.

    Parameters
    ----------
    eq : GCNEquation
        The equation to validate.

    Returns
    -------
    ValidationErrorCollection
        Collection of validation errors found.
    """
    errors = ValidationErrorCollection()

    # Check that calibrating equations have a parameter on LHS or RHS
    if eq.is_calibrating and eq.calibrating_parameter is None:
        errors.add_error("Calibrating equation has no calibrating parameter")

    return errors


def check_undefined_variables(
    model: GCNModel,
    external_variables: set[str] | None = None,
) -> ValidationErrorCollection:
    """
    Check for variables used but not defined.

    Parameters
    ----------
    model : GCNModel
        The model to check.
    external_variables : set[str], optional
        Set of variable names that are defined externally (e.g., from data).

    Returns
    -------
    ValidationErrorCollection
        Collection of validation errors for undefined variables.
    """
    errors = ValidationErrorCollection()
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
        errors.add_warning(f"Variable '{var}' is used but not defined")

    return errors


def check_undefined_parameters(
    model: GCNModel,
    external_parameters: set[str] | None = None,
) -> ValidationErrorCollection:
    """
    Check for parameters used but not calibrated.

    Parameters
    ----------
    model : GCNModel
        The model to check.
    external_parameters : set[str], optional
        Set of parameter names that are defined externally.

    Returns
    -------
    ValidationErrorCollection
        Collection of validation errors for undefined parameters.
    """
    errors = ValidationErrorCollection()
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
        errors.add_warning(f"Parameter '{param}' is used but not calibrated")

    return errors


def full_validation(
    model: GCNModel,
    external_variables: set[str] | None = None,
    external_parameters: set[str] | None = None,
) -> ValidationErrorCollection:
    """
    Run full validation on a model.

    Parameters
    ----------
    model : GCNModel
        The model to validate.
    external_variables : set[str], optional
        Set of externally defined variable names.
    external_parameters : set[str], optional
        Set of externally defined parameter names.

    Returns
    -------
    ValidationErrorCollection
        All validation errors and warnings.
    """
    errors = ValidationErrorCollection()

    # Basic model validation
    model_errors = validate_model(model)
    for e in model_errors:
        errors._errors.append(e)

    # Check undefined variables (as warnings)
    var_errors = check_undefined_variables(model, external_variables)
    for e in var_errors:
        errors._errors.append(e)

    # Check undefined parameters (as warnings)
    param_errors = check_undefined_parameters(model, external_parameters)
    for e in param_errors:
        errors._errors.append(e)

    return errors
