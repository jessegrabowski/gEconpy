from collections import defaultdict
from collections.abc import Iterable

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


def validate_block(block: GCNBlock) -> ErrorCollector:
    """
    Check one block for duplicate names and for an incomplete optimization problem.

    The duplicate check covers the controls, the shocks, and the calibrated parameters.

    Parameters
    ----------
    block : GCNBlock
        The block to validate.

    Returns
    -------
    errors : ErrorCollector
        Errors for each duplicate found. Controls without an objective, or an objective without constraints, add a
        warning.
    """
    errors = ErrorCollector()

    for name in _repeated_names(v.name for v in block.controls):
        errors.add(GCNSemanticError(f"Duplicate control variable '{name}' in block {block.name}", code=ErrorCode.E101))

    for name in _repeated_names(v.name for v in block.shocks):
        errors.add(GCNSemanticError(f"Duplicate shock variable '{name}' in block {block.name}", code=ErrorCode.E101))

    for name in _repeated_names(_calibrated_parameter_names(block)):
        errors.add(
            GCNSemanticError(f"Duplicate calibration for parameter '{name}' in block {block.name}", code=ErrorCode.E101)
        )

    if block.controls and not block.objective:
        errors.add(
            GCNSemanticError(
                f"Block {block.name} has controls but no objective function",
                code=ErrorCode.W002,
                severity=Severity.WARNING,
            )
        )

    if block.objective and not block.constraints:
        errors.add(
            GCNSemanticError(
                f"Block {block.name} has objective but no constraints", code=ErrorCode.W003, severity=Severity.WARNING
            )
        )

    return errors


def validate_model(model: GCNModel) -> ErrorCollector:
    """
    Validate every block, then check for duplicate block names and parameters calibrated in more than one block.

    Parameters
    ----------
    model : GCNModel
        The model to validate.

    Returns
    -------
    errors : ErrorCollector
        The errors and warnings from every block, plus the model-level errors.
    """
    errors = ErrorCollector()

    for block in model.blocks:
        for error in validate_block(block):
            errors.add(error)

    for name in _repeated_names(block.name for block in model.blocks):
        errors.add(GCNSemanticError(f"Duplicate block name: {name}", code=ErrorCode.E100))

    defining_blocks: dict[str, list[str]] = defaultdict(list)
    for block in model.blocks:
        for param_name in _calibrated_parameter_names(block):
            defining_blocks[param_name].append(block.name)

    for param_name, block_names in defining_blocks.items():
        if len(block_names) > 1:
            errors.add(
                GCNSemanticError(
                    f"Parameter '{param_name}' defined in multiple blocks: {', '.join(block_names)}",
                    code=ErrorCode.E101,
                )
            )

    return errors


def validate_equation(eq: GCNEquation) -> ErrorCollector:  # noqa: ARG001
    """
    Validate a single equation.

    Parameters
    ----------
    eq : GCNEquation
        The equation to validate.

    Returns
    -------
    errors : ErrorCollector
        Always empty. A GCNEquation carries no state that the parser can leave inconsistent.
    """
    return ErrorCollector()


def check_undefined_variables(
    model: GCNModel,
    external_variables: set[str] | None = None,
) -> ErrorCollector:
    """
    Warn about variables that appear on the right-hand side of an equation without being defined anywhere.

    A variable is defined when it is a control, a shock, or the left-hand side of a definition, identity, or
    objective in any block.

    Parameters
    ----------
    model : GCNModel
        The model to check.
    external_variables : set of str, optional
        Names of variables defined outside the model, such as from data. Defaults to none.

    Returns
    -------
    errors : ErrorCollector
        One warning per undefined variable, in sorted name order.
    """
    errors = ErrorCollector()
    external = external_variables or set()

    defined = set()
    used = set()
    for block in model.blocks:
        defined |= _defined_variable_names(block)
        for eq in block.definitions + block.objective + block.constraints + block.identities:
            used |= collect_variable_names(eq.rhs)

    for var in sorted(used - defined - external):
        errors.add(GCNSemanticError(f"Variable '{var}' is used but not defined", severity=Severity.WARNING))

    return errors


def check_undefined_parameters(
    model: GCNModel,
    external_parameters: set[str] | None = None,
) -> ErrorCollector:
    """
    Warn about parameters that appear in an equation without a calibration entry in any block.

    Parameters
    ----------
    model : GCNModel
        The model to check.
    external_parameters : set of str, optional
        Names of parameters defined outside the model. Defaults to none.

    Returns
    -------
    errors : ErrorCollector
        One warning per uncalibrated parameter, in sorted name order.
    """
    errors = ErrorCollector()
    external = external_parameters or set()

    calibrated = set()
    used = set()
    for block in model.blocks:
        calibrated |= set(_calibrated_parameter_names(block))
        for eq in block.definitions + block.objective + block.constraints + block.identities:
            used |= collect_parameter_names(eq.lhs)
            used |= collect_parameter_names(eq.rhs)

    for param in sorted(used - calibrated - external):
        errors.add(GCNSemanticError(f"Parameter '{param}' is used but not calibrated", severity=Severity.WARNING))

    return errors


def full_validation(
    model: GCNModel,
    external_variables: set[str] | None = None,
    external_parameters: set[str] | None = None,
) -> ErrorCollector:
    """
    Run :func:`validate_model`, :func:`check_undefined_variables`, and :func:`check_undefined_parameters`.

    Parameters
    ----------
    model : GCNModel
        The model to validate.
    external_variables : set of str, optional
        Names of variables defined outside the model. Defaults to none.
    external_parameters : set of str, optional
        Names of parameters defined outside the model. Defaults to none.

    Returns
    -------
    errors : ErrorCollector
        All errors and warnings from the three checks, in that order.
    """
    errors = ErrorCollector()

    for check_errors in (
        validate_model(model),
        check_undefined_variables(model, external_variables),
        check_undefined_parameters(model, external_parameters),
    ):
        for error in check_errors:
            errors.add(error)

    return errors


def _repeated_names(names: Iterable[str]) -> list[str]:
    seen = set()
    repeated = []
    for name in names:
        if name in seen:
            repeated.append(name)
        seen.add(name)
    return repeated


def _calibrated_parameter_names(block: GCNBlock) -> list[str]:
    names = []
    for item in block.calibration:
        if isinstance(item, GCNDistribution):
            names.append(item.parameter_name)
        elif isinstance(item, GCNEquation) and isinstance(item.lhs, Parameter):
            names.append(item.lhs.name)
    return names


def _defined_variable_names(block: GCNBlock) -> set[str]:
    names = {v.name for v in block.controls} | {v.name for v in block.shocks}
    for eq in block.definitions + block.identities + block.objective:
        if isinstance(eq.lhs, Variable):
            names.add(eq.lhs.name)
    return names
