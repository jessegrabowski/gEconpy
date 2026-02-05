from collections import defaultdict
from pathlib import Path
from typing import Any

from gEconpy.parser.ast import (
    BinaryOp,
    Expectation,
    FunctionCall,
    GCNBlock,
    GCNDistribution,
    GCNEquation,
    GCNModel,
    Node,
    Number,
    Operator,
    Parameter,
    UnaryOp,
    Variable,
)
from gEconpy.parser.ast.printer import print_expression
from gEconpy.parser.preprocessor import preprocess

OPERATOR_TOKENS = {
    Operator.ADD: "+",
    Operator.SUB: "-",
    Operator.MUL: "*",
    Operator.DIV: "/",
    Operator.POW: "^",
    Operator.NEG: "-",
}


def _expression_to_tokens(node: Node) -> list[str]:  # noqa: PLR0911
    """Convert an AST expression to a list of tokens (old parser format)."""
    match node:
        case Number(value=value):
            if value == int(value):
                return [str(int(value))]
            return [str(value)]

        case Parameter(name=name):
            return [name]

        case Variable(name=name, time_index=time_index):
            return [f"{name}{time_index}"]

        case UnaryOp(op=op, operand=operand):
            return [OPERATOR_TOKENS[op], *_expression_to_tokens(operand)]

        case BinaryOp(left=left, op=op, right=right):
            left_tokens = _expression_to_tokens(left)
            right_tokens = _expression_to_tokens(right)
            return [*left_tokens, OPERATOR_TOKENS[op], *right_tokens]

        case FunctionCall(func_name=func_name, args=args):
            tokens = [func_name, "("]
            for i, arg in enumerate(args):
                if i > 0:
                    tokens.append(",")
                tokens.extend(_expression_to_tokens(arg))
            tokens.append(")")
            return tokens

        case Expectation(expr=expr):
            inner_tokens = _expression_to_tokens(expr)
            return ["E[]", "[", *inner_tokens, "]"]

        case _:
            return [str(node)]


def _equation_to_tokens(eq: GCNEquation) -> list[str]:
    """Convert a GCNEquation to a list of tokens (old parser format)."""
    lhs_tokens = _expression_to_tokens(eq.lhs)
    rhs_tokens = _expression_to_tokens(eq.rhs)

    tokens = [*lhs_tokens, "=", *rhs_tokens]

    if eq.lagrange_multiplier:
        tokens.extend([":", f"{eq.lagrange_multiplier}[]"])

    if eq.is_calibrating and eq.calibrating_parameter:
        tokens.extend(["->", eq.calibrating_parameter])

    return tokens


def block_to_legacy_dict(block: GCNBlock) -> dict[str, list[list[str]]]:
    """
    Convert a GCNBlock to the old parser's block dictionary format.

    Parameters
    ----------
    block : GCNBlock
        The new AST block representation.

    Returns
    -------
    dict[str, list[list[str]]]
        Block dictionary in old format: component -> list of equation token lists.
    """
    block_dict: dict[str, list[list[str]]] = defaultdict(list)

    # Definitions
    for eq in block.definitions:
        block_dict["definitions"].append(_equation_to_tokens(eq))

    # Controls (as variable list)
    if block.controls:
        control_tokens = [print_expression(v) for v in block.controls]
        block_dict["controls"].append(control_tokens)

    # Objective
    if block.objective:
        block_dict["objective"].append(_equation_to_tokens(block.objective))

    # Constraints
    for eq in block.constraints:
        block_dict["constraints"].append(_equation_to_tokens(eq))

    # Identities
    for eq in block.identities:
        block_dict["identities"].append(_equation_to_tokens(eq))

    # Shocks (as variable list)
    if block.shocks:
        shock_tokens = [print_expression(v) for v in block.shocks]
        block_dict["shocks"].append(shock_tokens)

    # Calibration
    for item in block.calibration:
        if isinstance(item, GCNEquation):
            block_dict["calibration"].append(_equation_to_tokens(item))
        elif isinstance(item, GCNDistribution) and item.initial_value is not None:
            # For distributions, we need the parameter = initial_value format
            block_dict["calibration"].append([item.parameter_name, "=", str(item.initial_value)])

    return dict(block_dict)


def model_to_legacy_block_dict(
    model: GCNModel,
) -> dict[str, str]:
    """
    Convert a GCNModel to the old parser's raw block dictionary format.

    This produces a dictionary mapping block names to their raw string content,
    similar to what split_gcn_into_dictionaries returns.

    Parameters
    ----------
    model : GCNModel
        The new AST model representation.

    Returns
    -------
    dict[str, str]
        Dictionary mapping block names to block content strings.
    """
    from gEconpy.parser.ast.printer import print_block  # noqa: PLC0415

    block_dict = {}
    for block in model.blocks:
        # Use the printer to get the block content, then strip the header
        block_str = print_block(block)
        # Remove "block NAME\n{" from start and "};" from end
        lines = block_str.split("\n")
        # Skip first two lines (block NAME, {) and last line (};)
        content_lines = lines[2:-1]
        content = "\n".join(content_lines)
        block_dict[block.name] = "{ " + content + " };"

    return block_dict


def extract_prior_dict(model: GCNModel) -> dict[str, str]:
    """
    Extract prior distribution strings from a GCNModel.

    Parameters
    ----------
    model : GCNModel
        The new AST model representation.

    Returns
    -------
    dict[str, str]
        Dictionary mapping parameter names to distribution strings.
    """
    from gEconpy.parser.ast.printer import print_distribution  # noqa: PLC0415

    prior_dict = {}

    for block in model.blocks:
        for item in block.calibration:
            if isinstance(item, GCNDistribution):
                # Format: "param ~ Dist(...) = initial"
                dist_str = print_distribution(item)
                # Remove "param ~ " prefix
                if " ~ " in dist_str:
                    _, dist_part = dist_str.split(" ~ ", 1)
                    prior_dict[item.parameter_name] = dist_part

        # Also check shocks for distributions
        # Shocks with distributions are stored differently in old parser

    return prior_dict


def parse_gcn_legacy_format(
    source: str,
) -> tuple[dict[str, str], dict[str, str], list[str], dict[str, dict[str, bool]], dict[str, str]]:
    """
    Parse a GCN source string using the new parser but return old format.

    This is the main compatibility function for gradual migration.

    Parameters
    ----------
    source : str
        GCN source text.

    Returns
    -------
    tuple containing:
        - block_dict: dict[str, str] - Raw block strings by name
        - options: dict[str, str] - Model options
        - tryreduce: list[str] - Variables to try to reduce
        - assumptions: dict[str, dict[str, bool]] - Variable assumptions
        - prior_dict: dict[str, str] - Distribution strings by parameter
    """
    result = preprocess(source, validate=False)
    model = result.ast

    block_dict = model_to_legacy_block_dict(model)
    prior_dict = extract_prior_dict(model)

    return (
        block_dict,
        model.options,
        model.tryreduce,
        model.assumptions,
        prior_dict,
    )


def parse_gcn_file_legacy_format(
    filepath: str | Path,
) -> tuple[dict[str, str], dict[str, str], list[str], dict[str, dict[str, bool]], dict[str, str]]:
    """
    Parse a GCN file using the new parser but return old format.

    Parameters
    ----------
    filepath : str | Path
        Path to GCN file.

    Returns
    -------
    tuple
        Same as parse_gcn_legacy_format.
    """
    filepath = Path(filepath)
    source = filepath.read_text()
    return parse_gcn_legacy_format(source)


def compare_parser_outputs(
    source: str,
) -> dict[str, Any]:
    """
    Compare outputs of old and new parser for testing.

    Parameters
    ----------
    source : str
        GCN source text.

    Returns
    -------
    dict
        Dictionary with 'old', 'new', and 'differences' keys.
    """
    from gEconpy.parser.gEcon_parser import preprocess_gcn, split_gcn_into_dictionaries  # noqa: PLC0415

    # Old parser
    try:
        old_processed, old_prior = preprocess_gcn(source)
        old_blocks, old_options, old_tryreduce, old_assumptions = split_gcn_into_dictionaries(old_processed)
        old_result = {
            "blocks": old_blocks,
            "options": old_options,
            "tryreduce": old_tryreduce,
            "assumptions": dict(old_assumptions),
            "prior_dict": old_prior,
        }
    except Exception as e:
        old_result = {"error": str(e)}

    # New parser
    try:
        new_blocks, new_options, new_tryreduce, new_assumptions, new_prior = parse_gcn_legacy_format(source)
        new_result = {
            "blocks": new_blocks,
            "options": new_options,
            "tryreduce": new_tryreduce,
            "assumptions": dict(new_assumptions),
            "prior_dict": new_prior,
        }
    except Exception as e:
        new_result = {"error": str(e)}

    # Compute differences
    differences = {}
    if "error" not in old_result and "error" not in new_result:
        for key in ["options", "tryreduce"]:
            if old_result[key] != new_result[key]:
                differences[key] = {
                    "old": old_result[key],
                    "new": new_result[key],
                }

        # Compare block names
        old_block_names = set(old_result["blocks"].keys())
        new_block_names = set(new_result["blocks"].keys())
        if old_block_names != new_block_names:
            differences["block_names"] = {
                "old": old_block_names,
                "new": new_block_names,
            }

    return {
        "old": old_result,
        "new": new_result,
        "differences": differences,
    }
