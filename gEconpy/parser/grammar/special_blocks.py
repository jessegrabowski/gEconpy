import re

from collections import defaultdict

import pyparsing as pp

from sympy.core.assumptions import _assume_rules

from gEconpy.parser.constants import DEFAULT_ASSUMPTIONS, SYMPY_ASSUMPTIONS
from gEconpy.parser.validation import find_typos_and_guesses
from gEconpy.utilities import flatten_list


def _parse_options_content(content: str) -> dict[str, bool | str]:
    """Parse the content of an options block into a dictionary."""
    result = {}
    content = re.sub("[{}]", "", content)
    lines = [line.strip() for line in content.split(";") if line.strip()]

    for line in lines:
        if "=" not in line:
            continue
        flag, value = line.split("=", 1)
        value = value.strip()
        if value.lower() == "true":
            value = True
        elif value.lower() == "false":
            value = False
        result[flag.strip()] = value

    return result


def _parse_tryreduce_content(content: str) -> list[str]:
    """Parse the content of a tryreduce block into a list of variable names."""
    content = re.sub("[{};]", "", content).strip()
    if not content:
        return []
    return [x.strip().replace(",", "") for x in content.split() if x.strip()]


def _extract_assumption_sub_blocks(text: str) -> dict[str, list[str]]:
    """Extract assumption sub-blocks using pyparsing."""
    LBRACE, RBRACE, SEMI, _COMMA = map(pp.Suppress, "{};,")
    BLOCK_END = RBRACE + SEMI
    header = pp.Keyword("assumptions")
    VARIABLE = pp.Word(pp.alphas, pp.alphanums + "_" + "[]").set_name("variable")
    PARAM = pp.Word(pp.alphas, pp.alphanums + "_").set_name("parameter")
    BLOCK_NAME = pp.Word(pp.alphas, pp.alphanums + "_")

    VAR_LIST = pp.delimitedList((VARIABLE | PARAM), delim=",").set_name("var_list")
    VAR_LINE = pp.Group(VAR_LIST + SEMI).set_name("variable_list")

    ANYTHING = pp.Group(pp.Regex("[^{};]+") + SEMI).set_name("generic_line")

    LINE = VAR_LINE | ANYTHING

    SUBBLOCK = pp.Forward()
    SUBBLOCK <<= pp.Dict(pp.Group((BLOCK_NAME + pp.Group(LBRACE + (LINE | SUBBLOCK) + BLOCK_END)) | LINE))

    LAYERED_BLOCK = pp.Forward()
    LAYERED_BLOCK <<= pp.Dict(pp.Group(header + LBRACE + pp.OneOrMore(LAYERED_BLOCK | SUBBLOCK) + BLOCK_END))

    return LAYERED_BLOCK.parse_string(text).as_dict()["assumptions"]


def _validate_assumptions(block_dict: dict[str, list[str]]) -> None:
    """Verify that all keys are valid sympy assumptions."""
    for assumption in block_dict:
        if assumption not in SYMPY_ASSUMPTIONS:
            best_guess, _ = find_typos_and_guesses([assumption], SYMPY_ASSUMPTIONS)
            message = f'Assumption "{assumption}" is not a valid Sympy assumption.'
            if best_guess is not None:
                message += f' Did you mean "{best_guess}"?'
            raise ValueError(message)


def _create_assumption_kwargs(
    assumption_dicts: dict[str, list[str]],
) -> dict[str, dict[str, bool]]:
    """Convert assumption sub-blocks into per-variable assumption dictionaries."""
    assumption_kwargs = defaultdict(DEFAULT_ASSUMPTIONS.copy)
    user_assumptions = defaultdict(dict)

    for assumption, variable_list in assumption_dicts.items():
        for var in flatten_list(variable_list):
            base_var = re.sub(r"\[\]", "", var)
            user_assumptions[base_var][assumption] = True
            assumption_kwargs[base_var][assumption] = True

    all_variables = set(flatten_list(list(assumption_dicts.values())))

    for var in all_variables:
        base_var = re.sub(r"\[\]", "", var)

        for k, v in DEFAULT_ASSUMPTIONS.items():
            implications = dict(_assume_rules.full_implications[(k, v)])
            for user_k, user_v in user_assumptions[base_var].items():
                if ((user_k == k) and (user_v == v)) or (user_k not in implications):
                    continue
                if implications[user_k] != user_v:
                    del assumption_kwargs[base_var][k]

    return assumption_kwargs


def parse_options(text: str) -> dict[str, bool | str]:
    """
    Parse an options block from GCN text.

    Parameters
    ----------
    text : str
        The full options block text, e.g.:
        "options { output logfile = TRUE; output LaTeX = FALSE; };"

    Returns
    -------
    dict[str, bool | str]
        Dictionary of option names to values.
    """
    match = re.search(r"options\s*\{(.*?)\};", text, re.DOTALL | re.IGNORECASE)
    if not match:
        return {}
    return _parse_options_content(match.group(1))


def parse_tryreduce(text: str) -> list[str]:
    """
    Parse a tryreduce block from GCN text.

    Parameters
    ----------
    text : str
        The full tryreduce block text, e.g.:
        "tryreduce { U[], TC[]; };"

    Returns
    -------
    list[str]
        List of variable names to try to reduce.
    """
    match = re.search(r"tryreduce\s*\{(.*?)\};", text, re.DOTALL | re.IGNORECASE)
    if not match:
        return []
    return _parse_tryreduce_content(match.group(1))


def parse_assumptions(text: str) -> dict[str, dict[str, bool]]:
    """
    Parse an assumptions block from GCN text.

    Parameters
    ----------
    text : str
        The full assumptions block text, e.g.:
        "assumptions { positive { C[], K[]; }; negative { TC[]; }; };"

    Returns
    -------
    dict[str, dict[str, bool]]
        Dictionary mapping variable names to their assumption dictionaries.
    """
    if "assumptions" not in text.lower():
        return defaultdict(lambda: DEFAULT_ASSUMPTIONS)

    # Extract just the assumptions block
    match = re.search(r"assumptions\s*\{", text, re.IGNORECASE)
    if not match:
        return defaultdict(lambda: DEFAULT_ASSUMPTIONS)

    # Find the matching closing brace
    start = match.start()
    brace_start = match.end() - 1
    depth = 0
    pos = brace_start

    while pos < len(text):
        if text[pos] == "{":
            depth += 1
        elif text[pos] == "}":
            depth -= 1
            if depth == 0:
                break
        pos += 1

    # Include the closing };
    end = pos + 1
    if end < len(text) and text[end] == ";":
        end += 1

    assumptions_block = text[start:end]

    # Check if block is empty
    inner = re.sub(r"assumptions\s*\{|\};?", "", assumptions_block).strip()
    if not inner:
        return defaultdict(lambda: DEFAULT_ASSUMPTIONS)

    sub_blocks = _extract_assumption_sub_blocks(assumptions_block)
    _validate_assumptions(sub_blocks)
    return _create_assumption_kwargs(sub_blocks)


def extract_special_block_content(text: str, block_name: str) -> str | None:
    """
    Extract the raw content of a special block from text.

    Parameters
    ----------
    text : str
        The GCN text to search.
    block_name : str
        The name of the block to extract ("options", "tryreduce", or "assumptions").

    Returns
    -------
    str | None
        The block content including braces, or None if not found.
    """
    pattern = rf"{block_name}\s*\{{.*?\}};"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(0)
    return None


def remove_special_block(text: str, block_name: str) -> str:
    """
    Remove a special block from text.

    Parameters
    ----------
    text : str
        The GCN text.
    block_name : str
        The name of the block to remove.

    Returns
    -------
    str
        The text with the block removed.
    """
    pattern = rf"{block_name}\s*\{{.*?\}};"
    return re.sub(pattern, "", text, flags=re.DOTALL | re.IGNORECASE)
