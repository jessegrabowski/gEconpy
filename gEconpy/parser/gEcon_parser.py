import re

from collections import defaultdict
from typing import Literal, Optional, Union, cast

import pyparsing as pp

from sympy.core.assumptions import _assume_rules

from gEconpy.exceptions.exceptions import GCNSyntaxError
from gEconpy.parser.constants import (
    DEFAULT_ASSUMPTIONS,
    SYMPY_ASSUMPTIONS,
)
from gEconpy.parser.parse_equations import rebuild_eqs_from_parser_output
from gEconpy.parser.parse_plaintext import (
    add_spaces_around_operators,
    delete_block,
    extract_distributions,
    remove_comments,
    remove_extra_spaces,
    remove_newlines_and_tabs,
)
from gEconpy.parser.validation import (
    block_is_empty,
    find_typos_and_guesses,
    validate_key,
)
from gEconpy.shared.utilities import flatten_list

SPECIAL_BLOCK = Literal["tryreduce", "assumptions", "options"]
ASSUMPTION_DICT = dict[str, dict[str, bool]]
SPECIAL_BLOCK_DEFAULT = {
    "tryreduce": [],
    "assumptions": defaultdict(lambda: DEFAULT_ASSUMPTIONS),
    "options": {},
}


def block_to_clean_list(block: str) -> list[str]:
    """
    Processes a block of text by removing certain characters, and then splitting it into a list of strings.

    Parameters
    ----------
    block : str
        The block of text to process.

    Returns
    -------
    block: list of str
        The processed list of strings.
    """

    block = re.sub("[{};]", "", block)
    block = remove_extra_spaces(block).strip()
    block = [x.replace(",", "").strip() for x in block.split()]

    return block


def extract_assumption_sub_blocks(block_str) -> dict[str, list[str]]:
    """
    Extracts the special "Assumptions" block from the GCN file. Saves each user-provided assumption to a dictionary,
    along with all variables associated to that assumption.

    Parameters
    ----------
    block: list of str
        The block of text to process.

    Returns
    -------
    assumptions, dict
        A dictionary containing assumptions and variables, with the assumption names as keys and associated variables
        as values.
    """
    LBRACE, RBRACE, SEMI, COMMA = map(pp.Suppress, "{};,")
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
    SUBBLOCK << pp.Dict(
        pp.Group((BLOCK_NAME + pp.Group(LBRACE + (LINE | SUBBLOCK) + BLOCK_END)) | LINE)
    )

    LAYERED_BLOCK = pp.Forward()
    LAYERED_BLOCK << pp.Dict(
        pp.Group(header + LBRACE + pp.OneOrMore(LAYERED_BLOCK | SUBBLOCK) + BLOCK_END)
    )

    return LAYERED_BLOCK.parse_string(block_str).as_dict()["assumptions"]


def validate_assumptions(block_dict: dict[str, list[str]]) -> None:
    """
    Verify that all keys extracted from the assumption block are valid sympy assumptions.

    Parameters
    ----------
    block_dict: dict
        Dictionary of assumption: variable list key-value pairs, extracted from the GCN file by the
        extract_assumption_sub_block function.

    Returns
    -------
    None
    """

    for assumption in block_dict.keys():
        if assumption not in SYMPY_ASSUMPTIONS:
            best_guess, maybe_typo = find_typos_and_guesses([assumption], SYMPY_ASSUMPTIONS)
            message = f'Assumption "{assumption}" is not a valid Sympy assumption.'
            if best_guess is not None:
                message += f' Did you mean "{best_guess}"?'
            raise ValueError(message)


def create_assumption_kwargs(
    assumption_dicts: dict[str, list[str]],
) -> dict[str, dict[str, bool]]:
    """
    Extracts assumption flags from `assumption_dicts` and returns them in a dictionary keyed by variable names.

    Parameters
    ----------
    assumption_dicts: dict
        A dictionary containing assumptions and variables, with the assumption names as keys and associated variables
        as values.

    Returns
    -------
    assumptions: dict
        A dictionary of flags and values keyed by variable names.
    """

    assumption_kwargs = defaultdict(lambda: DEFAULT_ASSUMPTIONS.copy())
    user_assumptions = defaultdict(dict)

    # Maintain two dicts in first pass: one with user values + defaults, and one with just user values
    # The user assumption dict will be used as the source of truth in the 2nd pass to resolve conflicts with defaults
    for assumption, variable_list in assumption_dicts.items():
        for var in flatten_list(variable_list):
            base_var = re.sub(r"\[\]", "", var)
            user_assumptions[base_var][assumption] = True
            assumption_kwargs[base_var][assumption] = True

    all_variables = set(flatten_list(list(assumption_dicts.values())))

    for var in all_variables:
        base_var = re.sub(r"\[\]", "", var)

        # Check default assumptions against the user assumptions
        for k, v in DEFAULT_ASSUMPTIONS.items():
            implications = dict(_assume_rules.full_implications[(k, v)])
            for user_k, user_v in user_assumptions[base_var].items():
                # Assumptions agree, move along
                if ((user_k == k) and (user_v == v)) or (user_k not in implications.keys()):
                    continue

                # Assumptions disagree -- delete the default
                if implications[user_k] != user_v:
                    del assumption_kwargs[base_var][k]

    return assumption_kwargs


def preprocess_gcn(gcn_raw: str) -> tuple[str, dict[str, str]]:
    """
    Preprocesses `gcn_raw` and returns the result.

    Parameters
    ----------
    gcn_raw : str
        Raw model file returned by function `load_gcn`.

    Returns
    -------
    gcn_processed: str
        Model file with basic preprocessing applied

    prior_dict: dict
        Dictionary of variables and associated prior distributions
    """

    gcn_processed = remove_comments(gcn_raw)
    gcn_processed, prior_dict = extract_distributions(gcn_processed)
    gcn_processed = remove_newlines_and_tabs(gcn_processed)
    gcn_processed = add_spaces_around_operators(gcn_processed)

    return gcn_processed, prior_dict


def parse_options_flags(options: str) -> Optional[dict[str, bool]]:
    """
    Extracts flags and values from `options`.

    Parameters
    ----------
    options : str
        Text from the "options" block of a model file.

    Returns
    -------
    Optional[dict[str, bool]]
        A dictionary of flags and values if they exist, or None if no options were found.

    Notes
    -----
    Currently nothing is done with these values, and this step is primarily to ensure backwards compatibility with
    .GCN files written for the gEcon R package.
    """

    result = dict()
    options = re.sub("[{}]", "", options)
    options = [line.strip() for line in options.split(";") if len(line.strip()) > 0]

    if len(options) == 0:
        return result

    for option in options:
        flag, value = option.split("=")
        value = value.replace(";", "").strip()
        value = True if value.lower() == "true" else False if value.lower() == "false" else value

        result[flag.strip()] = value

    return result


def extract_special_block(text: str, block_name: str) -> dict[str, list[str]]:
    """
    Parameters
    ----------
    text: str
        Plaintext representation of a block form a GCN file. Should already be preprocessed by the
        preprocess_gcn function.
    block_name: str
        Name of the block, used as the key in the block dictionary.

    Returns
    -------
    block_dict: dict
        A dictionary with the name as the key and the contents of the block as the values. The contents are split into
        a list of strings, with each item in the list as a single line from the GCN file. Empty lines are discarded.
    """
    result = {
        block_name: defaultdict(lambda: DEFAULT_ASSUMPTIONS)
        if block_name == "assumptions"
        else None
    }

    if block_name not in text:
        return result[block_name]

    block = re.search(block_name + " {.*?" + "};", text)[0]
    block = block.replace(block_name, "")

    if block_is_empty(block):
        return result[block_name]

    elif block_name == "options":
        block = parse_options_flags(block)

    elif block_name == "tryreduce":
        block = block_to_clean_list(block)

    elif block_name == "assumptions":
        block = extract_assumption_sub_blocks(text)
        validate_assumptions(block)
        block = create_assumption_kwargs(block)

    return block


def process_special_block_text(text: str, name: SPECIAL_BLOCK) -> tuple[str, Union[dict, list]]:
    """
    Extract special blocks from a preprocessed GCN text string. Modifies the GCN text string in-place by deleting
    the special block.

    Parameters
    ----------
    text: str
        Preprocessed GCN string
    name: str
        Name of special block. One of "tryreduce", "assumptions", "options"

    Returns
    -------
    text: str
        Preprocessed GCN file, with special block text removed

    result: list or dict
        Special block data. "tryreduce" returns a list, otherwise a dictionary
    """
    name = name.lower()
    result = extract_special_block(text, name)
    text = delete_block(text, name)

    if result is None:
        result = SPECIAL_BLOCK_DEFAULT[name]

    return text, result


def split_gcn_into_dictionaries(
    text: str,
) -> tuple[dict[str, str], dict[str, str], list[str], ASSUMPTION_DICT]:
    """
    Split the preprocessed GCN text by blocks.

    Currently there are three special blocks: "options", "tryreduce", and "assumptions". These are extracted from
    the text and removed from the main text block. The remaining blocks are organized into a dictionary with the
    block name as the key and the (raw) block text as the value.

    Parameters
    ----------
    text : str
        Text of a model file after text processing using the preprocess_gcn function. I.e., comments are expected to be
        removed, all tokens are splittable on single white spaces, and blocks are wrapped by { };

    Returns
    -------
    block_dict: dict[str, str]
        A "block dictionary" with key, value pairs of block_name:block_text. Special blocks are processed first
        (currently "options" and "tryreduce"), then deleted. Normal model blocks are assumed to follow a standard format
        of block NAME { component_1 { Equations }; component_2 { ... }; };

    options: dict[str, str]
        A dictionary of flags and values from the "options" block.

    tryreduce: list[str]
        A list of variables to attempt to reduce.

    assumptions: dict[str, dict[str, bool]]
        Dictionary of assumption flags for each variable in the model. Keys are variable names, values are dictionaries
        of assumption flags and values. If no assumptions are provided, the default assumptions are used. For more
        details, see the Sympy documentation.
    """

    # TODO: Add checks that model blocks follow the correct format and fail more helpfully.

    block_dict = dict()
    text, tryreduce = process_special_block_text(text, "tryreduce")
    text, options = process_special_block_text(text, "options")
    text, assumptions = process_special_block_text(text, "assumptions")

    assumptions = cast(ASSUMPTION_DICT, assumptions)
    tryreduce = cast(list[str], tryreduce)
    options = cast(dict[str, str], options)

    gcn_blocks = [block for block in text.split("block") if len(block) > 0]
    for block in gcn_blocks:
        tokens = block.strip().split()
        name = tokens.pop(0)
        block_dict[name] = " ".join(tokens)

    return block_dict, options, tryreduce, assumptions


def parsed_block_to_dict(block: str) -> dict[str, list[list[str]]]:
    """
    Extracts the block components and equations from a pre-processed model block.

    Parameters
    ----------
    block: str
        Pre-processed text of the standard model block format.

    Returns
    -------
    block_dict: dict[str, list[list[str]]]
        A dict of lists, containing lists of equation tokens. Keys are the block components found
        in the block string. Equations are represented as lists of tokens, while sub-blocks are lists of equation lists.

    Example
    -------

    .. code::python

    >> Input: {definition { u[] = log ( C[] ) + log( L[] ); }; objective { U[] = u[] + beta * E[][U[1]] ;} };
    >> Output: dict("definition" = ["u[]", "=", "log", "(", "C[]", ")", "+", "log", "(", "L[]", ")", ";"],
                    "objective" = ["U[]", "=", "u[]", "+", "beta", "*", "E[][U[1]]", ";"])
    """
    block_dict = defaultdict(list)
    parsed_block = next(iter(pp.nestedExpr("{", "};").parseString(block).asList()))
    current_key = parsed_block.pop(0)

    if isinstance(current_key, list):
        # block[0] is an equation, should not be possible
        raise GCNSyntaxError(block_name=block, key=current_key)

    validate_key(key=current_key, block_name=block)

    for element in parsed_block:
        if isinstance(element, str):
            current_key = element
            validate_key(key=current_key, block_name=block)
        else:
            equations_list = rebuild_eqs_from_parser_output(element)
            block_dict[current_key].extend(equations_list)

    return block_dict
