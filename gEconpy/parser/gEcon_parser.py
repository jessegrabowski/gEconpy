import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import pyparsing

from gEconpy.exceptions.exceptions import GCNSyntaxError
from gEconpy.parser.constants import SPECIAL_BLOCK_NAMES, SYMPY_ASSUMPTIONS
from gEconpy.parser.parse_equations import rebuild_eqs_from_parser_output
from gEconpy.parser.parse_plaintext import (
    add_spaces_around_operators,
    delete_block,
    extract_distributions,
    remove_comments,
    remove_extra_spaces,
    remove_newlines_and_tabs,
)
from gEconpy.parser.validation import block_is_empty, validate_key


def block_to_clean_list(block: str) -> List[str]:
    """
    Processes a block of text by removing certain characters, and then splitting it into a list of strings.

    Parameters
    ----------
    block : str
        The block of text to process.

    Returns
    -------
    List[str]
        The processed list of strings.
    """

    block = re.sub("[{};]", "", block)
    block = remove_extra_spaces(block).strip()
    block = [x.replace(",", "").strip() for x in block.split()]

    return block


def extract_assumption_sub_blocks(block: List[str]) -> Dict[str, List[str]]:
    """
    Extracts the special "Assumptions" block from the GCN file. Saves each user-provided assumption to a dictionary,
    along with all variables associated to that assumption.

    Parameters
    ----------
    block : List[str]
        The block of text to process.

    Returns
    -------
    Dict[str, List[str]]
        A dictionary containing assumptions and variables, with the assumption names as keys and associated variables
        as values.
    """
    sub_blocks = {}
    assumption_idxs = {}

    for assumption in SYMPY_ASSUMPTIONS:
        if assumption in block:
            assumption_idxs[assumption] = block.index(assumption)

    used_assumptions = list(assumption_idxs.keys())
    if len(used_assumptions) == 0:
        return sub_blocks

    sorted_assumptions = sorted(used_assumptions, key=assumption_idxs.get)
    start_idx = 0

    for i in range(1, len(sorted_assumptions)):
        curr_assumption = sorted_assumptions[i - 1]
        next_assumption = sorted_assumptions[i]
        end_idx = assumption_idxs[next_assumption]

        block_slice = slice(start_idx + 1, end_idx)
        sub_blocks[curr_assumption] = block[block_slice]

        start_idx = end_idx

    i = len(sorted_assumptions) - 1
    curr_assumption = sorted_assumptions[i]
    block_slice = slice(start_idx + 1, None)

    sub_blocks[curr_assumption] = block[block_slice]

    return sub_blocks


def create_assumption_kwargs(
    assumption_dicts: Dict[str, List[str]]
) -> Dict[str, Dict[str, bool]]:
    """
    Extracts assumption flags from `assumption_dicts` and returns them in a dictionary keyed by variable names.

    Parameters
    ----------
    assumption_dicts : Dict[str, List[str]]
        A dictionary containing assumptions and variables, with the assumption names as keys and associated variables
        as values.

    Returns
    -------
    Dict[str, Dict[str, bool]]
        A dictionary of flags and values keyed by variable names.
    """

    assumption_kwargs = defaultdict(lambda: {})

    for assumption, variable_list in assumption_dicts.items():
        for var in variable_list:
            base_var = re.sub(r"\[\]", "", var)
            assumption_kwargs[base_var][assumption] = True

    return assumption_kwargs


def preprocess_gcn(gcn_raw: str) -> Tuple[str, Dict[str, str]]:
    """
    Preprocesses `gcn_raw` and returns the result.

    Parameters
    ----------
    gcn_raw : str
        Raw model file returned by function `load_gcn`.

    Returns
    -------
    Tuple[str, Dict[str, str]]
        Model file with basic preprocessing and prior distributions, respectively.
    """

    gcn_processed = remove_comments(gcn_raw)
    gcn_processed, prior_dict = extract_distributions(gcn_processed)
    gcn_processed = remove_newlines_and_tabs(gcn_processed)
    gcn_processed = add_spaces_around_operators(gcn_processed)

    return gcn_processed, prior_dict


def parse_options_flags(options: str) -> Optional[Dict[str, bool]]:
    """
    Extracts flags and values from `options`.

    Parameters
    ----------
    options : str
        Text from the "options" block of a model file.

    Returns
    -------
    Optional[Dict[str, bool]]
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
        value = (
            True
            if value.lower() == "true"
            else False
            if value.lower() == "false"
            else value
        )

        result[flag.strip()] = value

    return result


def extract_special_block(text: str, block_name: str) -> Dict[str, List[str]]:
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
        block_name: defaultdict(lambda: defaultdict(lambda: dict))
        if block_name == "assumptions"
        else None
    }

    if block_name not in text:
        return result

    block = re.search(block_name + " {.*?" + "};", text)[0]
    block = block.replace(block_name, "")

    if block_is_empty(block):
        return result

    elif block_name == "options":
        block = parse_options_flags(block)

    elif block_name == "tryreduce":
        block = block_to_clean_list(block)

    elif block_name == "assumptions":
        block = re.search(block_name + " {.*?" + "}; };", text)[0]
        block = block.replace(block_name, "")
        block = block_to_clean_list(block)
        block = extract_assumption_sub_blocks(block)
        block = create_assumption_kwargs(block)

    result[block_name] = block

    return result


def split_gcn_into_block_dictionary(text: str) -> Dict[str, str]:
    """
    Split the preprocessed GCN text by block and stores the results in a dictionary.

    Parameters
    ----------
    text : str
        Text of a model file after text processing using the preprocess_gcn function. I.e., comments are expected to be
        removed, all tokens are splittable on single white spaces, and blocks are wrapped by { };

    Returns
    -------
    Dict[str, str]
        A "block dictionary" with key, value pairs of block_name:block_text. Special blocks are processed first
        (currently "options" and "tryreduce"), then deleted. Normal model blocks are assumed to follow a standard format
        of block NAME { component_1 { Equations }; component_2 { ... }; };

    TODO: Add checks that model blocks follow the correct format and fail more helpfully.
    """
    results = dict()

    for name in SPECIAL_BLOCK_NAMES:
        name = name.lower()
        result = extract_special_block(text, name)
        results.update(result)
        text = delete_block(text, name)

    gcn_blocks = [block for block in text.split("block") if len(block) > 0]
    for block in gcn_blocks:
        tokens = block.strip().split()
        name = tokens[0]
        results[name] = " ".join(tokens[1:])

    return results


def parsed_block_to_dict(block: str) -> Dict[str, List[List[str]]]:
    """
    Extracts the block components and equations from a pre-processed model block.

    Parameters
    ----------
    block: str
        Pre-processed text of the standard model block format.

    Returns
    -------
    Dict[str, List[List[str]]]
        A defaultdict of lists, containing lists of equation tokens. Keys are the block components found
        in the block string. Equations are represented as lists of tokens, while sub-blocks are lists of equation lists.

    Example:
    >> Input: {definition { u[] = log ( C[] ) + log( L[] ); }; objective { U[] = u[] + beta * E[][U[1]] ;} };
    >> Output: dict("definition" = ["u[]", "=", "log", "(", "C[]", ")", "+", "log", "(", "L[]", ")", ";"],
                    "objective" = ["U[]", "=", "u[]", "+", "beta", "*", "E[][U[1]]", ";"])
    """
    block_dict = defaultdict(list)
    parsed_block = pyparsing.nestedExpr("{", "};").parseString(block).asList()[0]
    current_key = parsed_block[0]

    if isinstance(current_key, list):
        # block[0] is an equation, should not be possible
        raise GCNSyntaxError(block_name=block, key=current_key)

    validate_key(key=current_key, block_name=block)

    for element in parsed_block[1:]:
        if isinstance(element, str):
            current_key = element
            validate_key(key=current_key, block_name=block)
        else:
            equations_list = rebuild_eqs_from_parser_output(element)
            block_dict[current_key].extend(equations_list)

    return block_dict


def make_assumption_dict(raw_blocks: Dict[str, str]) -> Dict[str, str]:
    pass
