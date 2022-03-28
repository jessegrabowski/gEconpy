import pyparsing
import re
from typing import Optional, Dict, List, Tuple
from collections import defaultdict
import sympy as sp

from gEcon.parser.constants import SPECIAL_BLOCK_NAMES
from gEcon.parser.parse_plaintext import remove_comments, remove_newlines_and_tabs, add_spaces_around_operators, \
    remove_extra_spaces, delete_block, extract_distributions
from gEcon.parser.validation import block_is_empty, validate_key
from gEcon.parser.parse_equations import rebuild_eqs_from_parser_output
from gEcon.exceptions.exceptions import GCNSyntaxError


def preprocess_gcn(gcn_raw: str) -> Tuple[str, Dict[str, str]]:
    """
    :param gcn_raw: str, raw model file returned by function load_gcn
    :return: str, model file with basic preprocessing

    Helper function to wrap all basic steps text processing steps necessary prior to parsing the model
    equations themselves.
    """

    gcn_processed = remove_comments(gcn_raw)
    gcn_processed, prior_dict = extract_distributions(gcn_processed)
    gcn_processed = remove_newlines_and_tabs(gcn_processed)
    gcn_processed = add_spaces_around_operators(gcn_processed)

    return gcn_processed, prior_dict


def parse_options_flags(options: str) -> Optional[Dict[str, bool]]:
    """
    :param options: str, text from the "options" block of a model file
    :return: dict or None, a dictionary of flags and values if they exist, or None if no options were found.

    Function to extract the flags and values from an options block.

    NOTE: Currently nothing is done with these values, and this step is primarily to ensure backwards compatibility with
    .GCN files written for the gEcon R package.
    """
    result = dict()
    options = re.sub('[{}]', '', options)
    options = [line.strip() for line in options.split(';') if len(line.strip()) > 0]

    if len(options) == 0:
        return

    for option in options:
        flag, value = option.split('=')
        value = value.replace(';', '').strip()
        value = True if value.lower() == 'true' else False if value.lower() == 'false' else value

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
    result = {block_name: None}

    if block_name not in text:
        return result

    block = re.search(block_name + ' {.*?};', text)[0]
    block = block.replace(block_name, '')

    if block_is_empty(block):
        block = None

    elif block_name == 'options':
        block = parse_options_flags(block)

    elif block_name == 'tryreduce':
        block = re.sub('[{};]', '', block)
        block = remove_extra_spaces(block).strip()
        block = [x.strip() for x in block.split(',')]

    result[block_name] = block

    return result


def split_gcn_into_block_dictionary(text: str) -> Dict[str, str]:
    """
    :param text: str, text of a model file after text processing using the preprocess_gcn function. I.e., comments are
                 expected to be removed, all tokens are splittable on single white spaces, and blocks are wrapped by
                 { };
    :return: dict, a "block dictionary" with key, value pairs of block_name:block_text.

    This function splits the preprocessed GCN text by block and stores the results in a dictionary. Special blocks are
    processed first (currently "options" and "tryreduce"), then deleted. Normal model blocks are assumed to follow
    a standard format of block NAME { component_1 { Equations }; component_2 { ... }; };

    TODO: Add checks that model blocks follow the correct format and fail more helpfully.
    """
    results = dict()

    for name in SPECIAL_BLOCK_NAMES:
        name = name.lower()
        result = extract_special_block(text, name)
        results.update(result)
        text = delete_block(text, name)

    gcn_blocks = [block for block in text.split('block') if len(block) > 0]
    for block in gcn_blocks:
        tokens = block.strip().split()
        name = tokens[0]
        results[name] = ' '.join(tokens[1:])

    return results


def parsed_block_to_dict(block: str) -> Dict[str, List[List[str]]]:
    """
    :param block: str, pre-processed text of the standard model block format.
    :return: dict, a defaultdict of lists, containing lists of equation tokens. Keys are the block components found
            in the block string.

    Divide model blocks into a dictionary of component sub-blocks and the equations found in that sub-block. Equations
    are represented as lists of tokens, while sub-blocks are lists of equation lists.

    Example:
    >> Input: {definition { u[] = log ( C[] ) + log( L[] ); }; objective { U[] = u[] + beta * E[][U[1]] ;} };
    >> Output: dict("definition" = ["u[]", "=", "log", "(", "C[]", ")", "+", "log", "(", "L[]", ")"],
                    "objective"  = ["U[]", "=", "u[]", "+", "beta", "*", "E[][", "U[1]]", ])
    """
    block_dict = defaultdict(list)
    parsed_block = pyparsing.nestedExpr('{', '};').parseString(block).asList()[0]
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
