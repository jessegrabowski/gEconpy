from gEcon.parser.constants import BLOCK_COMPONENTS
from typing import List, Tuple, Union
from gEcon.exceptions.exceptions import InvalidComponentNameException


def block_is_empty(block: str) -> bool:
    """
    :param block: str, raw text of a model block
    :return: bool

    Check whether a model block is empty, i.e. contains no flags, variables, or equations.
    """
    return block.strip() == '{ };'


def validate_key(key: str, block_name: str) -> None:
    """
    :param block_name: str, optional, the name of the block
    :param key: str, a block sub-component name
    :return: None

    The R implementation of gEcon only allows the names in BLOCK_COMPONENTS to be used inside model blocks.
    This function checks that a component name matches something in that list, and raises an error if not.

    TODO: Allow arbitrary component names? Is there any need to?
    """

    if key.upper() not in BLOCK_COMPONENTS:
        valid_names = ', '.join(BLOCK_COMPONENTS)
        error = f'Valid sub-block names are: {valid_names}\n'
        error += f'Found: {key} in block {block_name}'

        raise InvalidComponentNameException(component_name=key, block_name=block_name, message=error)


def jaccard_distance(s: str, d: str) -> float:
    s = set(s)
    d = set(d)
    union = len(s.union(d))
    intersection = len(s.intersection(d))

    return intersection / union


def elementwise_jaccard_distance(s: str, l: List[str]) -> List[float]:
    return [jaccard_distance(s, element) for element in l]


def find_typos_and_guesses(user_inputs: List[str],
                           valid_inputs: List[str],
                           match_threshold: float = 0.8) -> Tuple[Union[str, None], Union[str, None]]:
    # TODO: Tune match_threshold
    best_guess = max(valid_inputs,
                     key=lambda x: elementwise_jaccard_distance(x, user_inputs))
    maybe_typo = max(user_inputs,
                     key=lambda x: elementwise_jaccard_distance(x, valid_inputs))

    if jaccard_distance(best_guess, maybe_typo) < match_threshold:
        return None, None

    return best_guess, maybe_typo
