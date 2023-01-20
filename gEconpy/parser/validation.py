from typing import List, Tuple, Union

from gEconpy.exceptions.exceptions import InvalidComponentNameException
from gEconpy.parser.constants import BLOCK_COMPONENTS


def block_is_empty(block: str) -> bool:
    """
    Check whether a model block is empty, i.e. contains no flags, variables, or equations.

    Parameters
    ----------
    block : str
        Raw text of a model block.

    Returns
    -------
    bool
        Whether the block is empty or not.
    """

    return block.strip() == "{ };"


def validate_key(key: str, block_name: str) -> None:
    """
    Check that the component name matches something in the list of valid block components.

    The R implementation of gEcon only allows the names in BLOCK_COMPONENTS to be used inside model blocks.
    This function checks that a component name matches something in that list, and raises an error if not.

    Parameters
    ----------
    block_name : str
        The name of the block.
    key : str
        A block sub-component name.

    Returns
    -------
    None

    Raises
    ------
    InvalidComponentNameException
        If the component name is invalid.

    # TODO: Allow arbitrary component names? Is there any need to?
    """
    if key.upper() not in BLOCK_COMPONENTS:
        valid_names = ", ".join(BLOCK_COMPONENTS)
        error = f"Valid sub-block names are: {valid_names}\n"
        error += f"Found: {key} in block {block_name}"

        raise InvalidComponentNameException(
            component_name=key, block_name=block_name, message=error
        )


def jaccard_distance(s: str, d: str) -> float:
    """
    Calculate the Jaccard distance between two strings.

    The Jaccard distance is defined as the size of the intersection of two sets divided by the size of their union.
    For example, the Jaccard distance between the sets {"C", "A", "T"} and {"C", "U", "T"} is 1/2 because the
    intersection of these two sets is {"C", "T"} (of size 2) and the union is {"C", "A", "T", "U"} (of size 4).
    Therefore, the Jaccard distance is 2/4 = 1/2.

    Parameters
    ----------
    s : str
        The first string.
    d : str
        The second string.

    Returns
    -------
    float
        The Jaccard distance between the two strings.
    """

    s = set(s)
    d = set(d)
    union = len(s.union(d))
    intersection = len(s.intersection(d))

    return intersection / union


def elementwise_jaccard_distance(s: str, l: List[str]) -> List[float]:
    """
    Calculate the Jaccard distance between a string and each element in a list of strings.

    Parameters
    ----------
    s : str
        The string to compare against.
    l : list of str
        The list of strings to compare to `s`.

    Returns
    -------
    list of float
        A list of the Jaccard distances between `s` and each element in `l`.
    """
    return [jaccard_distance(s, element) for element in l]


def find_typos_and_guesses(
    user_inputs: List[str], valid_inputs: List[str], match_threshold: float = 0.8
) -> Tuple[Union[str, None], Union[str, None]]:
    """
    Find the best matching suggestion from a list of valid inputs for a list of invalid user inputs.

    Parameters
    ----------
    user_inputs : list of str
        The list of invalid user inputs.
    valid_inputs : list of str
        The list of valid inputs.
    match_threshold : float, optional
        The minimum Jaccard distance required to consider a user input a typo. Default is 0.8.

    Returns
    -------
    tuple of (str or None, str or None)
        A tuple containing the best matching valid input and the user input that may be a typo, if they are above the
        match threshold. If no user input is above the threshold, both elements of the tuple will be None.

    TODO: Tune match_threshold
    """

    best_guess = max(
        valid_inputs, key=lambda x: elementwise_jaccard_distance(x, user_inputs)
    )
    maybe_typo = max(
        user_inputs, key=lambda x: elementwise_jaccard_distance(x, valid_inputs)
    )

    if jaccard_distance(best_guess, maybe_typo) < match_threshold:
        return None, None

    return best_guess, maybe_typo
