from collections.abc import Iterable
from difflib import SequenceMatcher

from gEconpy.parser.constants import (
    KNOWN_ASSUMPTIONS,
    KNOWN_COMPONENTS,
    KNOWN_DISTRIBUTIONS,
    KNOWN_SPECIAL_BLOCKS,
    KNOWN_WRAPPERS,
)

DEFAULT_SIMILARITY_THRESHOLD = 0.6


def find_similar_names(
    name: str,
    candidates: Iterable[str],
    max_results: int = 3,
    min_similarity: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> list[str]:
    """
    Rank the candidates that resemble a misspelled name, for "did you mean" suggestions.

    Parameters
    ----------
    name : str
        The name to find matches for.
    candidates : iterable of str
        The valid names to search.
    max_results : int, optional
        Maximum number of suggestions to return. Defaults to 3.
    min_similarity : float, optional
        Minimum :class:`difflib.SequenceMatcher` ratio, between 0 and 1, for a candidate to count as similar.
        Defaults to 0.6.

    Returns
    -------
    suggestions : list of str
        Similar candidates, most similar first, with ties broken alphabetically. Empty when a candidate matches
        ``name`` exactly, ignoring case, or when no candidate is similar enough.
    """
    name_lower = name.lower()
    candidates_list = list(candidates)

    if any(candidate.lower() == name_lower for candidate in candidates_list):
        return []

    scored = [(_similarity_ratio(name, candidate), candidate) for candidate in candidates_list]
    similar = [(ratio, candidate) for ratio, candidate in scored if ratio >= min_similarity]
    similar.sort(key=lambda pair: (-pair[0], pair[1]))

    return [candidate for _, candidate in similar[:max_results]]


def suggest_distribution(name: str) -> list[str]:
    """
    Suggest PreliZ distribution names resembling an unknown one.

    Parameters
    ----------
    name : str
        The unknown distribution name.

    Returns
    -------
    suggestions : list of str
        Known distribution names, most similar first.
    """
    return find_similar_names(name, KNOWN_DISTRIBUTIONS)


def suggest_wrapper(name: str) -> list[str]:
    """
    Suggest distribution wrapper names resembling an unknown one.

    Parameters
    ----------
    name : str
        The unknown wrapper name.

    Returns
    -------
    suggestions : list of str
        Known wrapper names, most similar first.
    """
    return find_similar_names(name, KNOWN_WRAPPERS)


def suggest_block_component(name: str) -> list[str]:
    """
    Suggest block component names resembling a misspelled one.

    Parameters
    ----------
    name : str
        The misspelled component name.

    Returns
    -------
    suggestions : list of str
        Known component names, most similar first.
    """
    return find_similar_names(name, KNOWN_COMPONENTS)


def suggest_special_block(name: str) -> list[str]:
    """
    Suggest special block names resembling a misspelled one.

    Parameters
    ----------
    name : str
        The misspelled block name.

    Returns
    -------
    suggestions : list of str
        Known special block names, most similar first.
    """
    return find_similar_names(name, KNOWN_SPECIAL_BLOCKS)


def suggest_assumption(name: str) -> list[str]:
    """
    Suggest assumption names resembling a misspelled one.

    Parameters
    ----------
    name : str
        The misspelled assumption name.

    Returns
    -------
    suggestions : list of str
        Known assumption names, most similar first.
    """
    return find_similar_names(name, KNOWN_ASSUMPTIONS)


def _similarity_ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()
