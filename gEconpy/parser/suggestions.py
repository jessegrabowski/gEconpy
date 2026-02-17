from collections.abc import Iterable
from difflib import SequenceMatcher

from gEconpy.parser.constants import (
    BLOCK_COMPONENTS,
    GCN_ASSUMPTIONS,
    PRELIZ_DIST_WRAPPERS,
    PRELIZ_DISTS,
    SPECIAL_BLOCK_NAMES,
)

DEFAULT_SIMILARITY_THRESHOLD = 0.6

KNOWN_DISTRIBUTIONS = frozenset(PRELIZ_DISTS)

KNOWN_WRAPPERS = frozenset(PRELIZ_DIST_WRAPPERS)

KNOWN_COMPONENTS = frozenset(c.lower() for c in BLOCK_COMPONENTS)

KNOWN_SPECIAL_BLOCKS = frozenset(c.lower() for c in SPECIAL_BLOCK_NAMES)

KNOWN_ASSUMPTIONS = frozenset(GCN_ASSUMPTIONS)


def _similarity_ratio(a: str, b: str) -> float:
    """
    Compute similarity ratio between two strings.

    Uses SequenceMatcher for a balance of speed and quality.
    Returns a value between 0.0 (no similarity) and 1.0 (identical).
    """
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def find_similar_names(
    name: str,
    candidates: Iterable[str],
    max_results: int = 3,
    min_similarity: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> list[str]:
    """
    Find similar names from a set of candidates using edit distance.

    Parameters
    ----------
    name : str
        The name to find similar matches for.
    candidates : iterable of str
        The set of valid names to search.
    max_results : int
        Maximum number of suggestions to return.
    min_similarity : float
        Minimum similarity ratio (0.0 to 1.0) to include in results.

    Returns
    -------
    suggestions : list of str
        Similar names, sorted by similarity (most similar first).
        Returns empty list if exact match exists or no similar names found.
    """
    name_lower = name.lower()
    candidates_list = list(candidates)

    # Check for exact match (case-insensitive)
    for candidate in candidates_list:
        if candidate.lower() == name_lower:
            return []

    # Compute similarities
    scored = []
    for candidate in candidates_list:
        ratio = _similarity_ratio(name, candidate)
        if ratio >= min_similarity:
            scored.append((ratio, candidate))

    # Sort by similarity (descending) then alphabetically
    scored.sort(key=lambda x: (-x[0], x[1]))

    return [candidate for _, candidate in scored[:max_results]]


def suggest_distribution(name: str) -> list[str]:
    """
    Suggest corrections for an unknown distribution name.

    Parameters
    ----------
    name : str
        The unknown distribution name.

    Returns
    -------
    suggestions : list of str
        Suggested distribution names from the known set.
    """
    return find_similar_names(name, KNOWN_DISTRIBUTIONS)


def suggest_wrapper(name: str) -> list[str]:
    """
    Suggest corrections for an unknown wrapper name.

    Parameters
    ----------
    name : str
        The unknown wrapper name.

    Returns
    -------
    suggestions : list of str
        Suggested wrapper names.
    """
    return find_similar_names(name, KNOWN_WRAPPERS)


def suggest_block_component(name: str) -> list[str]:
    """
    Suggest corrections for a misspelled block component name.

    Parameters
    ----------
    name : str
        The misspelled component name.

    Returns
    -------
    suggestions : list of str
        Suggested component names.
    """
    return find_similar_names(name, KNOWN_COMPONENTS)


def suggest_special_block(name: str) -> list[str]:
    """
    Suggest corrections for a misspelled special block name.

    Parameters
    ----------
    name : str
        The misspelled block name.

    Returns
    -------
    suggestions : list of str
        Suggested special block names.
    """
    return find_similar_names(name, KNOWN_SPECIAL_BLOCKS)


def suggest_assumption(name: str) -> list[str]:
    """
    Suggest corrections for a misspelled assumption type.

    Parameters
    ----------
    name : str
        The misspelled assumption type.

    Returns
    -------
    suggestions : list of str
        Suggested assumption types.
    """
    return find_similar_names(name, KNOWN_ASSUMPTIONS)
