from warnings import warn

import sympy as sp

from gEconpy.classes.containers import SymbolDictionary
from gEconpy.classes.time_aware_symbol import TimeAwareSymbol
from gEconpy.exceptions import (
    ExtraParameterError,
    ExtraParameterWarning,
    InvalidComponentNameException,
    OrphanParameterError,
)
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

        raise InvalidComponentNameException(component_name=key, block_name=block_name, message=error)


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


def elementwise_jaccard_distance(s: str, elements: list[str]) -> list[float]:
    """
    Calculate the Jaccard distance between a string and each element in a list of strings.

    Parameters
    ----------
    s : str
        The string to compare against.
    elements : list of str
        The list of strings to compare to `s`.

    Returns
    -------
    list of float
        A list of the Jaccard distances between `s` and each element in `l`.
    """
    return [jaccard_distance(s, element) for element in elements]


def find_typos_and_guesses(
    user_inputs: list[str], valid_inputs: list[str], match_threshold: float = 0.8
) -> tuple[str | None, str | None]:
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
    """
    #     TODO: Tune match_threshold

    best_guess = max(valid_inputs, key=lambda x: elementwise_jaccard_distance(x, user_inputs))
    maybe_typo = max(user_inputs, key=lambda x: elementwise_jaccard_distance(x, valid_inputs))

    if jaccard_distance(best_guess, maybe_typo) < match_threshold:
        return None, None

    return best_guess, maybe_typo


def check_for_orphan_params(equations: list[sp.Expr], param_dict: SymbolDictionary) -> None:
    """
    Check for parameters used in equations but not defined in param_dict.

    Parameters
    ----------
    equations : list of sp.Expr
        Model equations.
    param_dict : SymbolDictionary
        Dictionary of defined parameters.

    Raises
    ------
    OrphanParameterError
        If orphan parameters are found.
    """
    parameters = list(param_dict.to_sympy().keys())
    param_equations = [x for x in param_dict.values() if isinstance(x, sp.Expr)]

    orphans = [
        atom
        for eq in equations
        for atom in eq.atoms()
        if (
            isinstance(atom, sp.Symbol)
            and not isinstance(atom, TimeAwareSymbol)
            and atom not in parameters
            and not any(eq.has(atom) for eq in param_equations)
        )
    ]

    if len(orphans) > 0:
        raise OrphanParameterError(orphans)


def check_for_extra_params(
    equations: list[sp.Expr],
    param_dict: SymbolDictionary,
    on_unused_parameters: str = "raise",
    distribution_atoms: set[sp.Symbol] | None = None,
) -> None:
    """
    Check for parameters defined but not used in any equations.

    Parameters
    ----------
    equations : list of sp.Expr
        Model equations.
    param_dict : SymbolDictionary
        Dictionary of defined parameters.
    on_unused_parameters : str
        How to handle unused parameters: "raise", "warn", or "ignore".
    distribution_atoms : set of sp.Symbol, optional
        Atoms used in distribution definitions (e.g., shock standard deviations).

    Raises
    ------
    ExtraParameterError
        If extra parameters are found and on_unused_parameters="raise".
    """
    parameters = list(param_dict.to_sympy().keys())
    param_equations = [x for x in param_dict.values() if isinstance(x, sp.Expr)]

    all_atoms = {atom for eq in equations + param_equations for atom in eq.atoms()}
    if distribution_atoms:
        all_atoms |= distribution_atoms
    extras = [parameter for parameter in parameters if parameter not in all_atoms]

    if len(extras) > 0:
        if on_unused_parameters == "raise":
            raise ExtraParameterError(extras)
        if on_unused_parameters == "warn":
            warn(ExtraParameterWarning(extras), stacklevel=2)


def validate_results(
    equations: list[sp.Expr],
    steady_state_relationships: list[sp.Expr],
    param_dict: SymbolDictionary,
    calib_dict: SymbolDictionary,
    deterministic_dict: SymbolDictionary,
    on_unused_parameters: str = "raise",
    distributions: SymbolDictionary | None = None,
    distribution_param_names: set[str] | None = None,
) -> None:
    """
    Validate parsed model results for orphan and extra parameters.

    Parameters
    ----------
    equations : list of sp.Expr
        Model equations.
    steady_state_relationships : list of sp.Expr
        Steady-state equations.
    param_dict : SymbolDictionary
        Dictionary of parameters.
    calib_dict : SymbolDictionary
        Dictionary of calibrating equations.
    deterministic_dict : SymbolDictionary
        Dictionary of deterministic relationships.
    on_unused_parameters : str
        How to handle unused parameters: "raise", "warn", or "ignore".
    distributions : SymbolDictionary, optional
        Dictionary of distributions (for shock priors, etc.).
    distribution_param_names : set of str, optional
        Parameter names used in distribution definitions (e.g., shock standard deviations).
    """
    joint_dict = param_dict | calib_dict | deterministic_dict
    check_for_orphan_params(equations + steady_state_relationships, joint_dict)

    # Extract atoms used in distribution parameters
    distribution_atoms: set[sp.Symbol] = set()
    if distributions:
        for dist in distributions.values():
            if hasattr(dist, "args"):
                for arg in dist.args:
                    if isinstance(arg, sp.Expr):
                        distribution_atoms |= arg.atoms(sp.Symbol)

    # Also add parameter names referenced in shock distributions
    if distribution_param_names:
        # Get sympy version of joint_dict to match symbols properly
        sympy_dict = joint_dict.to_sympy()
        for param_name in distribution_param_names:
            # Find matching symbol in sympy_dict
            for sym in sympy_dict:
                if str(sym) == param_name:
                    distribution_atoms.add(sym)
                    break

    check_for_extra_params(
        equations + steady_state_relationships,
        joint_dict,
        on_unused_parameters,
        distribution_atoms=distribution_atoms,
    )
