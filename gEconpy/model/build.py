import logging
import sys

from pathlib import Path
from warnings import warn

import sympy as sp

from pymc.pytensorf import rewrite_pregrad
from pytensor import graph_replace

from gEconpy.classes.containers import SymbolDictionary
from gEconpy.classes.distributions import CompositeDistribution
from gEconpy.classes.time_aware_symbol import TimeAwareSymbol
from gEconpy.exceptions import ExtraParameterError, ExtraParameterWarning, OrphanParameterError
from gEconpy.model.compile import make_cache_key
from gEconpy.model.model import Model
from gEconpy.model.parameters import compile_param_dict_func
from gEconpy.model.perturbation import linearize_model as _linearize_model
from gEconpy.model.simplification import simplify_constants, simplify_tryreduce
from gEconpy.model.statespace import DSGEStateSpace
from gEconpy.model.steady_state import (
    ERROR_FUNCTIONS,
    _ss_residual_to_pytensor,
    build_minimize_graphs,
    build_root_graphs,
    compile_known_ss,
    propagate_steady_state_through_identities,
    simplify_provided_ss_equations,
    system_to_steady_state,
)
from gEconpy.model.timing import natural_sort_key
from gEconpy.parser.errors import GCNErrorCollection, GCNParseError
from gEconpy.parser.formatting import ErrorFormatter
from gEconpy.parser.loader import load_gcn_file
from gEconpy.utilities import get_name, substitute_repeatedly

_log = logging.getLogger(__name__)


def _print_parse_error(error: GCNParseError | GCNErrorCollection, gcn_path: Path) -> None:
    """Format and print a parse error to stderr."""
    formatter = ErrorFormatter()
    if isinstance(error, GCNErrorCollection):
        print(formatter.format_error_collection(error), file=sys.stderr)
    else:
        source = gcn_path.read_text() if gcn_path.exists() else None
        print(formatter.format_error(error, source), file=sys.stderr)


def split_out_hyper_params(
    param_dict: SymbolDictionary[str, float],
    shock_prior: SymbolDictionary[str, CompositeDistribution],
) -> tuple[SymbolDictionary[str, float], SymbolDictionary[str, float]]:
    """
    Separate shock hyper-parameters from the parameter dictionary.

    Hyper-parameters are parameters that appear in the definitions of shock prior distributions (e.g., the standard
    deviation of an AR(1) shock). They are removed from the main parameter dictionary and returned separately.

    Parameters
    ----------
    param_dict : SymbolDictionary
        Dictionary of initial parameter values.
    shock_prior : SymbolDictionary
        Dictionary of shock priors.

    Returns
    -------
    param_dict : SymbolDictionary
        Parameter dictionary with hyper-parameters removed.
    hyper_param_dict : SymbolDictionary
        Dictionary containing only the hyper-parameters and their values.
    """
    new_param_dict = param_dict.copy()
    hyper_param_dict = SymbolDictionary()

    all_hyper_params = [param for dist in shock_prior.values() for param in dist.param_name_to_hyper_name.values()]

    for param in all_hyper_params:
        if param in new_param_dict:
            del new_param_dict[param]
            hyper_param_dict[param] = param_dict[param]

    return new_param_dict, hyper_param_dict


def _block_dict_to_sub_dict(block_dict: dict) -> dict[sp.Expr, sp.Expr]:
    """Extract a substitution dictionary from block equations for tryreduce simplification."""
    sub_dict = {}
    for block in block_dict.values():
        for group in ["identities", "objective", "constraints"]:
            group_dict = getattr(block, group, None)
            if group_dict is not None:
                for eq in group_dict.values():
                    sub_dict[eq.lhs] = eq.rhs
    return sub_dict


def check_for_orphan_params(equations: list[sp.Expr], param_dict: SymbolDictionary) -> None:
    """
    Check for parameters used in equations but not defined in the parameter dictionary.

    Parameters
    ----------
    equations : list of sp.Expr
        Model equations.
    param_dict : SymbolDictionary
        Dictionary of defined parameters.
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

    if orphans:
        raise OrphanParameterError(orphans)


def check_for_extra_params(
    equations: list[sp.Expr],
    param_dict: SymbolDictionary,
    on_unused_parameters: str = "raise",
    distribution_atoms: set[sp.Symbol] | None = None,
) -> None:
    """
    Check for parameters defined but not used in any equation or distribution.

    Parameters
    ----------
    equations : list of sp.Expr
        Model equations.
    param_dict : SymbolDictionary
        Dictionary of defined parameters.
    on_unused_parameters : ``'raise'``, ``'warn'``, or ``'ignore'``
        How to handle unused parameters.
    distribution_atoms : set of sp.Symbol, optional
        Additional atoms referenced by prior distributions that should not be flagged as unused.
    """
    parameters = list(param_dict.to_sympy().keys())
    param_equations = [x for x in param_dict.values() if isinstance(x, sp.Expr)]

    all_atoms = {atom for eq in equations + param_equations for atom in eq.atoms()}
    if distribution_atoms:
        all_atoms |= distribution_atoms

    extras = [p for p in parameters if p not in all_atoms]

    if extras:
        if on_unused_parameters == "raise":
            raise ExtraParameterError(extras)
        if on_unused_parameters == "warn":
            warn(ExtraParameterWarning(extras), stacklevel=2)


def _collect_distribution_atoms(
    distributions: SymbolDictionary | None,
    distribution_param_names: set[str] | None,
    joint_dict: SymbolDictionary,
) -> set[sp.Symbol]:
    """Collect all sympy atoms referenced by prior distributions so they are not flagged as unused."""
    atoms: set[sp.Symbol] = set()

    if distributions:
        for dist in distributions.values():
            if hasattr(dist, "args"):
                for arg in dist.args:
                    if isinstance(arg, sp.Expr):
                        atoms |= arg.atoms(sp.Symbol)

    if distribution_param_names:
        sympy_dict = joint_dict.to_sympy()
        name_to_sym = {str(sym): sym for sym in sympy_dict}
        for param_name in distribution_param_names:
            if param_name in name_to_sym:
                atoms.add(name_to_sym[param_name])

    return atoms


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
    Validate that all parameters are both defined and used.

    Checks for orphan parameters (used in equations but not defined) and extra parameters (defined but not used in
    any equation or distribution).

    Parameters
    ----------
    equations : list of sp.Expr
        Model equations.
    steady_state_relationships : list of sp.Expr
        Steady-state equations.
    param_dict : SymbolDictionary
        Free parameters.
    calib_dict : SymbolDictionary
        Calibrating equations.
    deterministic_dict : SymbolDictionary
        Deterministic relationships.
    on_unused_parameters : ``'raise'``, ``'warn'``, or ``'ignore'``
        How to handle unused parameters.
    distributions : SymbolDictionary, optional
        Prior distributions.
    distribution_param_names : set of str, optional
        Parameter names referenced in distribution definitions (e.g., shock standard deviations).
    """
    all_equations = equations + steady_state_relationships
    joint_dict = param_dict | calib_dict | deterministic_dict

    check_for_orphan_params(all_equations, joint_dict)

    distribution_atoms = _collect_distribution_atoms(distributions, distribution_param_names, joint_dict)

    check_for_extra_params(
        all_equations,
        joint_dict,
        on_unused_parameters,
        distribution_atoms=distribution_atoms,
    )


def _apply_simplifications(
    try_reduce_vars: list,
    equations: list[sp.Expr],
    variables: list[TimeAwareSymbol],
    tryreduce_sub_dict: dict | None = None,
    simplify_tryreduce_flag: bool = True,
    simplify_constants_flag: bool = True,
) -> tuple[list[sp.Expr], list[TimeAwareSymbol], list | None, list | None]:
    """Apply tryreduce and constant-folding simplifications to the equation system."""
    eliminated_variables = None
    singletons = None

    if simplify_tryreduce_flag:
        equations, variables, eliminated_variables = simplify_tryreduce(
            try_reduce_vars, equations, variables, tryreduce_sub_dict
        )

    if simplify_constants_flag:
        equations, variables, singletons = simplify_constants(equations, variables)

    return equations, variables, eliminated_variables, singletons


def _resolve_deterministic_params(
    deterministic_dict: SymbolDictionary,
    equations: list[sp.Expr],
    steady_state_relationships: list[sp.Expr],
) -> tuple[SymbolDictionary, list[sp.Symbol]]:
    """
    Fully substitute deterministic parameters into each other, then remove any that no longer appear in the system.

    Parameters
    ----------
    deterministic_dict : SymbolDictionary
        Deterministic parameter definitions (mutated in-place to sympy form).
    equations : list of sp.Expr
        Model equations.
    steady_state_relationships : list of sp.Expr
        Steady-state equations.

    Returns
    -------
    deterministic_dict : SymbolDictionary
        Reduced dictionary with fully substituted expressions, converted back to string keys.
    reduced_params : list of sp.Symbol
        Parameters that were eliminated because they no longer appear in any equation.
    """
    deterministic_dict.to_sympy(inplace=True)
    for param, expr in deterministic_dict.items():
        deterministic_dict[param] = substitute_repeatedly(expr, deterministic_dict)

    all_equations = equations + steady_state_relationships
    reduced_params = []
    final = deterministic_dict.copy()

    for param in deterministic_dict:
        if not any(eq.has(param) for eq in all_equations):
            reduced_params.append(param)
            del final[param]

    return final.to_string(), reduced_params


def _split_distributions(
    distributions: SymbolDictionary,
    shock_distributions: SymbolDictionary,
    shock_names: set[str],
) -> tuple[SymbolDictionary, SymbolDictionary, set[str]]:
    """
    Partition distributions into parameter priors and shock priors.

    Returns
    -------
    param_priors : SymbolDictionary
    shock_priors : SymbolDictionary
    shock_hyper_param_names : set of str
        Names of hyper-parameters belonging to shock distributions (excluded from param_priors).
    """
    param_priors = SymbolDictionary()
    shock_priors = SymbolDictionary()
    shock_hyper_param_names: set[str] = set()

    for name, dist in shock_distributions.items():
        shock_priors[name] = dist
        if hasattr(dist, "param_name_to_hyper_name"):
            shock_hyper_param_names.update(dist.param_name_to_hyper_name.values())

    for name, dist in distributions.items():
        if name not in shock_names and name not in shock_hyper_param_names:
            param_priors[name] = dist

    return param_priors, shock_priors, shock_hyper_param_names


def _compile_gcn(
    gcn_path: Path,
    simplify_blocks: bool = True,
    simplify_tryreduce: bool = True,
    simplify_constants: bool = True,
    infer_steady_state: bool = True,
    verbose: bool = True,
    on_unused_parameters: str = "raise",
) -> tuple[tuple, tuple, tuple, dict]:
    """
    Parse and validate a GCN file, returning raw sympy primitives.

    No compilation is performed here. Graph building and compilation are deferred to the caller (``Model`` compiles
    lazily on first use; ``statespace_from_gcn`` builds pytensor graphs directly).

    Parameters
    ----------
    gcn_path : Path
        Path to the GCN file.
    simplify_blocks : bool
        Simplify block equations during parsing.
    simplify_tryreduce : bool
        Eliminate user-marked tryreduce variables.
    simplify_constants : bool
        Fold constant "variables" into equations.
    infer_steady_state : bool
        Propagate analytical steady-state solutions through identities.
    verbose : bool
        Print a build report on completion.
    on_unused_parameters : str
        How to handle unused parameters: ``'raise'``, ``'warn'``, or ``'ignore'``.

    Returns
    -------
    objects : tuple
        ``(variables, shocks, equations, steady_state_relationships, steady_state_equations, ss_solution_dict)``
    dictionaries : tuple
        ``(param_dict, hyper_param_dict, deterministic_dict, calib_dict)``
    priors : tuple
        ``(param_priors, shock_priors)``
    options : dict
        Model options parsed from the GCN file.
    """
    primitives = load_gcn_file(gcn_path, simplify_blocks=simplify_blocks)

    equations = primitives.equations
    variables = primitives.variables
    shocks = primitives.shocks
    param_dict = primitives.param_dict
    calib_dict = primitives.calib_dict
    deterministic_dict = primitives.deterministic_dict
    ss_solution_dict = primitives.ss_solution_dict
    options = primitives.options
    try_reduce = primitives.tryreduce
    block_dict = primitives.block_dict
    distributions = primitives.distributions
    shock_distributions = primitives.shock_distributions
    distribution_param_names = primitives.distribution_param_names

    tryreduce_sub_dict = _block_dict_to_sub_dict(block_dict)

    equations, variables, reduced_vars, singletons = _apply_simplifications(
        try_reduce,
        equations,
        variables,
        tryreduce_sub_dict,
        simplify_tryreduce_flag=simplify_tryreduce,
        simplify_constants_flag=simplify_constants,
    )

    shock_names = {s.base_name for s in shocks}
    param_priors, shock_priors, _ = _split_distributions(distributions, shock_distributions, shock_names)
    param_dict, hyper_param_dict = split_out_hyper_params(param_dict, shock_priors)

    ss_solution_dict = simplify_provided_ss_equations(ss_solution_dict, variables)
    steady_state_relationships = [sp.Eq(var, eq) for var, eq in ss_solution_dict.to_sympy().items()]

    deterministic_dict, reduced_params = _resolve_deterministic_params(
        deterministic_dict,
        equations,
        steady_state_relationships,
    )

    validate_results(
        equations,
        steady_state_relationships,
        param_dict,
        calib_dict,
        deterministic_dict,
        on_unused_parameters=on_unused_parameters,
        distributions=distributions,
        distribution_param_names=distribution_param_names,
    )

    steady_state_equations = system_to_steady_state(equations, shocks)

    user_provided_ss_vars = list(ss_solution_dict.to_sympy().keys()) if ss_solution_dict else []

    if infer_steady_state:
        ss_solution_dict = propagate_steady_state_through_identities(
            ss_solution_dict, steady_state_equations, variables
        )

    all_ss_vars = list(ss_solution_dict.to_sympy().keys()) if ss_solution_dict else []
    inferred_ss_vars = [v for v in all_ss_vars if v not in user_provided_ss_vars]

    steady_state_relationships = [sp.Eq(var, eq) for var, eq in ss_solution_dict.to_sympy().items()]

    variables = sorted(variables, key=natural_sort_key)
    shocks = sorted(shocks, key=natural_sort_key)

    if verbose:
        build_report(
            equations,
            param_dict,
            calib_dict,
            variables,
            shocks,
            param_priors,
            shock_priors,
            reduced_vars,
            reduced_params,
            singletons,
            user_provided_ss_vars,
            inferred_ss_vars,
        )

    objects = (variables, shocks, equations, steady_state_relationships, steady_state_equations, ss_solution_dict)
    dictionaries = (param_dict, hyper_param_dict, deterministic_dict, calib_dict)
    priors = (param_priors, shock_priors)

    return objects, dictionaries, priors, options


def model_from_gcn(
    gcn_path: str | Path,
    simplify_blocks: bool = True,
    simplify_tryreduce: bool = True,
    simplify_constants: bool = True,
    infer_steady_state: bool = True,
    verbose: bool = True,
    mode: str | None = None,
    error_function: ERROR_FUNCTIONS = "squared",
    on_unused_parameters: str = "raise",
    show_errors: bool = True,
    backend: str | None = None,
) -> Model:
    """
    Build a DSGE model from a GCN file.

    Parameters
    ----------
    gcn_path : str or Path
        Path to the GCN file.
    simplify_blocks : bool, default True
        Simplify block equations during parsing.
    simplify_tryreduce : bool, default True
        Eliminate user-marked tryreduce variables.
    simplify_constants : bool, default True
        Fold constant "variables" into equations.
    infer_steady_state : bool, default True
        Propagate analytical steady-state solutions through identities.
    verbose : bool, default True
        Print a build report on completion.
    mode : str or None
        Pytensor compilation mode. If None, uses the default mode (``FAST_RUN``). Check pytensor docs for
        available modes.
    error_function : str, default ``'squared'``
        Steady-state error function.
    on_unused_parameters : str, default ``'raise'``
        How to handle unused parameters: ``'raise'``, ``'warn'``, or ``'ignore'``.
    show_errors : bool, default True
        Pretty-print parse errors to stderr.
    backend : str, optional
        .. deprecated::
            Use ``mode`` instead. ``backend='numpy'`` maps to ``mode='FAST_COMPILE'``;
            ``backend='pytensor'`` maps to ``mode=None``.

    Returns
    -------
    Model
        A compiled DSGE model.
    """
    if backend is not None:
        if backend not in ("numpy", "pytensor"):
            raise ValueError(
                f"Invalid backend={backend!r}. Allowed values are 'numpy' or 'pytensor'. "
                "Prefer using the `mode` argument directly instead."
            )
        _log.warning(
            "The `backend` argument is deprecated and will be removed in a future release. "
            'Use `mode="FAST_COMPILE"` instead of `backend="numpy"`, '
            'or `mode=None` instead of `backend="pytensor"`.'
        )
        mode = "FAST_COMPILE" if backend == "numpy" else None

    gcn_path = Path(gcn_path)

    try:
        objects, dictionaries, priors, options = _compile_gcn(
            gcn_path,
            simplify_blocks=simplify_blocks,
            simplify_tryreduce=simplify_tryreduce,
            simplify_constants=simplify_constants,
            infer_steady_state=infer_steady_state,
            verbose=verbose,
            on_unused_parameters=on_unused_parameters,
        )
    except (GCNErrorCollection, GCNParseError) as e:
        if show_errors:
            _print_parse_error(e, gcn_path)
        raise

    variables, shocks, equations, ss_relationships, steady_state_equations, ss_solution_dict = objects
    param_dict, hyper_param_dict, deterministic_dict, calib_dict = dictionaries

    return Model(
        variables=variables,
        shocks=shocks,
        equations=equations,
        steady_state_relationships=ss_relationships,
        steady_state_equations=steady_state_equations,
        ss_solution_dict=ss_solution_dict,
        param_dict=param_dict,
        hyper_param_dict=hyper_param_dict,
        deterministic_dict=deterministic_dict,
        calib_dict=calib_dict,
        priors=priors,
        is_linear=options.get("linear", False),
        mode=mode,
        error_func=error_function,
    )


def statespace_from_gcn(
    gcn_path: str | Path,
    simplify_blocks: bool = True,
    simplify_tryreduce: bool = True,
    simplify_constants: bool = True,
    infer_steady_state: bool = True,
    verbose: bool = True,
    error_function: ERROR_FUNCTIONS = "squared",
    on_unused_parameters: str = "raise",
    log_linearize: bool = True,
    not_loglin_variables: list[str] | None = None,
    show_errors: bool = True,
) -> DSGEStateSpace:
    """
    Build a symbolic DSGE state-space model from a GCN file.

    Unlike ``model_from_gcn``, this returns a ``DSGEStateSpace`` whose steady-state and linearized system are
    represented as pytensor graphs parameterized by the model's free parameters. This is the entry point for
    Bayesian estimation via PyMC.

    Parameters
    ----------
    gcn_path : str or Path
        Path to the GCN file.
    simplify_blocks : bool, default True
        Simplify block equations during parsing.
    simplify_tryreduce : bool, default True
        Eliminate user-marked tryreduce variables.
    simplify_constants : bool, default True
        Fold constant "variables" into equations.
    infer_steady_state : bool, default True
        Propagate analytical steady-state solutions through identities.
    verbose : bool, default True
        Print a build report on completion.
    error_function : str, default ``'squared'``
        Steady-state error function.
    on_unused_parameters : str, default ``'raise'``
        How to handle unused parameters: ``'raise'``, ``'warn'``, or ``'ignore'``.
    log_linearize : bool, default True
        Whether to log-linearize the model.
    not_loglin_variables : list of str, optional
        Variable names to exclude from log-linearization.
    show_errors : bool, default True
        Pretty-print parse errors to stderr.

    Returns
    -------
    DSGEStateSpace
        A symbolic state-space model ready for estimation.
    """
    gcn_path = Path(gcn_path)

    try:
        objects, dictionaries, priors, options = _compile_gcn(
            gcn_path,
            simplify_blocks=simplify_blocks,
            simplify_tryreduce=simplify_tryreduce,
            simplify_constants=simplify_constants,
            infer_steady_state=infer_steady_state,
            verbose=verbose,
            on_unused_parameters=on_unused_parameters,
        )
    except (GCNErrorCollection, GCNParseError) as e:
        if show_errors:
            _print_parse_error(e, gcn_path)
        raise

    variables, shocks, equations, _ss_relationships, steady_state_equations, ss_solution_dict = objects
    param_dict, hyper_param_dict, deterministic_dict, calib_dict = dictionaries
    param_priors, shock_priors = priors

    if calib_dict:
        raise NotImplementedError("Calibration not yet implemented in StateSpace model")

    # Build symbolic pytensor graphs for the steady-state and linearization systems
    cache = {}
    parameter_mapping, cache = compile_param_dict_func(
        param_dict,
        deterministic_dict,
        cache=cache,
        return_symbolic=True,
    )

    all_params = list(param_dict.to_sympy().keys()) + list(deterministic_dict.to_sympy().keys())
    steady_state_mapping, cache = compile_known_ss(
        ss_solution_dict,
        variables,
        all_params,
        cache=cache,
        return_symbolic=True,
    )

    if steady_state_mapping is None or len(steady_state_mapping) != len(variables):
        raise NotImplementedError("Numeric steady state not yet implemented in StateSpace model")

    equations_pt, cache = _ss_residual_to_pytensor(
        steady_state_equations,
        SymbolDictionary(),
        variables,
        param_dict,
        deterministic_dict,
        calib_dict,
        cache=cache,
    )

    ss_variables = [x.to_ss() for x in variables]
    ss_nodes = []
    for v in ss_variables:
        ck = make_cache_key(v.name, type(v))
        if ck in cache:
            ss_nodes.append(cache[ck])

    ss_resid, ss_jac = build_root_graphs(equations_pt, ss_nodes, use_jac=True)
    ss_error, ss_grad, ss_hess, _, _ = build_minimize_graphs(
        equations_pt,
        ss_nodes,
        error_func=error_function,
        use_jac=True,
        use_hess=True,
        use_hessp=False,
    )

    if not_loglin_variables is None:
        not_loglin_variables = []

    var_names = [get_name(x, base_name=True) for x in variables]
    unknown = set(not_loglin_variables) - set(var_names)
    if unknown:
        raise ValueError(
            f"The following variables were requested not to be log-linearized, but are unknown to the model: "
            f"{', '.join(unknown)}"
        )

    if options.get("linear", False):
        log_linearize = False

    loglin_vars = [v for v in variables if v.base_name not in not_loglin_variables] if log_linearize else []

    [A, B, C, D], _ss_inputs = _linearize_model(
        variables=variables,
        equations=equations,
        shocks=shocks,
        cache=cache,
        loglin_variables=loglin_vars,
    )

    # Wire steady-state expressions through the parameter mapping so everything is a function of free parameters
    steady_state_mapping = {
        k: graph_replace(v, parameter_mapping, strict=False) for k, v in steady_state_mapping.items()
    }
    replacements = parameter_mapping | steady_state_mapping

    ss_resid, ss_jac, ss_error, ss_grad, ss_hess = graph_replace(
        [ss_resid, ss_jac, ss_error, ss_grad, ss_hess],
        replacements,
        strict=False,
    )
    A, B, C, D = rewrite_pregrad(graph_replace([A, B, C, D], replacements, strict=False))

    return DSGEStateSpace(
        variables=variables,
        shocks=shocks,
        equations=equations,
        param_dict=param_dict,
        param_priors=param_priors,
        hyper_param_dict=hyper_param_dict,
        shock_priors=shock_priors,
        parameter_mapping=parameter_mapping,
        steady_state_mapping=steady_state_mapping,
        ss_jac=ss_jac,
        ss_resid=ss_resid,
        ss_error=ss_error,
        ss_error_grad=ss_grad,
        ss_error_hess=ss_hess,
        linearized_system=[A, B, C, D],
        verbose=verbose,
    )


def _format_ss_var_list(label: str, variables: list, line_width: int = 80) -> str:
    """Format a labeled list of variable names with wrapping aligned to after the colon."""
    prefix = f"\t\t{label}: "
    indent = " " * len(prefix.expandtabs())
    names = [str(v) for v in variables]

    lines = []
    current_line = prefix
    for i, name in enumerate(names):
        sep = ", " if i < len(names) - 1 else ""
        candidate = name + sep

        if i == 0:
            current_line += candidate
        elif len(current_line.expandtabs()) + len(candidate) > line_width:
            lines.append(current_line + "\n")
            current_line = indent + candidate
        else:
            current_line += candidate

    lines.append(current_line + "\n")
    return "".join(lines)


def _pluralize(word: str, count: int) -> str:
    match word:
        case "has":
            return word if count == 1 else "have"
        case _:
            return word if count == 1 else word + "s"


def build_report(
    equations: list,
    param_dict: SymbolDictionary,
    calib_dict: SymbolDictionary,
    variables: list,
    shocks: list,
    param_priors: SymbolDictionary,
    shock_priors: SymbolDictionary,
    reduced_vars: list | None,
    reduced_params: list | None,
    singletons: list | None,
    user_provided_ss_vars: list | None = None,
    inferred_ss_vars: list | None = None,
) -> None:
    """
    Log a diagnostic summary after model compilation.

    Reports the number of equations, variables, shocks, parameters, and priors. Warns if the system is not square.

    Parameters
    ----------
    equations : list
        Model equations.
    param_dict : SymbolDictionary
        Free parameters.
    calib_dict : SymbolDictionary
        Calibrating equations.
    variables : list
        Model variables.
    shocks : list
        Exogenous shocks.
    param_priors : SymbolDictionary
        Parameter priors.
    shock_priors : SymbolDictionary
        Shock priors.
    reduced_vars : list or None
        Variables eliminated by tryreduce.
    reduced_params : list or None
        Parameters eliminated by deterministic substitution.
    singletons : list or None
        Constant "variables" folded into equations.
    user_provided_ss_vars : list or None
        Steady-state variables provided by the user.
    inferred_ss_vars : list or None
        Steady-state variables inferred from identities.
    """
    user_provided_ss_vars = user_provided_ss_vars or []
    inferred_ss_vars = inferred_ss_vars or []

    n_eq = len(equations)
    n_var = len(variables)
    n_shock = len(shocks)
    n_calib = len(calib_dict)
    n_free = len(param_dict)
    n_params = n_free + n_calib
    n_param_priors = len(param_priors)
    n_shock_priors = len(shock_priors)

    if singletons and len(singletons) == 0:
        singletons = None

    report = "Model Building Complete.\nFound:\n"
    report += f"\t{n_eq} {_pluralize('equation', n_eq)}\n"
    report += f"\t{n_var} {_pluralize('variable', n_var)}\n"

    if reduced_vars:
        report += "\t\tThe following variables were eliminated at user request:\n"
        report += "\t\t\t" + ", ".join([x.name for x in reduced_vars]) + "\n"

    if singletons:
        report += '\t\tThe following "variables" were defined as constants and have been substituted away:\n'
        report += "\t\t\t" + ", ".join([x.name for x in singletons]) + "\n"

    report += f"\t{n_shock} stochastic {_pluralize('shock', n_shock)}\n"
    report += f"\t\t {n_shock_priors} / {n_shock} {_pluralize('has', n_shock_priors)} a defined prior.\n"

    report += f"\t{n_params} {_pluralize('parameter', n_params)}\n"
    if reduced_params:
        report += "\t\tThe following parameters were eliminated via substitution into other parameters:\n"
        report += "\t\t\t" + ", ".join([x.name for x in reduced_params]) + "\n"

    report += f"\t\t {n_param_priors} / {n_params} {_pluralize('has', n_param_priors)} a defined prior.\n"
    report += f"\t{n_calib} {_pluralize('parameter', n_calib)} to calibrate.\n"

    n_user_provided = len(user_provided_ss_vars)
    n_inferred = len(inferred_ss_vars)
    n_total_ss = n_user_provided + n_inferred

    if n_total_ss > 0:
        report += f"\t{n_total_ss} / {n_var} variables have analytical steady-state values.\n"
        if n_user_provided > 0:
            report += _format_ss_var_list(f"{n_user_provided} user-provided", user_provided_ss_vars)
        if n_inferred > 0:
            report += _format_ss_var_list(f"{n_inferred} inferred", inferred_ss_vars)

    if n_eq == n_var:
        report += "Model appears well defined and ready to proceed to solving.\n"
    else:
        message = (
            f"The model does not appear correctly specified, there are {n_eq} {_pluralize('equation', n_eq)} but "
            f"{n_var} {_pluralize('variable', n_var)}. It will not be possible to solve this model. Please check "
            f"the specification using available diagnostic tools, and check the GCN file for typos."
        )
        warn(message, stacklevel=2)

    _log.info(report)
