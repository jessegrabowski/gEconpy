import logging
import sys

from pathlib import Path
from warnings import warn

import pytensor.tensor as pt
import sympy as sp

from pymc.pytensorf import rewrite_pregrad
from pytensor import graph_replace

from gEconpy.classes.containers import SymbolDictionary
from gEconpy.classes.distributions import CompositeDistribution
from gEconpy.classes.time_aware_symbol import TimeAwareSymbol
from gEconpy.exceptions import ExtraParameterError, ExtraParameterWarning, OrphanParameterError
from gEconpy.model.compile import BACKENDS
from gEconpy.model.model import Model
from gEconpy.model.perturbation import compile_linearized_system
from gEconpy.model.simplification import simplify_constants, simplify_tryreduce
from gEconpy.model.statespace import DSGEStateSpace
from gEconpy.model.steady_state import (
    ERROR_FUNCTIONS,
    compile_model_ss_functions,
    propagate_steady_state_through_identities,
    simplify_provided_ss_equations,
    system_to_steady_state,
)
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
) -> SymbolDictionary[str, float]:
    """
    Remove shock hyper parameters from the parameter dictionary.

    Parameters
    ----------
    param_dict: SymbolDictionary
        Dictionary of initial parameter values

    shock_prior: SymbolDictionary
        Dictionary of shock priors

    Returns
    -------
    param_dict: SymbolDictionary
        Dictionary of initial parameter values with shock hyper parameters removed.
    """
    new_param_dict = param_dict.copy()
    hyper_param_dict = SymbolDictionary()

    all_hyper_params = [param for dist in shock_prior.values() for param in dist.param_name_to_hyper_name.values()]

    for param in all_hyper_params:
        if param in new_param_dict:
            del new_param_dict[param]
            hyper_param_dict[param] = param_dict[param]

    return new_param_dict, hyper_param_dict


def _block_dict_to_sub_dict(block_dict):
    """Extract substitution dictionary from block equations for tryreduce."""
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


def _apply_simplifications(
    try_reduce_vars,
    equations,
    variables,
    tryreduce_sub_dict=None,
    do_simplify_tryreduce_flag=True,
    do_simplify_constants_flag=True,
):
    """Apply tryreduce and constant simplifications to equations."""
    eliminated_variables = None
    singletons = None

    if do_simplify_tryreduce_flag:
        equations, variables, eliminated_variables = simplify_tryreduce(
            try_reduce_vars, equations, variables, tryreduce_sub_dict
        )

    if do_simplify_constants_flag:
        equations, variables, singletons = simplify_constants(equations, variables)

    return equations, variables, eliminated_variables, singletons


def _compile_gcn(
    gcn_path: Path,
    simplify_blocks: bool = True,
    simplify_tryreduce: bool = True,
    simplify_constants: bool = True,
    infer_steady_state: bool = True,
    verbose: bool = True,
    backend: BACKENDS = "numpy",
    return_symbolic: bool = False,
    error_function: ERROR_FUNCTIONS = "squared",
    on_unused_parameters="raise",
    **kwargs,
) -> tuple[tuple, tuple, tuple, dict, tuple, dict]:
    # Load using new parser
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

    # Build sub_dict for tryreduce from block equations
    tryreduce_sub_dict = _block_dict_to_sub_dict(block_dict)

    # Apply simplifications
    equations, variables, reduced_vars, singletons = _apply_simplifications(
        try_reduce,
        equations,
        variables,
        tryreduce_sub_dict,
        do_simplify_tryreduce_flag=simplify_tryreduce,
        do_simplify_constants_flag=simplify_constants,
    )

    # Split distributions into param_priors and shock_priors
    param_priors = SymbolDictionary()
    shock_priors = SymbolDictionary()
    shock_names = {s.base_name for s in shocks}

    # Shock priors come from shock_distributions (CompositeDistribution objects)
    # Also collect hyper-param names to exclude from param_priors
    shock_hyper_param_names = set()
    for name, dist in shock_distributions.items():
        shock_priors[name] = dist
        if hasattr(dist, "param_name_to_hyper_name"):
            shock_hyper_param_names.update(dist.param_name_to_hyper_name.values())

    # Parameter priors are distributions not associated with shocks or shock hyper-params
    for name, dist in distributions.items():
        if name not in shock_names and name not in shock_hyper_param_names:
            param_priors[name] = dist

    param_dict, hyper_param_dict = split_out_hyper_params(param_dict, shock_priors)

    ss_solution_dict = simplify_provided_ss_equations(ss_solution_dict, variables)
    steady_state_relationships = [sp.Eq(var, eq) for var, eq in ss_solution_dict.to_sympy().items()]

    # TODO: Move this to a separate function
    # TODO: Add option to not eliminate deterministic parameters (the user might be interested in them)

    deterministic_dict.to_sympy(inplace=True)
    for param, expr in deterministic_dict.items():
        deterministic_dict[param] = substitute_repeatedly(expr, deterministic_dict)

    # If a deterministic parameter is only used in other parameters, it will now have been completely substituted away
    # and can be removed
    reduced_params = []
    final_deterministics = deterministic_dict.copy()

    for param in deterministic_dict:
        if not any(eq.has(param) for eq in equations + steady_state_relationships):
            reduced_params.append(param)
            del final_deterministics[param]

    deterministic_dict = final_deterministics.to_string()

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

    variables = sorted(variables, key=lambda x: x.base_name)
    shocks = sorted(shocks, key=lambda x: x.base_name)

    functions, cache = compile_model_ss_functions(
        steady_state_equations,
        ss_solution_dict,
        variables,
        param_dict,
        deterministic_dict,
        calib_dict,
        error_func=error_function,
        backend=backend,
        return_symbolic=return_symbolic,
        **kwargs,
    )

    f_params, f_ss, resid_funcs, error_funcs = functions
    f_ss_resid, f_ss_jac = resid_funcs
    f_ss_error, f_ss_grad, f_ss_hess, f_ss_hessp = error_funcs

    f_linearize, cache = compile_linearized_system(
        equations,
        variables,
        param_dict,
        deterministic_dict,
        calib_dict,
        shocks,
        backend=backend,
        return_symbolic=return_symbolic,
        cache=cache,
    )

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

    objects = (variables, shocks, equations, steady_state_relationships)
    dictionaries = (param_dict, hyper_param_dict, deterministic_dict, calib_dict)
    functions = (
        f_ss,
        f_ss_jac,
        f_params,
        f_ss_resid,
        f_ss_error,
        f_ss_grad,
        f_ss_hess,
        f_ss_hessp,
        f_linearize,
    )
    priors = (param_priors, shock_priors)

    return objects, dictionaries, functions, cache, priors, options


def model_from_gcn(
    gcn_path: str | Path,
    simplify_blocks: bool = True,
    simplify_tryreduce: bool = True,
    simplify_constants: bool = True,
    infer_steady_state: bool = True,
    verbose: bool = True,
    backend: BACKENDS = "numpy",
    error_function: ERROR_FUNCTIONS = "squared",
    on_unused_parameters="raise",
    show_errors: bool = True,
    **kwargs,
) -> Model:
    gcn_path = Path(gcn_path)

    try:
        objects, dictionaries, functions, _cache, priors, options = _compile_gcn(
            gcn_path,
            simplify_blocks=simplify_blocks,
            simplify_tryreduce=simplify_tryreduce,
            simplify_constants=simplify_constants,
            infer_steady_state=infer_steady_state,
            verbose=verbose,
            backend=backend,
            error_function=error_function,
            on_unused_parameters=on_unused_parameters,
            **kwargs,
        )
    except (GCNErrorCollection, GCNParseError) as e:
        if show_errors:
            _print_parse_error(e, gcn_path)
        raise

    variables, shocks, equations, ss_relationships = objects
    param_dict, hyper_param_dict, deterministic_dict, calib_dict = dictionaries

    (
        f_ss,
        f_ss_jac,
        f_params,
        f_ss_resid,
        f_ss_error,
        f_ss_grad,
        f_ss_hess,
        f_ss_hessp,
        f_linearize,
    ) = functions

    return Model(
        variables=variables,
        shocks=shocks,
        equations=equations,
        steady_state_relationships=ss_relationships,
        param_dict=param_dict,
        hyper_param_dict=hyper_param_dict,
        deterministic_dict=deterministic_dict,
        calib_dict=calib_dict,
        f_ss=f_ss,
        f_ss_jac=f_ss_jac,
        f_params=f_params,
        f_ss_resid=f_ss_resid,
        f_ss_error=f_ss_error,
        f_ss_error_grad=f_ss_grad,
        f_ss_error_hess=f_ss_hess,
        f_ss_error_hessp=f_ss_hessp,
        f_linearize=f_linearize,
        backend=backend,
        priors=priors,
        is_linear=options.get("linear", False),
    )


def statespace_from_gcn(
    gcn_path: str,
    simplify_blocks: bool = True,
    simplify_tryreduce: bool = True,
    simplify_constants: bool = True,
    verbose: bool = True,
    error_function: ERROR_FUNCTIONS = "squared",
    on_unused_parameters="raise",
    log_linearize: bool = True,
    not_loglin_variables: list[str] | None = None,
    show_errors: bool = True,
    **kwargs,
):

    gcn_path = Path(gcn_path)

    try:
        objects, dictionaries, functions, cache, priors, options = _compile_gcn(
            gcn_path,
            simplify_blocks=simplify_blocks,
            simplify_tryreduce=simplify_tryreduce,
            simplify_constants=simplify_constants,
            verbose=verbose,
            backend="pytensor",
            error_function=error_function,
            on_unused_parameters=on_unused_parameters,
            return_symbolic=True,
            **kwargs,
        )
    except (GCNErrorCollection, GCNParseError) as e:
        if show_errors:
            _print_parse_error(e, gcn_path)
        raise

    variables, shocks, equations, _ss_relationships = objects
    param_dict, hyper_param_dict, _deterministic_dict, calib_dict = dictionaries
    param_priors, shock_priors = priors

    if len(calib_dict) > 0:
        raise NotImplementedError("Calibration not yet implemented in StateSpace model")

    (
        steady_state_mapping,
        ss_jac,
        parameter_mapping,
        ss_resid,
        ss_error,
        ss_grad,
        ss_hess,
        _ss_hessp,
        linearized_matrices,
    ) = functions

    # Check that the entire steady state has been provided
    if steady_state_mapping is None or len(steady_state_mapping) != len(variables):
        raise NotImplementedError("Numeric steady state not yet implemented in StateSpace model")

    A, B, C, D = linearized_matrices

    not_loglin_flags = next(x for x in cache.values() if x.name == "not_loglin_variable")

    # First replace deterministic variables with functions of input variables in the user-provided steady state
    # expressiong
    steady_state_mapping = {
        k: graph_replace(v, parameter_mapping, strict=False) for k, v in steady_state_mapping.items()
    }

    ss_vec = pt.stack(list(steady_state_mapping.values()))
    if not_loglin_variables is None:
        not_loglin_variables = []

    var_names = [get_name(x, base_name=True) for x in variables]
    unknown_not_login = set(not_loglin_variables) - set(var_names)

    if len(unknown_not_login) > 0:
        raise ValueError(
            f"The following variables were requested not to be log-linearized, but are unknown to the model: "
            f"{', '.join(unknown_not_login)}"
        )

    if options.get("linear", False):
        log_linearize = False

    if log_linearize:
        not_loglin_mask = pt.as_tensor([x in not_loglin_variables for x in var_names])
        not_loglin_values = pt.le(ss_vec, 0.0).astype(float)
        not_loglin_values = not_loglin_values[not_loglin_mask].set(1.0)
    else:
        not_loglin_values = pt.ones(ss_vec.shape[0])

    not_loglin_replacement = {not_loglin_flags: not_loglin_values}

    replacements = parameter_mapping | steady_state_mapping | not_loglin_replacement

    # Replace all placeholders with functions of the input parameters
    ss_resid, ss_jac, ss_error, ss_grad, ss_hess = graph_replace(
        [ss_resid, ss_jac, ss_error, ss_grad, ss_hess], replacements, strict=False
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
    Write a diagnostic message after building the model.

    Note that successfully building the model does not guarantee that the model
    is correctly specified. For example, it is possible to build a model with
    more equations than parameters. This message will warn the user in this case.
    """
    user_provided_ss_vars = user_provided_ss_vars or []
    inferred_ss_vars = inferred_ss_vars or []

    n_equations = len(equations)
    n_variables = len(variables)
    n_shocks = len(shocks)
    n_params_to_calibrate = len(calib_dict)
    n_free_params = len(param_dict)

    if singletons and len(singletons) == 0:
        singletons = None

    eq_str = "equation" if n_equations == 1 else "equations"
    var_str = "variable" if n_variables == 1 else "variables"
    shock_str = "shock" if n_shocks == 1 else "shocks"
    free_par_str = "parameter" if len(param_dict) == 1 else "parameters"
    calib_par_str = "parameter" if n_params_to_calibrate == 1 else "parameters"

    n_params = n_free_params + n_params_to_calibrate

    param_prior_keys = param_priors.keys()
    shock_prior_keys = shock_priors.keys()

    report = "Model Building Complete.\nFound:\n"
    report += f"\t{n_equations} {eq_str}\n"
    report += f"\t{n_variables} {var_str}\n"

    if reduced_vars:
        report += "\t\tThe following variables were eliminated at user request:\n"
        report += "\t\t\t" + ", ".join([x.name for x in reduced_vars]) + "\n"

    if singletons:
        report += '\t\tThe following "variables" were defined as constants and have been substituted away:\n'
        report += "\t\t\t" + ", ".join([x.name for x in singletons]) + "\n"

    report += f"\t{n_shocks} stochastic {shock_str}\n"
    have_has = "have" if len(shock_prior_keys) == 1 else "has"
    report += f"\t\t {len(shock_prior_keys)} / {n_shocks} {have_has} a defined prior.\n"

    report += f"\t{n_params} {free_par_str}\n"
    if reduced_params:
        report += "\t\tThe following parameters were eliminated via substitution into other parameters:\n"
        report += "\t\t\t" + ", ".join([x.name for x in reduced_params]) + "\n"

    report += (
        f"\t\t {len(param_prior_keys)} / {n_params} parameters {'have' if len(param_prior_keys) == 1 else 'has'} "
        f"a defined prior. \n"
    )

    report += f"\t{n_params_to_calibrate} {calib_par_str} to calibrate.\n"

    # Report steady-state information
    n_user_provided = len(user_provided_ss_vars)
    n_inferred = len(inferred_ss_vars)
    n_total_ss = n_user_provided + n_inferred

    if n_total_ss > 0:
        report += f"\t{n_total_ss} / {n_variables} variables have analytical steady-state values.\n"
        if n_user_provided > 0:
            var_names = ", ".join(str(v) for v in user_provided_ss_vars)
            report += f"\t\t{n_user_provided} user-provided: {var_names}\n"
        if n_inferred > 0:
            var_names = ", ".join(str(v) for v in inferred_ss_vars)
            report += f"\t\t{n_inferred} automatically inferred: {var_names}\n"

    if n_equations == n_variables:
        report += "Model appears well defined and ready to proceed to solving.\n"
    else:
        message = (
            f"The model does not appear correctly specified, there are {n_equations} {eq_str} but "
            f"{n_variables} {var_str}. It will not be possible to solve this model. Please check the "
            f"specification using available diagnostic tools, and check the GCN file for typos."
        )
        warn(message, stacklevel=2)

    _log.info(report)
