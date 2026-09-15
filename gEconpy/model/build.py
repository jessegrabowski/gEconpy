import logging
import sys
import warnings

from dataclasses import dataclass
from pathlib import Path

import sympy as sp

from pymc_extras.statespace.utils.constants import JITTER_DEFAULT, MISSING_FILL
from pytensor.graph.replace import graph_replace

from gEconpy.classes.containers import SymbolDictionary
from gEconpy.classes.time_aware_symbol import TimeAwareSymbol
from gEconpy.exceptions import ExtraParameterError, ExtraParameterWarning, OrphanParameterError
from gEconpy.model.model import Model
from gEconpy.model.parameters import compile_param_dict_func
from gEconpy.model.perturbation import linearize_model
from gEconpy.model.simplification import simplify_constants, simplify_tryreduce
from gEconpy.model.statespace import DSGEStateSpace
from gEconpy.model.steady_state import (
    ERROR_FUNCTIONS,
    compile_known_ss,
    propagate_steady_state_through_identities,
    simplify_provided_ss_equations,
    system_to_steady_state,
)
from gEconpy.model.timing import natural_sort_key
from gEconpy.parser.errors import GCNErrorCollection, GCNParseError
from gEconpy.parser.formatting import ErrorFormatter
from gEconpy.parser.loader import load_gcn_file
from gEconpy.pytensorf.compile import rewrite_pregrad
from gEconpy.utilities import flatten_substitution_dict, get_name

_log = logging.getLogger(__name__)


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
    Build a :class:`~gEconpy.model.model.Model` from a GCN file.

    Parse the file, derive first-order conditions, apply the requested simplifications, validate that every
    parameter is both defined and used, and infer as much of the steady state as the identities allow. The returned
    model builds its PyTensor graphs on first use.

    Parameters
    ----------
    gcn_path : str or Path
        Path to the GCN file.
    simplify_blocks : bool, optional
        Simplify block equations during parsing. Default is True.
    simplify_tryreduce : bool, optional
        Eliminate the variables listed in the file's ``tryreduce`` block. Default is True.
    simplify_constants : bool, optional
        Substitute away variables that an equation pins to a constant, such as ``P[] = 1``. Default is True.
    infer_steady_state : bool, optional
        Extend the user-provided steady state by solving single-unknown identities. Default is True.
    verbose : bool, optional
        Log a build report on completion. Default is True.
    mode : str, optional
        PyTensor compilation mode for the functions the model compiles. Default is None, which uses the PyTensor
        default mode.
    error_function : str, optional
        Scalar error metric used when the steady state is found by minimization. One of ``'squared'``,
        ``'mean_squared'``, ``'abs'``, or ``'l2-norm'``. Default is ``'squared'``.
    on_unused_parameters : str, optional
        What to do with parameters that are defined but never used: ``'raise'``, ``'warn'``, or ``'ignore'``.
        Default is ``'raise'``.
    show_errors : bool, optional
        Print formatted parse errors to stderr before re-raising them. Default is True.
    backend : str, optional
        .. deprecated::
            Use ``mode`` instead. ``backend='numpy'`` maps to ``mode='FAST_COMPILE'`` and ``backend='pytensor'``
            maps to ``mode=None``.

    Returns
    -------
    model : Model
        The parsed model, ready to solve for its steady state and perturbation solution.

    Examples
    --------
    Build the packaged RBC model and solve for its steady state:

    .. code-block:: python

        import gEconpy as ge
        from gEconpy.data import get_example_gcn

        model = ge.model_from_gcn(get_example_gcn("RBC"), verbose=False)
        steady_state = model.steady_state(verbose=False, progressbar=False)
        ge.print_steady_state(steady_state)
    """
    if backend is not None:
        mode = _mode_from_deprecated_backend(backend)

    gcn_path = Path(gcn_path)
    primitives = _load_model_primitives(
        gcn_path,
        simplify_blocks=simplify_blocks,
        simplify_tryreduce=simplify_tryreduce,
        simplify_constants=simplify_constants,
        infer_steady_state=infer_steady_state,
        verbose=verbose,
        on_unused_parameters=on_unused_parameters,
        show_errors=show_errors,
    )

    return Model(
        variables=primitives.variables,
        shocks=primitives.shocks,
        equations=primitives.equations,
        steady_state_relationships=primitives.steady_state_relationships,
        steady_state_equations=primitives.steady_state_equations,
        ss_solution_dict=primitives.ss_solution_dict,
        param_dict=primitives.param_dict,
        hyper_param_dict=primitives.hyper_param_dict,
        deterministic_dict=primitives.deterministic_dict,
        calib_dict=primitives.calib_dict,
        priors=(primitives.param_priors, primitives.shock_priors),
        is_linear=primitives.options.get("linear", False),
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
    on_unused_parameters: str = "raise",
    log_linearize: bool = True,
    not_loglin_variables: list[str] | None = None,
    show_errors: bool = True,
    filter_type: str = "standard",
    mode: str | None = None,
    cov_jitter: float = JITTER_DEFAULT,
    missing_fill_value: float = MISSING_FILL,
) -> DSGEStateSpace:
    """
    Build a :class:`~gEconpy.model.statespace.DSGEStateSpace` from a GCN file.

    The returned object holds the steady state and the linearized system as PyTensor graphs of the model's free
    parameters, which makes it the entry point for Bayesian estimation with PyMC. The GCN file must provide, or
    allow inference of, an analytic steady state for every variable, and it cannot contain a calibration block.

    Parameters
    ----------
    gcn_path : str or Path
        Path to the GCN file.
    simplify_blocks : bool, optional
        Simplify block equations during parsing. Default is True.
    simplify_tryreduce : bool, optional
        Eliminate the variables listed in the file's ``tryreduce`` block. Default is True.
    simplify_constants : bool, optional
        Substitute away variables that an equation pins to a constant, such as ``P[] = 1``. Default is True.
    infer_steady_state : bool, optional
        Extend the user-provided steady state by solving single-unknown identities. Default is True.
    verbose : bool, optional
        Log a build report on completion. Default is True.
    on_unused_parameters : str, optional
        What to do with parameters that are defined but never used: ``'raise'``, ``'warn'``, or ``'ignore'``.
        Default is ``'raise'``.
    log_linearize : bool, optional
        Linearize in logs of the variables. Ignored when the GCN file declares the model linear. Default is True.
    not_loglin_variables : list of str, optional
        Variable names to linearize in levels even when ``log_linearize`` is True. Default is None, which
        log-linearizes every variable.
    show_errors : bool, optional
        Print formatted parse errors to stderr before re-raising them. Default is True.
    filter_type : str, optional
        Kalman filter variant used by the underlying ``PyMCStateSpace``. Default is ``'standard'``.
    mode : str, optional
        PyTensor compilation mode for post-estimation sampling functions. Default is None, which uses the PyTensor
        default mode.
    cov_jitter : float, optional
        Jitter added to the diagonal of covariance matrices inside the Kalman filter. Default is ``JITTER_DEFAULT``
        from pymc-extras.
    missing_fill_value : float, optional
        Sentinel that replaces missing observations before the filter runs. Default is ``MISSING_FILL`` from
        pymc-extras.

    Returns
    -------
    statespace : DSGEStateSpace
        A symbolic state-space model ready for estimation.

    Examples
    --------
    Build the packaged RBC model as a state-space model, the form needed to estimate it with PyMC:

    .. code-block:: python

        import gEconpy as ge
        from gEconpy.data import get_example_gcn

        statespace = ge.statespace_from_gcn(get_example_gcn("RBC"), verbose=False)
    """
    gcn_path = Path(gcn_path)
    primitives = _load_model_primitives(
        gcn_path,
        simplify_blocks=simplify_blocks,
        simplify_tryreduce=simplify_tryreduce,
        simplify_constants=simplify_constants,
        infer_steady_state=infer_steady_state,
        verbose=verbose,
        on_unused_parameters=on_unused_parameters,
        show_errors=show_errors,
    )

    if primitives.calib_dict:
        raise NotImplementedError(
            "DSGEStateSpace does not support calibrated parameters. Remove the calibrating equations from the GCN "
            "file and give each calibrated parameter a fixed value or a prior instead."
        )

    variables = primitives.variables
    cache: dict = {}
    parameter_mapping, cache = compile_param_dict_func(
        primitives.param_dict,
        primitives.deterministic_dict,
        cache=cache,
        return_symbolic=True,
    )

    all_params = list(primitives.param_dict.to_sympy().keys()) + list(primitives.deterministic_dict.to_sympy().keys())
    steady_state_mapping, cache = compile_known_ss(
        primitives.ss_solution_dict,
        variables,
        all_params,
        cache=cache,
        return_symbolic=True,
    )

    if steady_state_mapping is None or len(steady_state_mapping) != len(variables):
        raise NotImplementedError(
            "DSGEStateSpace requires an analytic steady state for every variable. Add the missing steady-state "
            "expressions to the STEADY_STATE block of the GCN file, or use model_from_gcn to solve the steady state "
            "numerically."
        )

    loglin_variables = _select_loglin_variables(
        variables,
        log_linearize=log_linearize and not primitives.options.get("linear", False),
        not_loglin_variables=not_loglin_variables,
    )

    # The equation permutation is dropped because the downstream solvers index T and R by variable, so only the
    # variable permutation has to be carried on to the state-space model.
    [A, B, C, D], _ss_inputs, _eq_order, var_order = linearize_model(
        variables=variables,
        equations=primitives.equations,
        shocks=primitives.shocks,
        cache=cache,
        loglin_variables=loglin_variables,
    )

    # Rewrite the steady state through the parameter mapping first, so that both the steady state and the linearized
    # system end up as functions of the free parameters alone.
    steady_state_mapping = {
        variable: graph_replace(expression, parameter_mapping, strict=False)
        for variable, expression in steady_state_mapping.items()
    }
    replacements = parameter_mapping | steady_state_mapping
    A, B, C, D = rewrite_pregrad(graph_replace([A, B, C, D], replacements, strict=False))

    return DSGEStateSpace(
        variables=variables,
        shocks=primitives.shocks,
        equations=primitives.equations,
        param_dict=primitives.param_dict,
        param_priors=primitives.param_priors,
        hyper_param_dict=primitives.hyper_param_dict,
        shock_priors=primitives.shock_priors,
        parameter_mapping=parameter_mapping,
        steady_state_mapping=steady_state_mapping,
        linearized_system=[A, B, C, D],
        var_order=var_order,
        log_linearized_variables=[variable.base_name for variable in loglin_variables],
        sympytensor_cache=cache,
        filter_type=filter_type,
        mode=mode,
        cov_jitter=cov_jitter,
        missing_fill_value=missing_fill_value,
        verbose=verbose,
    )


def build_report(
    equations: list[sp.Expr],
    param_dict: SymbolDictionary,
    calib_dict: SymbolDictionary,
    variables: list[TimeAwareSymbol],
    shocks: list[TimeAwareSymbol],
    param_priors: SymbolDictionary,
    shock_priors: SymbolDictionary,
    reduced_vars: list[TimeAwareSymbol] | None,
    reduced_params: list[sp.Symbol] | None,
    singletons: list[TimeAwareSymbol] | None,
    user_provided_ss_vars: list[TimeAwareSymbol] | None = None,
    inferred_ss_vars: list[TimeAwareSymbol] | None = None,
) -> None:
    """
    Log a summary of the parsed model and warn when the system is not square.

    Parameters
    ----------
    equations : list of sympy expression
        Model equations.
    param_dict : SymbolDictionary
        Free parameters.
    calib_dict : SymbolDictionary
        Calibrating equations.
    variables : list of TimeAwareSymbol
        Model variables.
    shocks : list of TimeAwareSymbol
        Exogenous shocks.
    param_priors : SymbolDictionary
        Parameter priors.
    shock_priors : SymbolDictionary
        Shock priors.
    reduced_vars : list of TimeAwareSymbol or None
        Variables eliminated by the ``tryreduce`` block.
    reduced_params : list of sympy Symbol or None
        Parameters eliminated by substitution into other parameters.
    singletons : list of TimeAwareSymbol or None
        Variables pinned to a constant and substituted away.
    user_provided_ss_vars : list of TimeAwareSymbol, optional
        Variables whose steady state the GCN file provides. Default is None, meaning none.
    inferred_ss_vars : list of TimeAwareSymbol, optional
        Variables whose steady state was inferred from identities. Default is None, meaning none.
    """
    user_provided_ss_vars = user_provided_ss_vars or []
    inferred_ss_vars = inferred_ss_vars or []

    n_eq = len(equations)
    n_var = len(variables)
    n_shock = len(shocks)
    n_calib = len(calib_dict)
    n_params = len(param_dict) + n_calib
    n_param_priors = len(param_priors)
    n_shock_priors = len(shock_priors)

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
        warnings.warn(message, stacklevel=2)

    _log.info(report)


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
    Check that every parameter in the model is both defined and used.

    A parameter that appears in an equation without a definition raises
    :class:`~gEconpy.exceptions.OrphanParameterError`. A parameter that is defined but appears in no equation and no
    prior distribution is handled according to ``on_unused_parameters``.

    Parameters
    ----------
    equations : list of sympy expression
        Model equations.
    steady_state_relationships : list of sympy expression
        Steady-state equations.
    param_dict : SymbolDictionary
        Free parameters.
    calib_dict : SymbolDictionary
        Calibrating equations.
    deterministic_dict : SymbolDictionary
        Deterministic parameter definitions.
    on_unused_parameters : str, optional
        What to do with unused parameters: ``'raise'``, ``'warn'``, or ``'ignore'``. Default is ``'raise'``.
    distributions : SymbolDictionary, optional
        Prior distributions. Parameters that appear in a distribution's arguments count as used. Default is None.
    distribution_param_names : set of str, optional
        Names of parameters that prior distributions reference, such as shock standard deviations. Default is None.
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


def check_for_orphan_params(equations: list[sp.Expr], param_dict: SymbolDictionary) -> None:
    """
    Raise :class:`~gEconpy.exceptions.OrphanParameterError` if an equation uses a parameter that is not defined.

    Parameters
    ----------
    equations : list of sympy expression
        Model equations.
    param_dict : SymbolDictionary
        Defined parameters.
    """
    parameters = list(param_dict.to_sympy().keys())
    param_equations = [value for value in param_dict.values() if isinstance(value, sp.Expr)]

    orphans = [
        atom
        for eq in equations
        for atom in eq.atoms()
        if (
            isinstance(atom, sp.Symbol)
            and not isinstance(atom, TimeAwareSymbol)
            and atom not in parameters
            and not any(param_eq.has(atom) for param_eq in param_equations)
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
    Raise or warn when a defined parameter appears in no equation and no prior distribution.

    Parameters
    ----------
    equations : list of sympy expression
        Model equations.
    param_dict : SymbolDictionary
        Defined parameters.
    on_unused_parameters : str, optional
        What to do with unused parameters: ``'raise'``, ``'warn'``, or ``'ignore'``. Default is ``'raise'``.
    distribution_atoms : set of sympy Symbol, optional
        Symbols referenced by prior distributions. These count as used. Default is None.
    """
    parameters = list(param_dict.to_sympy().keys())
    param_equations = [value for value in param_dict.values() if isinstance(value, sp.Expr)]

    used_atoms = {atom for eq in equations + param_equations for atom in eq.atoms()}
    if distribution_atoms:
        used_atoms |= distribution_atoms

    extras = [param for param in parameters if param not in used_atoms]
    if not extras:
        return

    if on_unused_parameters == "raise":
        raise ExtraParameterError(extras)
    if on_unused_parameters == "warn":
        warnings.warn(ExtraParameterWarning(extras), stacklevel=2)


def split_out_hyper_params(
    param_dict: SymbolDictionary,
    shock_prior: SymbolDictionary,
) -> tuple[SymbolDictionary, SymbolDictionary]:
    """
    Move the hyper-parameters of the shock priors out of the parameter dictionary.

    A hyper-parameter is a parameter that defines a shock prior, such as the standard deviation of an AR(1) shock.

    Parameters
    ----------
    param_dict : SymbolDictionary
        Initial parameter values.
    shock_prior : SymbolDictionary
        Shock priors, keyed by shock name.

    Returns
    -------
    param_dict : SymbolDictionary
        Parameter values with the hyper-parameters removed.
    hyper_param_dict : SymbolDictionary
        The hyper-parameters and their values.
    """
    free_params = param_dict.copy()
    hyper_param_dict = SymbolDictionary()

    hyper_param_names = [name for dist in shock_prior.values() for name in dist.param_name_to_hyper_name.values()]
    for name in hyper_param_names:
        if name in free_params:
            del free_params[name]
            hyper_param_dict[name] = param_dict[name]

    return free_params, hyper_param_dict


@dataclass(frozen=True)
class _ModelPrimitives:
    variables: list[TimeAwareSymbol]
    shocks: list[TimeAwareSymbol]
    equations: list[sp.Expr]
    steady_state_relationships: list[sp.Eq]
    steady_state_equations: list[sp.Expr]
    ss_solution_dict: SymbolDictionary
    param_dict: SymbolDictionary
    hyper_param_dict: SymbolDictionary
    deterministic_dict: SymbolDictionary
    calib_dict: SymbolDictionary
    param_priors: SymbolDictionary
    shock_priors: SymbolDictionary
    options: dict


def _load_model_primitives(
    gcn_path: Path,
    simplify_blocks: bool,
    simplify_tryreduce: bool,
    simplify_constants: bool,
    infer_steady_state: bool,
    verbose: bool,
    on_unused_parameters: str,
    show_errors: bool,
) -> _ModelPrimitives:
    try:
        return _derive_model_primitives(
            gcn_path,
            simplify_blocks=simplify_blocks,
            simplify_tryreduce=simplify_tryreduce,
            simplify_constants=simplify_constants,
            infer_steady_state=infer_steady_state,
            verbose=verbose,
            on_unused_parameters=on_unused_parameters,
        )
    except (GCNErrorCollection, GCNParseError) as error:
        if show_errors:
            _print_parse_error(error, gcn_path)
        raise


def _derive_model_primitives(
    gcn_path: Path,
    simplify_blocks: bool,
    simplify_tryreduce: bool,
    simplify_constants: bool,
    infer_steady_state: bool,
    verbose: bool,
    on_unused_parameters: str,
) -> _ModelPrimitives:
    """
    Parse a GCN file into sympy primitives, simplify and validate them, and infer the steady state.

    No PyTensor graph is built here. :class:`~gEconpy.model.model.Model` compiles lazily on first use and
    :func:`statespace_from_gcn` builds its graphs from the returned primitives.
    """
    parsed = load_gcn_file(gcn_path, simplify_blocks=simplify_blocks)

    equations, variables, reduced_vars, singletons = _apply_simplifications(
        parsed.tryreduce,
        parsed.equations,
        parsed.variables,
        _block_dict_to_sub_dict(parsed.block_dict),
        use_tryreduce=simplify_tryreduce,
        use_constants=simplify_constants,
    )

    shock_names = {shock.base_name for shock in parsed.shocks}
    param_priors, shock_priors = _split_distributions(parsed.distributions, parsed.shock_distributions, shock_names)
    param_dict, hyper_param_dict = split_out_hyper_params(parsed.param_dict, shock_priors)

    ss_solution_dict = simplify_provided_ss_equations(parsed.ss_solution_dict, variables)
    user_provided_relationships = [sp.Eq(var, eq) for var, eq in ss_solution_dict.to_sympy().items()]

    deterministic_dict, reduced_params = _resolve_deterministic_params(
        parsed.deterministic_dict,
        equations,
        user_provided_relationships,
    )

    validate_results(
        equations,
        user_provided_relationships,
        param_dict,
        parsed.calib_dict,
        deterministic_dict,
        on_unused_parameters=on_unused_parameters,
        distributions=parsed.distributions,
        distribution_param_names=parsed.distribution_param_names,
    )

    steady_state_equations = system_to_steady_state(equations, parsed.shocks)

    user_provided_ss_vars = list(ss_solution_dict.to_sympy().keys())
    if infer_steady_state:
        ss_solution_dict = propagate_steady_state_through_identities(
            ss_solution_dict, steady_state_equations, variables
        )

    all_ss_vars = list(ss_solution_dict.to_sympy().keys())
    inferred_ss_vars = [var for var in all_ss_vars if var not in user_provided_ss_vars]
    steady_state_relationships = [sp.Eq(var, eq) for var, eq in ss_solution_dict.to_sympy().items()]

    variables = sorted(variables, key=natural_sort_key)
    shocks = sorted(parsed.shocks, key=natural_sort_key)

    if verbose:
        build_report(
            equations,
            param_dict,
            parsed.calib_dict,
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

    return _ModelPrimitives(
        variables=variables,
        shocks=shocks,
        equations=equations,
        steady_state_relationships=steady_state_relationships,
        steady_state_equations=steady_state_equations,
        ss_solution_dict=ss_solution_dict,
        param_dict=param_dict,
        hyper_param_dict=hyper_param_dict,
        deterministic_dict=deterministic_dict,
        calib_dict=parsed.calib_dict,
        param_priors=param_priors,
        shock_priors=shock_priors,
        options=parsed.options,
    )


def _mode_from_deprecated_backend(backend: str) -> str | None:
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
    return "FAST_COMPILE" if backend == "numpy" else None


def _print_parse_error(error: GCNParseError | GCNErrorCollection, gcn_path: Path) -> None:
    formatter = ErrorFormatter()
    if isinstance(error, GCNErrorCollection):
        print(formatter.format_error_collection(error), file=sys.stderr)  # noqa: T201
        return

    source = gcn_path.read_text(encoding="utf-8") if gcn_path.exists() else None
    print(formatter.format_error(error, source), file=sys.stderr)  # noqa: T201


def _select_loglin_variables(
    variables: list[TimeAwareSymbol],
    log_linearize: bool,
    not_loglin_variables: list[str] | None,
) -> list[TimeAwareSymbol]:
    if not_loglin_variables is None:
        not_loglin_variables = []

    var_names = [get_name(variable, base_name=True) for variable in variables]
    unknown = set(not_loglin_variables) - set(var_names)
    if unknown:
        raise ValueError(
            f"The following variables were requested not to be log-linearized, but are unknown to the model: "
            f"{', '.join(unknown)}"
        )

    if not log_linearize:
        return []

    return [variable for variable in variables if variable.base_name not in not_loglin_variables]


def _block_dict_to_sub_dict(block_dict: dict) -> dict[sp.Expr, sp.Expr]:
    """Collect ``lhs -> rhs`` substitutions from every block's identities, objective, and constraints."""
    sub_dict = {}
    for block in block_dict.values():
        for group in ["identities", "objective", "constraints"]:
            group_equations = getattr(block, group, None)
            if group_equations is None:
                continue
            for eq in group_equations.values():
                sub_dict[eq.lhs] = eq.rhs

    return sub_dict


def _apply_simplifications(
    try_reduce_vars: list[TimeAwareSymbol],
    equations: list[sp.Expr],
    variables: list[TimeAwareSymbol],
    tryreduce_sub_dict: dict[sp.Expr, sp.Expr],
    use_tryreduce: bool,
    use_constants: bool,
) -> tuple[list[sp.Expr], list[TimeAwareSymbol], list[TimeAwareSymbol] | None, list[TimeAwareSymbol] | None]:
    eliminated_variables = None
    singletons = None

    if use_tryreduce:
        equations, variables, eliminated_variables = simplify_tryreduce(
            try_reduce_vars, equations, variables, tryreduce_sub_dict
        )

    if use_constants:
        equations, variables, singletons = simplify_constants(equations, variables)

    return equations, variables, eliminated_variables, singletons


def _resolve_deterministic_params(
    deterministic_dict: SymbolDictionary,
    equations: list[sp.Expr],
    steady_state_relationships: list[sp.Expr],
) -> tuple[SymbolDictionary, list[sp.Symbol]]:
    """
    Substitute deterministic parameters into each other, then drop any that no longer appear in the system.

    Parameters
    ----------
    deterministic_dict : SymbolDictionary
        Deterministic parameter definitions.
    equations : list of sympy expression
        Model equations.
    steady_state_relationships : list of sympy expression
        Steady-state equations.

    Returns
    -------
    deterministic_dict : SymbolDictionary
        Fully substituted definitions of the parameters that remain, keyed by name.
    reduced_params : list of sympy Symbol
        Parameters dropped because no equation refers to them.
    """
    resolved = deterministic_dict.to_sympy()
    flat = flatten_substitution_dict(dict(resolved))
    for param in resolved:
        resolved[param] = flat[param]

    all_equations = equations + steady_state_relationships
    reduced_params = [param for param in resolved if not any(eq.has(param) for eq in all_equations)]
    for param in reduced_params:
        del resolved[param]

    return resolved.to_string(), reduced_params


def _split_distributions(
    distributions: SymbolDictionary,
    shock_distributions: SymbolDictionary,
    shock_names: set[str],
) -> tuple[SymbolDictionary, SymbolDictionary]:
    """Partition the parsed distributions into parameter priors and shock priors."""
    shock_priors = SymbolDictionary(shock_distributions)
    shock_hyper_param_names = {
        name for dist in shock_priors.values() for name in dist.param_name_to_hyper_name.values()
    }

    param_priors = SymbolDictionary()
    for name, dist in distributions.items():
        if name not in shock_names and name not in shock_hyper_param_names:
            param_priors[name] = dist

    return param_priors, shock_priors


def _collect_distribution_atoms(
    distributions: SymbolDictionary | None,
    distribution_param_names: set[str] | None,
    joint_dict: SymbolDictionary,
) -> set[sp.Symbol]:
    """Collect every sympy symbol a prior distribution refers to, so those parameters count as used."""
    atoms: set[sp.Symbol] = set()

    if distributions:
        for dist in distributions.values():
            for arg in getattr(dist, "args", ()):
                if isinstance(arg, sp.Expr):
                    atoms |= arg.atoms(sp.Symbol)

    if distribution_param_names:
        name_to_symbol = {str(symbol): symbol for symbol in joint_dict.to_sympy()}
        atoms |= {name_to_symbol[name] for name in distribution_param_names if name in name_to_symbol}

    return atoms


def _format_ss_var_list(label: str, variables: list[TimeAwareSymbol], line_width: int = 80) -> str:
    """Format ``label: a, b, c`` with continuation lines aligned under the first name."""
    prefix = f"\t\t{label}: "
    indent = " " * len(prefix.expandtabs())
    names = [str(variable) for variable in variables]

    lines = []
    current_line = prefix
    for i, name in enumerate(names):
        separator = ", " if i < len(names) - 1 else ""
        candidate = name + separator

        if i > 0 and len(current_line.expandtabs()) + len(candidate) > line_width:
            lines.append(current_line + "\n")
            current_line = indent + candidate
        else:
            current_line += candidate

    lines.append(current_line + "\n")
    return "".join(lines)


def _pluralize(word: str, count: int) -> str:
    if count == 1:
        return word
    return "have" if word == "has" else word + "s"
