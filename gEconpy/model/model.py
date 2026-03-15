import difflib
import logging

from collections.abc import Callable, Sequence
from copy import deepcopy
from typing import Literal

import numpy as np
import sympy as sp

from better_optimize import minimize, root
from preliz.distributions.distributions import Distribution
from pytensor import tensor as pt
from pytensor.graph.traversal import explicit_graph_inputs
from pytensor.tensor.variable import TensorVariable

from gEconpy.classes.containers import SteadyStateResults, SymbolDictionary
from gEconpy.classes.distributions import CompositeDistribution
from gEconpy.classes.time_aware_symbol import TimeAwareSymbol
from gEconpy.exceptions import (
    GensysFailedException,
    ModelUnknownParameterError,
)
from gEconpy.model.compile import compile_for_scipy, make_cache_key, pack_and_compile
from gEconpy.model.parameters import compile_param_dict_func
from gEconpy.model.perturbation import (
    check_perturbation_solution,
    make_not_loglin_flags,
)
from gEconpy.model.perturbation import (
    linearize_model as _linearize_model,
)
from gEconpy.model.statistics import (
    _maybe_solve_steady_state,
)
from gEconpy.model.steady_state import (
    ERROR_FUNCTIONS,
    _ss_residual_to_pytensor,
    build_minimize_graphs,
    build_root_graphs,
    compile_known_ss,
    system_to_steady_state,
)
from gEconpy.pytensorf.compile import compile_pytensor_function
from gEconpy.solvers.backward_looking import solve_policy_function_with_backward_direct
from gEconpy.solvers.cycle_reduction import solve_policy_function_with_cycle_reduction
from gEconpy.solvers.gensys import (
    interpret_gensys_output,
    solve_policy_function_with_gensys,
)
from gEconpy.utilities import get_name, postprocess_optimizer_res, safe_to_ss

VariableType = sp.Symbol | TimeAwareSymbol
_log = logging.getLogger(__name__)


def infer_variable_bounds(variable):
    assumptions = variable.assumptions0
    is_positive = assumptions.get("positive", False)
    is_negative = assumptions.get("negative", False)
    lhs = 1e-8 if is_positive else None
    rhs = -1e-8 if is_negative else None

    return lhs, rhs


def _initialize_x0(optimizer_kwargs, variables, jitter_x0):
    n_variables = len(variables)

    use_default_x0 = "x0" not in optimizer_kwargs
    x0 = optimizer_kwargs.pop("x0", np.full(n_variables, 0.8))

    if use_default_x0:
        negative_idx = [x.assumptions0.get("negative", False) for x in variables]
        x0[negative_idx] = -x0[negative_idx]

    if jitter_x0:
        rng = np.random.default_rng()
        x0 += rng.normal(scale=1e-4, size=n_variables)

    return x0


class Model:
    """
    A Dynamic Stochastic General Equilibrium (DSGE) Model.

    Stores model primitives (variables, parameters, shocks, equations) and compiles steady-state and linearization
    functions lazily on first use. Construction accepts raw sympy objects; all graph building and pytensor compilation
    is deferred to the point of use so that only the graphs actually needed are ever built.
    """

    def __init__(
        self,
        variables: list[TimeAwareSymbol],
        shocks: list[TimeAwareSymbol],
        equations: list[sp.Expr],
        steady_state_relationships: list[sp.Eq],
        steady_state_equations: list[sp.Expr],
        ss_solution_dict: SymbolDictionary,
        param_dict: SymbolDictionary,
        hyper_param_dict: SymbolDictionary,
        deterministic_dict: SymbolDictionary,
        calib_dict: SymbolDictionary,
        priors: tuple,
        is_linear: bool = False,
        mode: str | None = None,
        error_func: ERROR_FUNCTIONS = "squared",
    ) -> None:
        """Initialize a DSGE model from sympy primitives.

        Parameters
        ----------
        variables : list of TimeAwareSymbol
            Model variables.
        shocks : list of TimeAwareSymbol
            Exogenous shocks.
        equations : list of sp.Expr
            Model equations.
        steady_state_relationships : list of sp.Eq
            Analytical steady-state relationships.
        steady_state_equations : list of sp.Expr
            Steady-state equations in residual form (each equals zero at the steady state).
        ss_solution_dict : SymbolDictionary
            Analytically known steady-state solutions.
        param_dict : SymbolDictionary
            Free parameter names and default values.
        hyper_param_dict : SymbolDictionary
            Shock distribution hyperparameters.
        deterministic_dict : SymbolDictionary
            Deterministic parameter definitions.
        calib_dict : SymbolDictionary
            Calibration equations.
        priors : tuple
            ``(param_priors, shock_priors)`` — prior distribution dictionaries.
        is_linear : bool
            Whether the model is linear.
        mode : str or None
            Pytensor compilation mode (e.g. ``'FAST_COMPILE'``, ``'FAST_RUN'``).
        error_func : str
            Error metric for minimize-based steady-state solving.
        """
        self._variables = variables
        self._shocks = shocks
        self._equations = equations
        self._params = list(param_dict.to_sympy().keys())
        self.is_linear = is_linear
        self._backward_looking = not any(x.time_index == 1 for eq in equations for x in eq.atoms(TimeAwareSymbol))

        self._hyper_params = list(hyper_param_dict.to_sympy().keys())
        self._deterministic_params = list(deterministic_dict.to_sympy().keys())
        self._calibrated_params = list(calib_dict.to_sympy().keys())

        self._steady_state_relationships = steady_state_relationships

        self._all_names_to_symbols = {
            get_name(x, base_name=True): x
            for x in (self.variables + self.params + self.calibrated_params + self.deterministic_params + self.shocks)
        }

        self._priors = priors
        self._default_params = param_dict.copy()

        # Sympy primitives stored for lazy graph construction
        self._steady_state_equations = steady_state_equations
        self._ss_solution_dict = ss_solution_dict
        self._param_dict = param_dict
        self._deterministic_dict = deterministic_dict
        self._calib_dict = calib_dict
        self._mode = mode
        self._error_func = error_func

        # Lazily populated
        self._cache: dict | None = None
        self._f_params: Callable | None = None
        self._f_ss: Callable | None = None
        self._equation_tensors: list[TensorVariable] | None = None
        self._full_equation_tensors: list[TensorVariable] | None = None

    def _ensure_cache(self) -> dict:
        """Return the shared sympytensor cache, creating it on first call.

        The cache maps sympy symbol identifiers to pytensor graph nodes and is shared across all graph-building
        operations (residual, linearization, etc.) to ensure consistent node identity.
        """
        if self._cache is None:
            self._cache = {}

            compile_param_dict_func(self._param_dict, self._deterministic_dict, cache=self._cache, return_symbolic=True)
        return self._cache

    def _ensure_equation_tensors(self, filter_known: bool = False) -> list[TensorVariable]:
        """Build and cache the pytensor equation graphs.

        When ``filter_known`` is False (default), all equations are returned with all SS variables and parameters as
        free inputs. When True, equations that are fully determined by analytically known steady-state values are
        removed, and the known SS values are substituted into the remaining equations.
        """
        if filter_known:
            if self._equation_tensors is None:
                cache = self._ensure_cache()
                self._equation_tensors, _ = _ss_residual_to_pytensor(
                    self._steady_state_equations,
                    self._ss_solution_dict,
                    self._variables,
                    self._param_dict,
                    self._deterministic_dict,
                    self._calib_dict,
                    cache=cache,
                )
            return self._equation_tensors

        if self._full_equation_tensors is None:
            cache = self._ensure_cache()
            self._full_equation_tensors, _ = _ss_residual_to_pytensor(
                self._steady_state_equations,
                SymbolDictionary(),
                self._variables,
                self._param_dict,
                self._deterministic_dict,
                self._calib_dict,
                cache=cache,
            )
        return self._full_equation_tensors

    def ss_tensors(self, filter_known: bool = False) -> list[TensorVariable]:
        """Pytensor scalar variables for the model's steady-state symbols.

        Returns one scalar ``TensorVariable`` per model variable plus calibrated parameter, in model order. Variables
        are created in the shared sympytensor cache on first access so that repeated calls and downstream
        graph-building share the same object identity.

        Parameters
        ----------
        filter_known : bool
            If True, exclude variables whose steady-state values are analytically known (from the ``STEADY_STATE``
            block). Default is False (return all).

        Returns
        -------
        list of TensorVariable
        """
        cache = self._ensure_cache()

        if filter_known and self._ss_solution_dict:
            known_names = {safe_to_ss(k).name for k in self._ss_solution_dict.to_sympy()}
        else:
            known_names = set()

        all_vars = list(self._variables) + list(self._calib_dict.to_sympy().keys())
        result = []
        for v in all_vars:
            ss_sym = v.to_ss() if hasattr(v, "to_ss") else v
            if ss_sym.name in known_names:
                continue
            cache_key = make_cache_key(ss_sym.name, type(ss_sym))
            if cache_key not in cache:
                cache[cache_key] = pt.scalar(name=ss_sym.name, dtype="floatX")
            result.append(cache[cache_key])
        return result

    def param_tensors(
        self,
        include_free: bool = True,
        include_deterministic: bool = True,
        include_calibrated: bool = False,
    ) -> list[TensorVariable]:
        """Pytensor scalar variables for the model's parameters.

        By default returns free and deterministic parameters. Use the ``include_*`` flags to select which parameter
        groups appear in the result.

        Parameters
        ----------
        include_free : bool
            Include free parameters (those with numeric defaults). Default is True.
        include_deterministic : bool
            Include deterministic parameters (defined as functions of other parameters). Default is True.
        include_calibrated : bool
            Include calibrated parameters (pinned by steady-state equations). Default is False.

        Returns
        -------
        list of TensorVariable
        """
        cache = self._ensure_cache()
        param_symbols: list[sp.Symbol] = []
        if include_free:
            param_symbols += list(self._param_dict.to_sympy().keys())
        if include_deterministic:
            param_symbols += list(self._deterministic_dict.to_sympy().keys())
        if include_calibrated:
            param_symbols += list(self._calib_dict.to_sympy().keys())

        result = []
        for p in param_symbols:
            cache_key = make_cache_key(p.name, type(p))
            if cache_key not in cache:
                cache[cache_key] = pt.scalar(name=p.name, dtype="floatX")
            result.append(cache[cache_key])
        return result

    @property
    def f_params(self) -> Callable:
        """Compiled function mapping free parameter values to the full parameter dictionary."""
        if self._f_params is None:
            f, _cache = compile_param_dict_func(self._param_dict, self._deterministic_dict)
            self._f_params = f
        return self._f_params

    @property
    def f_ss(self) -> Callable | None:
        """Compiled function mapping parameters to known steady-state values, or None if no analytical solutions."""
        if self._f_ss is None:
            if not self._ss_solution_dict:
                return None

            _, cache = compile_param_dict_func(self._param_dict, self._deterministic_dict)
            all_params = list(self._param_dict.to_sympy().keys()) + list(self._deterministic_dict.to_sympy().keys())
            f_ss, _cache = compile_known_ss(
                self._ss_solution_dict,
                self._variables,
                all_params,
                cache=cache,
            )
            self._f_ss = f_ss
        return self._f_ss

    def equation_tensors(self, filter_known: bool = False) -> list[TensorVariable]:
        """Pytensor graphs for the model's steady-state equations.

        Returns one scalar ``TensorVariable`` per equation, each equal to zero at the steady state. All variables
        and parameters in the shared sympytensor cache are used as graph inputs, so the returned graphs share node
        identity with ``ss_tensors`` and ``param_tensors``.

        Parameters
        ----------
        filter_known : bool
            If True, substitute analytically known steady-state values and drop equations that become fully
            determined. Default is False (return all equations with all inputs).

        Returns
        -------
        list of TensorVariable
        """
        return self._ensure_equation_tensors(filter_known=filter_known)

    @property
    def sympy_to_pytensor_cache(self) -> dict:
        """Cache mapping sympy symbol identifiers to pytensor graph nodes.

        Maintained across graph-building calls to ensure consistent node identity. Useful for advanced users who need
        to look up or construct pytensor nodes corresponding to specific model symbols.
        """
        return self._ensure_cache()

    @property
    def _vars_to_solve(self) -> list[sp.Symbol]:
        """Steady-state variables that must be solved numerically (not provided analytically)."""
        known_names = (
            {safe_to_ss(k).name for k in self._ss_solution_dict.to_sympy()} if self._ss_solution_dict else set()
        )
        ss_variables = [x.to_ss() for x in self._variables] + list(self._calib_dict.to_sympy().keys())
        return [v for v in ss_variables if v.name not in known_names]

    @property
    def variables(self) -> list[TimeAwareSymbol]:
        """
        List of variables in the model, stored as Sympy symbols.

        Variables are associated with the model;s endogenous states, identified by the presence of a time subscript.
        """
        return self._variables

    @property
    def shocks(self) -> list[TimeAwareSymbol]:
        """
        List of shocks in the model, stored as Sympy symbols.

        Shocks are exogenous variables in the model, and the source of stochasticity in the model.
        """
        return self._shocks

    @property
    def equations(self) -> list[sp.Expr]:
        """List of equations in the model, stored as Sympy expressions."""
        return self._equations

    @property
    def params(self) -> list[sp.Symbol]:
        """
        List of parameters in the model, stored as :class:`sympy.Symbol` objects.

        Parameters are fixed values in the model, associated with the structural equations of the model. These are
        sometimes called "deep parameters" because of their (supposed) microeconomic foundations.
        """
        return self._params

    @property
    def hyper_params(self) -> list[sp.Symbol]:
        """
        List of hyperparameters in the model, stored as :class:`sympy.Symbol` objects.

        Hyperparameters are parameters associated with the distribution of shocks in the model, for example the
        standard deviation of a normally distributed shock.
        """
        return self._hyper_params

    @property
    def deterministic_params(self) -> list[sp.Symbol]:
        """
        List of deterministic parameters in the model, stored as :class:`sympy.Symbol` objects.

        Deterministic parameters are parameters defined as functions of other parameters in the model. They are
        not directly calibrated, but are instead derived deterministically from other parameters.
        """
        return self._deterministic_params

    @property
    def param_priors(self) -> dict[str, Distribution]:
        """
        Dictionary of prior distributions for the model parameters.

        The dictionary keys are parameter names, and the values are instances of :class:`preliz.Distribution`.
        """
        return self._priors[0]

    @property
    def shock_priors(self) -> dict[str, CompositeDistribution]:
        """
        Dictionary of prior distributions for the model shocks.

        The dictionary keys are shock names, and the values are instances of :class:`preliz.Distribution`.
        """
        return self._priors[1]

    @property
    def calibrated_params(self) -> list[sp.Symbol]:
        """
        List of calibrated parameters in the model, stored as :class:`sympy.Symbol` objects.

        Calibrated parameters are pseudo-parameters whose values are an implicit function of the model parameters.
        Each calibrated parameter must be associated with a function of steady-state variables. This function is added
        to the model equations when solving for the steady state, and the calibrated parameter is then solved for
        numerically.
        """
        return self._calibrated_params

    @property
    def steady_state_relationships(self) -> list[sp.Eq]:
        """List of model equations, evaluated at the deterministic steady state."""
        return self._steady_state_relationships

    @property
    def n_variables(self) -> int:
        """Number of endogenous variables in the model."""
        return len(self._variables)

    @property
    def backward_variables(self) -> list[TimeAwareSymbol]:
        """Variables that appear at t-1 in at least one equation (state variables)."""
        if not hasattr(self, "_backward_variables"):
            lagged = {a.set_t(0) for eq in self._equations for a in eq.atoms(TimeAwareSymbol) if a.time_index == -1}
            self._backward_variables = [v for v in self._variables if v in lagged]
        return self._backward_variables

    @property
    def forward_variables(self) -> list[TimeAwareSymbol]:
        """Variables that appear at t+1 in at least one equation (forward-looking/jump variables)."""
        if not hasattr(self, "_forward_variables"):
            leads = {a.set_t(0) for eq in self._equations for a in eq.atoms(TimeAwareSymbol) if a.time_index == 1}
            self._forward_variables = [v for v in self._variables if v in leads]
        return self._forward_variables

    @property
    def n_backward(self) -> int:
        """Number of backward-looking (state) variables."""
        return len(self.backward_variables)

    @property
    def n_forward(self) -> int:
        """Number of forward-looking (jump) variables."""
        return len(self.forward_variables)

    @property
    def lead_var_idx(self) -> np.ndarray:
        """Column indices of forward-looking variables in the Jacobian matrices."""
        if not hasattr(self, "_lead_var_idx"):
            fwd_set = set(self.forward_variables)
            self._lead_var_idx = np.array([i for i, v in enumerate(self._variables) if v in fwd_set], dtype=int)
        return self._lead_var_idx

    def parameters(self, **updates: float) -> SymbolDictionary[str, float]:
        """
        Compute the full set of free parameters for the model, including deterministic parameters.

        Calibrated parameters are not returned by this function. These are computed as part of the steady-state
        solution.

        If a parameter is not provided in the updates, the default value (as defined in the model GCN file) is used.

        Parameters
        ----------
        updates: float
            Parameters to update. These are passed as keyword arguments, with the parameter name as the keyword and the
            new value as the value.

        Returns
        -------
        SymbolDictionary
            Dictionary of parameter names and values.
        """
        # Remove deterministic parameters for updates. These can appear **self.parameters() into a fitting function
        deterministic_names = [x.name for x in self.deterministic_params]
        updates = {k: v for k, v in updates.items() if k not in deterministic_names}

        # Check for unknown updates (typos, etc)
        param_dict = self._default_params.copy()
        unknown_updates = set(updates.keys()) - set(param_dict.keys())
        if unknown_updates:
            raise ModelUnknownParameterError(list(unknown_updates))
        param_dict.update(updates)

        return self.f_params(**param_dict).to_string()

    def get(self, name: str) -> sp.Symbol:
        """
        Get a model variable or parameter by name.

        Variables are returned as TimeAwareSymbols, and parameters are returned as regular Sympy Symbols. If the name
        ends with "_ss", the steady-state version of the variable is returned.

        Parameters
        ----------
        name: str
            Name of the variable or parameter to retrieve

        Returns
        -------
        sp.Symbol
            The requested variable or parameter.
        """
        ss_requested = name.endswith("_ss")
        name = name.removesuffix("_ss")

        result = self._all_names_to_symbols.get(name)
        if result is None:
            close_match = difflib.get_close_matches(name, [get_name(x) for x in self._all_names_to_symbols], n=1)[0]
            raise IndexError(f"Did not find {name} among model objects. Did you mean {close_match}?")
        if ss_requested:
            return result.to_ss()
        return result

    def _validate_provided_steady_state_variables(self, user_fixed_variables: Sequence[str]):
        # User is allowed to pass the variable name either with or without the _ss suffix. Begin by normalizing the
        # inputs
        fixed_variables_normed = [x.removesuffix("_ss") for x in user_fixed_variables]

        # Check for duplicated values. This should only be possible if the user passed both `x` and `x_ss`.
        counts = [fixed_variables_normed.count(x) for x in fixed_variables_normed]
        duplicates = [x for x, c in zip(fixed_variables_normed, counts, strict=False) if c > 1]
        if len(duplicates) > 0:
            raise ValueError(
                "The following variables were provided twice (once with a _ss prefix and once without):\n"
                f"{', '.join(duplicates)}"
            )

        # Check that all variables are in the model
        model_variable_names = [x.base_name for x in self.variables]
        unknown_fixed = set(fixed_variables_normed) - set(model_variable_names)

        if len(unknown_fixed) > 0:
            raise ValueError(
                f"The following variables or calibrated parameters were given fixed steady state values but are "
                f"unknown to the model: {', '.join(unknown_fixed)}"
            )

    def _linear_steady_state(self) -> SteadyStateResults:
        """Return the trivial zero steady state for a linear model."""
        ss_dict = SteadyStateResults({x.to_ss(): 0.0 for x in self.variables}).to_string()
        ss_dict.success = True
        return ss_dict

    def _try_analytic_steady_state(
        self,
        param_dict: SymbolDictionary,
        f_ss: Callable | None,
    ) -> SteadyStateResults | None:
        """Attempt a pure analytic solve. Returns None if the analytic solution is incomplete."""
        if f_ss is None:
            return None
        ss_dict = f_ss(**param_dict)
        if len(ss_dict) != len(self.variables):
            return None

        full_resid = pt.stack(self.equation_tensors())
        f_resid = compile_for_scipy(full_resid, mode=self._mode)
        residual = np.asarray(f_resid(**ss_dict, **param_dict))
        success = np.allclose(residual, 0.0, atol=1e-8)

        result = SteadyStateResults(ss_dict.to_sympy()).to_string()
        result.success = success
        if not success:
            _log.warning(f"Steady State was not found. Sum of square residuals: {np.square(residual).sum()}")
        return result

    def _evaluate_all_resolved(
        self,
        f_ss: Callable | None,
        param_dict: SymbolDictionary,
        fixed_values: dict[str, float] | None,
    ) -> SteadyStateResults:
        """Build and validate a steady state when every variable is already known (analytically or via fixed_values)."""
        provided_ss_values = f_ss(**param_dict).to_sympy() if f_ss is not None else {}
        if fixed_values is not None:
            provided_ss_values.update({safe_to_ss(self.get(k)): v for k, v in fixed_values.items()})

        full_resid = pt.stack(self.equation_tensors())
        f_resid = compile_for_scipy(full_resid, mode=self._mode)
        residual = np.asarray(f_resid(**{str(k): v for k, v in provided_ss_values.items()}, **param_dict))

        ss_variables = [x.to_ss() for x in self.variables] + list(self.calibrated_params)
        result = SteadyStateResults({x: provided_ss_values[x] for x in ss_variables}).to_string()
        result.success = np.allclose(residual, 0.0, atol=1e-8)
        return result

    def _postprocess_numerical_result(
        self,
        res,
        compiled_funcs: dict[str, Callable],
        vars_to_solve: list[sp.Symbol],
        f_ss: Callable | None,
        param_dict: SymbolDictionary,
        fixed_values: dict[str, float] | None,
        tol: float,
        verbose: bool,
    ) -> SteadyStateResults:
        """Assemble steady-state dict from optimizer result and run convergence diagnostics."""
        provided_ss_values = f_ss(**param_dict).to_sympy() if f_ss is not None else {}
        if fixed_values is not None:
            provided_ss_values.update({safe_to_ss(self.get(k)): v for k, v in fixed_values.items()})

        optimizer_results = SymbolDictionary({var: res.x[i] for i, var in enumerate(vars_to_solve)})
        res_dict = optimizer_results | provided_ss_values

        ss_variables = [x.to_ss() for x in self.variables] + list(self.calibrated_params)
        res_dict = SteadyStateResults({x: res_dict[x] for x in ss_variables}).to_string()

        f_resid_compiled = compiled_funcs["resid"]
        f_jac_compiled = compiled_funcs.get("jac")
        f_grad_compiled = compiled_funcs.get("grad")

        def f_resid_for_postprocess(**kw):
            return np.asarray(f_resid_compiled(**kw, **param_dict))

        def f_jac_for_postprocess(**kw):
            if f_grad_compiled is not None:
                return np.asarray(f_grad_compiled(**kw, **param_dict))
            if f_jac_compiled is not None:
                resid_val = np.asarray(f_resid_compiled(**kw, **param_dict)).ravel()
                jac_val = np.asarray(f_jac_compiled(**kw, **param_dict))
                return 2.0 * jac_val.T @ resid_val
            return np.zeros(len(vars_to_solve))

        return postprocess_optimizer_res(
            res=res,
            res_dict=res_dict,
            f_resid=f_resid_for_postprocess,
            f_jac=f_jac_for_postprocess,
            tol=tol,
            verbose=verbose,
        )

    def steady_state(
        self,
        how: Literal["analytic", "root", "minimize"] = "analytic",
        use_jac=True,
        use_hess=True,
        use_hessp=False,
        progressbar=True,
        optimizer_kwargs: dict | None = None,
        verbose=True,
        bounds: dict[str, tuple[float, float]] | None = None,
        fixed_values: dict[str, float] | None = None,
        jitter_x0: bool = False,
        **updates: float,
    ) -> SteadyStateResults:
        r"""
        Solve for the deterministic steady state of the DSGE model.

        A steady state is defined as the fixed point in the system of  nonlinear equations that describe the model's
        equilibrium. Given a system of model equations :math:`F(x_{t+1}, x_t, x_{t-1}, \varepsilon_t)`, the steady state
        is defined as a state vector :math:`\bar{x}` such that

        .. math::

            F(\bar{x}, \bar{x}, \bar{x}, 0) = 0

        where :math:`0` is the zero vector. At the point :math:`\bar{x}`, the system will not change, absent an
        exogenous shock.

        The steady state is a key concept in DSGE modeling, as it is the point around which the model is linearized.

        Parameters
        ----------
        how: str, one of ['analytic', 'root', 'minimize'], default: 'analytic'
            Method to use to solve for the steady state. If ``'analytic'``, the model is solved analytically using
            user-provided steady-state equations. This is only possible if the steady-state equations are fully
            defined. If ``'root'``, the steady state is solved using a root-finding algorithm. If ``'minimize'``, the
            steady state is solved by minimizing a squared error loss function.

        use_jac: bool, default: True
            Flag indicating whether to use the Jacobian of the error function when solving for the steady state. Ignored
            if ``how`` is 'analytic'.

        use_hess: bool, default: False
            Flag indicating whether to use the Hessian of the error function when solving for the steady state. Ignored
            if ``how`` is not 'minimize'

        use_hessp: bool, default: True
            Flag indicating whether to use the Hessian-vector product of the error function when solving for the
            steady state. This should be preferred over ``use_hess`` if your chosen method supports it. For larger
            problems it is substantially more performant.
            Ignored if ``how`` not "minimize".

        progressbar: bool, default: True
            Flag indicating whether to display a progress bar when solving for the steady state.

        optimizer_kwargs: dict, optional
            Keyword arguments passed to either scipy.optimize.root or scipy.optimize.minimize, depending on the value of
            ``how``. Common argments include:

            - 'method': str,
                The optimization method to use. Default is ``'hybr'`` for ``how = 'root'`` and ``trust-krylov`` for
                ``how = 'minimize'``
            - 'maxiter': int,
                The maximum number of iterations to use. Default is 5000. This argument will be automatically renamed
                to match the argument expected by different optimizers (for example, the ``'hybr'`` method uses
                ``maxfev``).

        verbose: bool, default True
            If true, print a message about convergence (or not) to the console .

        bounds: dict, optional
            Dictionary of bounds for the steady-state variables. The keys are the variable names and the values are
            tuples of the form (lower_bound, upper_bound). These are passed to the scipy.optimize.minimize function,
            see that docstring for more information.

        fixed_values: dict, optional
            Dictionary of fixed values for the steady-state variables. The keys are the variable names and the values
            are the fixed values. These are not check for validity, and passing an inaccurate value may result in the
            system becoming unsolvable.

        jitter_x0: bool
            Whether to apply some small N(0, 1e-4) jitter to the initial point

        **updates: float, optional
            Parameter values at which to solve the steady state. Passed to self.parameters. If not provided, the default
            parameter values (those originally defined during model construction) are used.

        Returns
        -------
        steady_state: SteadyStateResults
            Dictionary of steady-state values

        """
        if optimizer_kwargs is None:
            optimizer_kwargs = {}

        tol = optimizer_kwargs.get("tol", 1e-8)
        param_dict = self.parameters(**updates)
        f_ss = self.f_ss

        if self.is_linear:
            return self._linear_steady_state()

        if fixed_values is None:
            analytic_result = self._try_analytic_steady_state(param_dict, f_ss)
            if analytic_result is not None:
                return analytic_result

        if how == "analytic":
            how = "minimize"

        if fixed_values is not None:
            self._validate_provided_steady_state_variables(list(fixed_values.keys()))
            equations, vars_to_solve, ss_nodes = self._build_resid_with_fixed_values(fixed_values, param_dict)
        else:
            equations = self.equation_tensors(filter_known=True)
            ss_nodes = self.ss_tensors(filter_known=True)
            vars_to_solve = self._vars_to_solve

        if not vars_to_solve:
            return self._evaluate_all_resolved(f_ss, param_dict, fixed_values)

        if how == "root":
            n_eqs = len(equations)
            n_vars = len(vars_to_solve)
            if n_eqs != n_vars:
                raise ValueError(
                    'Solving a partially provided steady state with how = "root" is only allowed if applying '
                    f"the given values results in a new square system.\n"
                    f"Remaining: {n_vars} variable{'s' if n_vars != 1 else ''}, "
                    f"{n_eqs} equation{'s' if n_eqs != 1 else ''}."
                )
            res, compiled_funcs = self._solve_steady_state_with_root(
                equations,
                ss_nodes,
                vars_to_solve,
                use_jac=use_jac,
                progressbar=progressbar,
                optimizer_kwargs=optimizer_kwargs,
                jitter_x0=jitter_x0,
                **updates,
            )
        elif how == "minimize":
            res, compiled_funcs = self._solve_steady_state_with_minimize(
                equations,
                ss_nodes,
                vars_to_solve,
                use_jac=use_jac,
                use_hess=use_hess,
                use_hessp=use_hessp,
                progressbar=progressbar,
                bounds=bounds,
                optimizer_kwargs=optimizer_kwargs,
                jitter_x0=jitter_x0,
                **updates,
            )
        else:
            raise NotImplementedError()

        return self._postprocess_numerical_result(
            res,
            compiled_funcs,
            vars_to_solve,
            f_ss,
            param_dict,
            fixed_values,
            tol,
            verbose,
        )

    def _build_resid_with_fixed_values(
        self,
        fixed_values: dict[str, float],
        param_dict: SymbolDictionary,
    ) -> tuple[list[TensorVariable], list[sp.Symbol], list[TensorVariable]]:
        """Build a residual graph with user-supplied fixed values merged into the known-SS dict.

        Equations fully determined by the fixed values alone are validated for consistency. The remaining system
        is returned for numerical solving.
        """
        merged_ss = SymbolDictionary(self._ss_solution_dict.copy() if self._ss_solution_dict else {})
        for name, value in fixed_values.items():
            merged_ss[safe_to_ss(self.get(name))] = float(value)

        cache = self._ensure_cache()
        equations, _ = _ss_residual_to_pytensor(
            self._steady_state_equations,
            merged_ss,
            self._variables,
            self._param_dict,
            self._deterministic_dict,
            self._calib_dict,
            cache=cache,
        )

        self._validate_fixed_value_equations(fixed_values, param_dict)

        merged_names = {safe_to_ss(k).name for k in merged_ss.to_sympy()}
        ss_variables = [x.to_ss() for x in self._variables] + list(self._calib_dict.to_sympy().keys())
        vars_to_solve = [v for v in ss_variables if v.name not in merged_names]

        ss_nodes = []
        for v in vars_to_solve:
            ck = make_cache_key(v.name, type(v))
            if ck in cache:
                ss_nodes.append(cache[ck])

        return equations, vars_to_solve, ss_nodes

    def _validate_fixed_value_equations(self, fixed_values: dict[str, float], param_dict: SymbolDictionary) -> None:
        """Check that equations fully determined by fixed values have zero residuals.

        Only equations whose every SS-variable input is in ``fixed_values`` are evaluated; equations involving any
        unfixed variable are skipped.
        """
        all_equations = self.equation_tensors()
        cache = self._ensure_cache()
        fixed_kw = {safe_to_ss(self.get(k)).name: float(v) for k, v in fixed_values.items()}

        all_vars = list(self._variables) + list(self._calib_dict.to_sympy().keys())
        non_fixed_ids: set[int] = set()
        for v in all_vars:
            ss_sym = v.to_ss() if hasattr(v, "to_ss") else v
            if ss_sym.name not in fixed_kw:
                ck = make_cache_key(ss_sym.name, type(ss_sym))
                if ck in cache:
                    non_fixed_ids.add(id(cache[ck]))

        fully_determined_indices = []
        for i, eq in enumerate(all_equations):
            if not any(id(inp) in non_fixed_ids for inp in explicit_graph_inputs(eq)):
                fully_determined_indices.append(i)

        if not fully_determined_indices:
            return

        check_kw = {**fixed_kw, **param_dict}
        determined_eqs = [all_equations[i] for i in fully_determined_indices]
        determined_resid = pt.stack(determined_eqs)
        f_resid = compile_for_scipy(determined_resid, mode=self._mode)
        residuals = np.asarray(f_resid(**check_kw))
        bad_indices = [fully_determined_indices[j] for j, val in enumerate(residuals) if abs(val) > 1e-8]

        if bad_indices:
            ss_system = system_to_steady_state(self.equations, self.shocks)
            bad_strs = [str(ss_system[i]) for i in bad_indices if i < len(ss_system)]
            raise ValueError(
                "User-provided steady state is not valid. The following equations had non-zero residuals "
                "after substitution:\n" + "\n".join(bad_strs)
            )

    def _evaluate_steady_state(self, **updates: float) -> np.ndarray:
        """Evaluate the full residual system at the analytical SS values and current parameters."""
        param_dict = self.parameters(**updates)
        full_resid = pt.stack(self.equation_tensors())
        f_resid = compile_for_scipy(full_resid, mode=self._mode)
        ss_dict = self.f_ss(**param_dict) if self.f_ss else {}
        return np.asarray(f_resid(**ss_dict, **param_dict))

    def evaluate_residual(self, ss_dict: dict[str, float], param_dict: SymbolDictionary) -> np.ndarray:
        """Evaluate the steady-state residual at given variable and parameter values."""
        full_resid = pt.stack(self.equation_tensors())
        f_resid = compile_for_scipy(full_resid, mode=self._mode)
        return np.asarray(f_resid(**ss_dict, **param_dict))

    def _solve_steady_state_with_root(
        self,
        equations: list[TensorVariable],
        ss_nodes: list[TensorVariable],
        vars_to_solve: list[sp.Symbol],
        use_jac: bool = True,
        progressbar: bool = True,
        optimizer_kwargs: dict | None = None,
        jitter_x0: bool = False,
        **param_updates,
    ):
        if optimizer_kwargs is None:
            optimizer_kwargs = {}
        optimizer_kwargs = deepcopy(optimizer_kwargs)

        maxiter = optimizer_kwargs.pop("maxiter", 5000)
        method = optimizer_kwargs.pop("method", "hybr")

        if "options" not in optimizer_kwargs:
            optimizer_kwargs["options"] = {}
        if method in ["hybr", "df-sane"]:
            optimizer_kwargs["options"].update({"maxfev": maxiter})
        else:
            optimizer_kwargs["options"].update({"maxiter": maxiter})

        x0 = _initialize_x0(optimizer_kwargs, vars_to_solve, jitter_x0)
        param_dict = self.parameters(**param_updates)

        resid, jac_graph = build_root_graphs(equations, ss_nodes, use_jac=use_jac)

        f_resid = pack_and_compile(resid, ss_nodes, param_dict=param_dict, mode=self._mode)
        f_jac = (
            pack_and_compile(jac_graph, ss_nodes, param_dict=param_dict, mode=self._mode)
            if jac_graph is not None
            else None
        )

        f_resid_kw = compile_for_scipy(resid, mode=self._mode)
        f_jac_kw = compile_for_scipy(jac_graph, mode=self._mode) if jac_graph is not None else None

        with np.errstate(all="ignore"):
            res = root(f=f_resid, x0=x0, jac=f_jac, method=method, progressbar=progressbar, **optimizer_kwargs)

        compiled_funcs = {"resid": f_resid_kw}
        if f_jac_kw is not None:
            compiled_funcs["jac"] = f_jac_kw

        return res, compiled_funcs

    def _solve_steady_state_with_minimize(
        self,
        equations: list[TensorVariable],
        ss_nodes: list[TensorVariable],
        vars_to_solve: list[sp.Symbol],
        use_jac: bool = True,
        use_hess: bool = False,
        use_hessp: bool = True,
        progressbar: bool = True,
        optimizer_kwargs: dict | None = None,
        jitter_x0: bool = False,
        bounds: dict[str, tuple[float, float]] | None = None,
        **param_updates,
    ):
        if optimizer_kwargs is None:
            optimizer_kwargs = {}
        optimizer_kwargs = deepcopy(optimizer_kwargs)

        x0 = _initialize_x0(optimizer_kwargs, vars_to_solve, jitter_x0)
        tol = optimizer_kwargs.pop("tol", 1e-30)

        user_bounds = {} if bounds is None else bounds
        bound_dict = {x.name: infer_variable_bounds(x) for x in vars_to_solve}
        bound_dict.update(user_bounds)

        bounds_list = [bound_dict[x.name] for x in vars_to_solve]
        has_bounds = any(x != (None, None) for x in bounds_list)

        method = optimizer_kwargs.pop("method", "trust-ncg" if not has_bounds else "trust-constr")
        if method not in ["trust-constr", "L-BFGS-B", "powell"]:
            has_bounds = False

        maxiter = optimizer_kwargs.pop("maxiter", 5000)
        if "options" not in optimizer_kwargs:
            optimizer_kwargs["options"] = {}
        optimizer_kwargs["options"].update({"maxiter": maxiter})
        if method == "L-BFGS-B":
            optimizer_kwargs["options"].update({"maxfun": maxiter})

        if use_hess and use_hessp:
            _log.warning("Both use_hess and use_hessp are set to True. use_hessp will be used.")
            use_hess = False

        error_graph, grad_graph, hess_graph, hessp_graph, hessp_p = build_minimize_graphs(
            equations,
            ss_nodes,
            error_func=self._error_func,
            use_jac=use_jac,
            use_hess=use_hess,
            use_hessp=use_hessp,
        )

        param_dict = self.parameters(**param_updates)

        f_error = pack_and_compile(error_graph, ss_nodes, param_dict=param_dict, mode=self._mode)
        f_grad = (
            pack_and_compile(grad_graph, ss_nodes, param_dict=param_dict, mode=self._mode)
            if grad_graph is not None
            else None
        )
        f_hess = (
            pack_and_compile(hess_graph, ss_nodes, param_dict=param_dict, mode=self._mode)
            if hess_graph is not None
            else None
        )

        # hessp takes a dynamic direction vector that can't be baked into pack_and_compile
        f_hessp = None
        if hessp_graph is not None:
            f_hessp_inner = compile_for_scipy(hessp_graph, mode=self._mode)
            var_names = [v.name for v in vars_to_solve]

            def f_hessp(x: np.ndarray, p: np.ndarray) -> np.ndarray:
                kw = dict(zip(var_names, x, strict=True))
                kw[hessp_p.name] = p
                return np.asarray(f_hessp_inner(**kw, **param_dict))

        resid = pt.stack(equations) if equations else pt.zeros(0)
        f_resid_kw = compile_for_scipy(resid, mode=self._mode)
        f_grad_kw = compile_for_scipy(grad_graph, mode=self._mode) if grad_graph is not None else None

        compiled_funcs = {"resid": f_resid_kw}
        if f_grad_kw is not None:
            compiled_funcs["grad"] = f_grad_kw

        res = minimize(
            f=f_error,
            x0=x0,
            jac=f_grad,
            hess=f_hess,
            hessp=f_hessp,
            method=method,
            bounds=bounds_list if has_bounds else None,
            tol=tol,
            progressbar=progressbar,
            **optimizer_kwargs,
        )

        return res, compiled_funcs

    def symbolic_linearization(
        self,
        order: Literal[1] = 1,
        log_linearize: bool = True,
        not_loglin_variables: list[str] | None = None,
        steady_state: dict | None = None,
        loglin_negative_ss: bool = False,
        verbose: bool = True,
    ) -> tuple[list[TensorVariable], list[TensorVariable], list[TensorVariable]]:
        r"""
        Return the symbolic pytensor graphs for the linearized Jacobian matrices.

        Builds (and caches) the four Jacobian matrices ``A, B, C, D`` as pytensor graph nodes representing the
        first-order approximation of the model around its steady state:

        .. math::
            A \hat{y}_{t-1} + B \hat{y}_t + C \hat{y}_{t+1} + D \varepsilon_t = 0

        Unlike :meth:`linearize_model`, this method does **not** compile or numerically evaluate the graphs. The
        returned nodes can be inspected, manipulated, or compiled by the caller.

        Parameters
        ----------
        order : int, default 1
            Order of the Taylor expansion. Only ``order=1`` is currently supported.
        log_linearize : bool, default True
            If True, all variables are log-linearized. If False, all variables are left in levels.
        not_loglin_variables : list of str, optional
            Variable names to exclude from log-linearization. Ignored if ``log_linearize`` is False.
        steady_state : dict, optional
            Steady-state values used to determine which variables have non-positive steady states (and therefore
            cannot be log-linearized). If not provided, the steady state is solved internally.
        loglin_negative_ss : bool, default False
            If True, variables with negative steady-state values are still log-linearized.
        verbose : bool, default True
            Log warnings about excluded variables.

        Returns
        -------
        jacobians : list of TensorVariable
            Four pytensor matrix graph nodes ``[A, B, C, D]``.
        ss_input_nodes : list of TensorVariable
            Steady-state variable input nodes consumed by the Jacobian graphs.
        param_input_nodes : list of TensorVariable
            Parameter input nodes consumed by the Jacobian graphs (discovered via ``explicit_graph_inputs``).

        See Also
        --------
        linearize_model : Compile and numerically evaluate the linearized system.

        Examples
        --------
        .. code-block:: python

            model = model_from_gcn("rbc.gcn")
            jacobians, ss_nodes, param_nodes = model.symbolic_linearization()
            A, B, C, D = jacobians

            # Inspect the pytensor graph
            import pytensor

            pytensor.dprint(A)
        """
        if order != 1:
            raise NotImplementedError("Only first order linearization is currently supported.")

        if self.is_linear:
            log_linearize = False

        # A steady state is only needed to decide loglin flags (sign / near-zero checks).
        # If not provided, solve for one using default parameters.
        if steady_state is None and log_linearize:
            if self.is_linear:
                steady_state = self.f_ss(**self.parameters())
            else:
                steady_state = self.steady_state(**self.parameters(), verbose=verbose)

        not_loglin_flags = make_not_loglin_flags(
            variables=self.variables,
            calibrated_params=self.calibrated_params,
            steady_state=steady_state if steady_state is not None else {},
            log_linearize=log_linearize,
            not_loglin_variables=not_loglin_variables,
            loglin_negative_ss=loglin_negative_ss,
            verbose=verbose,
        )

        loglin_vars = [v for v, flag in zip(self.variables, not_loglin_flags, strict=False) if flag == 0]
        loglin_key = frozenset(v.base_name for v in loglin_vars)

        if not hasattr(self, "_symbolic_linearize_cache"):
            self._symbolic_linearize_cache = {}

        if loglin_key not in self._symbolic_linearize_cache:
            jacobians, ss_input_nodes = _linearize_model(
                variables=self.variables,
                equations=self.equations,
                shocks=self.shocks,
                cache=self._ensure_cache(),
                loglin_variables=loglin_vars,
            )

            ss_names = {n.name for n in ss_input_nodes}
            param_input_nodes = [
                v for v in explicit_graph_inputs(jacobians) if v.name is not None and v.name not in ss_names
            ]

            self._symbolic_linearize_cache[loglin_key] = (jacobians, ss_input_nodes, param_input_nodes)

        return self._symbolic_linearize_cache[loglin_key]

    def linearize_model(
        self,
        order: Literal[1] = 1,
        log_linearize: bool = True,
        not_loglin_variables: list[str] | None = None,
        steady_state: dict | None = None,
        loglin_negative_ss: bool = False,
        steady_state_kwargs: dict | None = None,
        verbose: bool = True,
        **parameter_updates,
    ):
        r"""
        Linearize the model around the deterministic steady state.

        Parameters
        ----------
        order: int, default: 1
            Order of the Taylor expansion to use. Currently only first order linearization is supported.
        log_linearize: bool, default: True
            If True, all variables are log-linearized. If False, all variables are left in levels.
        not_loglin_variables: list of strings, optional
            List of variables to not log-linearize. If provided, these variables will be left in levels, while all
            others will be log-linearized. Ignored if log_linearize is False.
        steady_state: dict, optional
            Dictionary of steady-state values. If provided, these values will be used to linearize the model. If not
            provided, the steady state will be solved for using the ``steady_state`` method.
        loglin_negative_ss: bool, default: False
            If True, variables with negative steady-state values will be log-linearized. While technically possible,
            this is not recommended, as it can lead to incorrect results. Ignored if log_linearize is False.
        steady_state_kwargs: dict, optional
            Keyword arguments passed to the ``steady_state`` method. Ignored if a steady-state solution is provided
        verbose: bool, default: True
            Flag indicating whether to print the linearization results to the terminal.
        parameter_updates: dict
            New parameter values at which to linearize the model. Unspecified values will be taken from the initial
            values set in the GCN file.

            .. warning::

                If a steady state is provided, these values will *not* be used to update that solution! This can lead
                to an inconsistent linearization. The user is responsible for ensuring consistency in this case.

        Returns
        -------
        A: np.ndarray
            Jacobian matrix of the model with respect to :math:`x_{t+1}` evaluated at the steady state, right-multiplied
            by the diagonal matrix :math:`T`.
        B: np.ndarray
            Jacobian matrix of the model with respect to :math:`x_t` evaluated at the steady state, right-multiplied
            by the diagonal matrix :math:`T`.
        C: np.ndarray
            Jacobian matrix of the model with respect to :math:`x_{t-1}` evaluated at the steady state, right-multiplied
            by the diagonal matrix :math:`T`.
        D: np.ndarray
            Jacobian matrix of the model with respect to :math:`\varepsilon_t` evaluated at the steady state.

        Examples
        --------
        Given a DSGE model of the form:

        .. math::

            F(x_{t+1}, x_t, x_{t-1}, \varepsilon_t) = 0

        The "solution" to the model would be a policy function :math:`g(x_t, \varepsilon_t)`, such that:

        .. math::

            x_{t+1} = g(x_t, \varepsilon_t)

        With the exception of toy models, this policy function is not available in closed form. Instead, the model is
        linearized around the deterministic steady state, which is a fixed point in the system of equations. The linear
        approximation to the model is then used to approximate the policy function. Let :math:`\bar{x}` denote the
        deterministic steady state such that:

        .. math::

            F(\bar{x}, \bar{x}, \bar{x}, 0) = 0.

        A first-order Taylor expansion about (:math:`\bar{x}`, :math:`\bar{x}`, :math:`\bar{x}`, 0) yields

        .. math::

            A (x_{t+1} - \bar{x}) + B (x_t - \bar{x}) + C (x_{t-1} - \bar{x}) + D \varepsilon_t = 0,

        where the Jacobian matrices evaluated at the steady state are

        .. math::

            A = \left .\ frac{\partial F}{\partial x_{t+1}} \right |_{(\bar{x},\bar{x},\bar{x},0)}, \quad
            B = \left .\ frac{\partial F}{\partial x_t} \right |_{(\bar{x},\bar{x},\bar{x},0)}, \quad
            C = \left .\ frac{\partial F}{\partial x_{t-1}} \right|_{(\bar{x},\bar{x},\bar{x},0)}, \quad
            D = \left .\ frac{\partial F}{\partial \varepsilon_t} \right|_{(\bar{x},\bar{x},\bar{x},0)}

        It is common to perform a change of variables to log-linearize the model. Define a log-state vector,
        :math:`\tilde{x}_t = \log(x_t)`, with steady state :math:`\tilde{x}_{ss} = \log(\bar{x})`. We get back to the
        original variables by exponentiating the log-state vector.

        .. math::

            F(\exp(\tilde{x}_{t+1}), \exp(\tilde{x}_t), \exp(\tilde{x}_{t-1}), \varepsilon_t) = 0

        Taking derivaties with respect to :math:`\tilde{x}_t`, the linearized model is then:

        .. math::
            :nowrap:

            \[
            A \exp(\tilde{x}_{ss}) (\tilde{x}_{t+1} - \tilde{x}_{ss}) + B \exp(\tilde{x}_{ss}) (\tilde{x}_t -
            \tilde{x}_{ss}) + C \exp(\tilde{x}_{ss}) (\tilde{x}_{t-1} - \tilde{x}_{ss}) + D \varepsilon_t = 0
            \]

        Note that :math:`\tilde{x} - \tilde{x}_{ss} = \log(x - \bar{x}) = \log \left (\frac{x}{\bar{x}} \right )` is
        the approximate percent deviation of the variable from its steady state.

        The above derivation holds on a variable-by-variable basis. Some variables can be logged and others left in
        levels, all that is required is right-multiplication by a diagonal matrix of the form:

        .. math::

            T = \text{Diagonal}(\{h(x_1), h(x_2), \ldots, h(x_n)\})

        Where :math:`h(x_i) = 1` if the variable is left in levels, and :math:`h(x_i) = \exp(\tilde{x}_{ss})` if the
        variable is logged. This function returns the matrices :math:`AT`, :math:`BT`, :math:`CT`, and :math:`D`.
        """
        if order != 1:
            raise NotImplementedError("Only first order linearization is currently supported.")
        if steady_state_kwargs is None:
            steady_state_kwargs = {}
        if verbose not in steady_state_kwargs:
            steady_state_kwargs["verbose"] = verbose

        if self.is_linear:
            # If the model is linear, the linearization is already done; don't do it again
            log_linearize = False

        param_dict = self.parameters(**parameter_updates)

        if steady_state is None:
            if self.is_linear:
                steady_state = self.f_ss(**self.parameters(**param_dict))
            else:
                steady_state = self.steady_state(
                    **self.parameters(**param_dict),
                    **steady_state_kwargs,
                )

        jacobians, ss_input_nodes, param_input_nodes = self.symbolic_linearization(
            order=order,
            log_linearize=log_linearize,
            not_loglin_variables=not_loglin_variables,
            steady_state=steady_state,
            loglin_negative_ss=loglin_negative_ss,
            verbose=verbose,
        )

        # Cache the compiled function. Since symbolic_linearization returns the same graph objects on cache hit,
        # the id of the first jacobian node is a stable key.
        cache_key = id(jacobians[0])

        if not hasattr(self, "_linearize_cache"):
            self._linearize_cache = {}

        if cache_key not in self._linearize_cache:
            all_inputs = list(ss_input_nodes) + list(param_input_nodes)
            f = compile_pytensor_function(all_inputs, jacobians, on_unused_input="ignore")
            self._linearize_cache[cache_key] = f
        else:
            f = self._linearize_cache[cache_key]

        # Build input values: steady-state variables first, then parameters in graph order
        ss_values = {k.removesuffix("_ss"): v for k, v in steady_state.items()}
        ss_vals = [ss_values[v.base_name] for v in self.variables]
        param_vals = [param_dict[n.name] for n in param_input_nodes]

        A, B, C, D = f(*ss_vals, *param_vals)

        return [np.ascontiguousarray(x, dtype=A.dtype) for x in [A, B, C, D]]

    def _solve_with_gensys(
        self,
        A,
        B,
        C,
        D,
        n_variables: int,
        tol: float,
        verbose: bool,
        on_failure: str,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        gensys_results = solve_policy_function_with_gensys(A, B, C, D, tol)
        G_1, _constant, impact, _f_mat, _f_wt, _y_wt, _gev, eu, _loose = gensys_results

        success = all(x == 1 for x in eu[:2])
        if not success:
            if on_failure == "error":
                raise GensysFailedException(eu)
            if verbose:
                _log.info(interpret_gensys_output(eu))
            return None, None

        if verbose:
            _log.info(interpret_gensys_output(eu))

        T = G_1[:n_variables, :][:, :n_variables]
        R = impact[:n_variables, :]
        return T, R

    def _solve_with_cycle_reduction(
        self,
        A,
        B,
        C,
        D,
        max_iter: int,
        tol: float,
        verbose: bool,
        on_failure: str,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        T, R, result, _log_norm = solve_policy_function_with_cycle_reduction(A, B, C, D, max_iter, tol, verbose)
        if T is None:
            if on_failure == "error":
                raise GensysFailedException(result)
            if verbose:
                _log.info(result)
            return None, None
        return T, R

    def solve_model(
        self,
        solver="cycle_reduction",
        log_linearize: bool = True,
        not_loglin_variables: list[str] | None = None,
        order: Literal[1] = 1,
        loglin_negative_ss: bool = False,
        steady_state: dict | None = None,
        steady_state_kwargs: dict | None = None,
        tol: float = 1e-8,
        max_iter: int = 1000,
        verbose: bool = True,
        on_failure="error",
        **parameter_updates,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        r"""
        Solve for the linear approximation to the policy function via perturbation.

        Parameters
        ----------
        solver: str, default: 'cycle_reduction'
            Name of the algorithm to solve the linear solution. Currently "cycle_reduction", "gensys", and
            "backward_direct" are supported. Following Dynare, cycle_reduction is the default, but note that gEcon uses
            gensys.
        log_linearize: bool, default: True
            Whether to log-linearize the model. If False, the model will be solved in levels.
        not_loglin_variables: list of strings, optional
            Variables to not log linearize when solving the model. Variables with steady state values close to zero
            (or negative) will be automatically selected to not log linearize. Ignored if log_linearize is False.
        order: int, default: 1
            Order of taylor expansion to use to solve the model. Currently only 1st order approximation is supported.
        steady_state: dict, optional
            Dictionary of steady-state solutions. If not provided, the steady state will be solved for using the
            ``steady_state`` method.
        steady_state_kwargs: dict, optional
            Keyword arguments passed to the `steady_state` method. Ignored if a steady-state solution is provided
            via the steady_state argument, Default is None.
        loglin_negative_ss: bool, default is False
            Whether to force log-linearization of variable with negative steady-state. This is impossible in principle
            (how can :math:`exp(x_ss)` be negative?), but can still be done; see the docstring for
            :func:`perturbation.linearize_model` for details. Use with caution, as results will not correct. Ignored if
            log_linearize is False.
        tol: float, default 1e-8
            Desired level of floating point accuracy in the solution
        max_iter: int, default: 1000
            Maximum number of cycle_reduction iterations. Not used if solver is 'gensys'.
        verbose: bool, default: True
            Flag indicating whether to print solver results to the terminal
        on_failure: str, one of ['error', 'ignore'], default: 'error'
            Instructions on what to do if the algorithm to find a linearized policy matrix. "Error" will raise an error,
            while "ignore" will return None. "ignore" is useful when repeatedly solving the model, e.g. when sampling.
        parameter_updates: dict
            New parameter values at which to solve the model. Unspecified values will be taken from the initial values
            set in the GCN file.

        Returns
        -------
        T: np.ndarray, optional
            Transition matrix, approximated to the requested order. Represents the policy function, governing agent's
            optimal state-conditional actions. If the solver fails, None is returned instead.

        R: np.ndarray, optional
            Selection matrix, approximated to the requested order. Represents the state- and agent-conditional
            transmission of stochastic shocks through the economy. If the solver fails, None is returned instead.

        Examples
        --------
        This method solves the model by linearizing it around the deterministic steady state, and then solving for the
        policy function using a perturbation method. We begin with a model defined as a function of the form:

        .. math::
           :nowrap:

           \[
           \mathbb{E} \left [ F(x_{t+1}, x_t, x_{t-1}, \varepsilon_t) \right ] = 0
           \]

        The linear approximation is then given by the matrices :math:`A`, :math:`B`, :math:`C`, and :math:`D`, as:

        .. math::
           :nowrap:

           \[
           A \hat{x}_{t+1} + B \hat{x}_t + C \hat{x}_{t-1} + D \varepsilon_t = 0
           \]

        where :math:`\hat{x}_t = x_t - \bar{x}` is the deviation of the state vector from its steady state (again,
        potentially in logs). A solution to the model seeks a function:

        .. math::
           :nowrap:

           \[
           x_t = g(x_{t-1}, \varepsilon_t)
           \]

        This implies that :math:`x_{t+1} = g(x_t, \varepsilon_{t+1})`, allowing us to write the model as:

        .. math::
           :nowrap:

           \[
           F_g(x_{t-1}, \varepsilon_t, \varepsilon_{t+1}) =
           f(g(g(x_{t-1}, \varepsilon_t), \varepsilon_{t+1}),
             g(x_{t-1}, \varepsilon_t), x_{t-1}, \varepsilon_t) = 0
           \]

        To lighten notation, define:

        .. math::
           :nowrap:

           \[
           u = \varepsilon_t, \quad
           u_+ = \varepsilon_{t+1}, \quad
           \hat{x} = x_{t-1} - \bar{x} \\
           f_{x_+} = \left. \frac{\partial F_g}{\partial x_{t+1}} \right |_{\bar{x}, \bar{x}, \bar{x}, 0}, \quad
           f_x = \left. \frac{\partial F_g}{\partial x_t}  \right |_{\bar{x}, \bar{x}, \bar{x}, 0}, \\
           f_{x_-} = \left. \frac{\partial F_g}{\partial x_{t-1}}  \right |_{\bar{x}, \bar{x}, \bar{x}, 0}, \quad
           f_u = \left. \frac{\partial F_g}{\partial u}  \right |_{\bar{x}, \bar{x}, \bar{x}, 0} \\
           g_x = \left. \frac{\partial g}{\partial x_{t-1}}  \right |_{\bar{x}, \bar{x}, \bar{x}, 0}, \quad
           g_u = \left. \frac{\partial g}{\partial \varepsilon_t}  \right |_{\bar{x}, \bar{x}, \bar{x}, 0}
           \]

        Under this new notation, the system is:

        .. math::
           :nowrap:

           \[
           F_g(x_-, u, u_+) = f(g(g(x_-, u), u_+), g(x_, u), x_-, u) = 0
           \]

        The function :math:`g` is unknown, but is implicitly defined by this expression, and can be approximated by a
        first order Taylor expansion around the steady state. The linearized system is then:

        .. math::
           :nowrap:

           \[
           0 \approx F_g(x_-, u, u_+) =
           f_{x_+} (g_x (g_x \hat{x} + g_u u) + g_u u_+) +
           f_x (g_x \hat{x} + g_u u) +
           f_{x_-} \hat{x} + f_u u
           \]

        The Jacobian matrices :math:`f_{x_+}`, :math:`f_x`, :math:`f_{x_-}`, and :math:`f_u` are the matrices :math:`A`,
        :math:`B`, :math:`C`, and :math:`D` respectively, evaluated at the steady state, and are thus known. The task
        is then to solve for unknown matrices :math:`g_x` and :math:`g_u`, which will give a linear approximation to the
        optimal policy function.

        Take expectations, and impose that :math:`\mathbb{E}_t[u_+] = 0`:

        .. math::
           :nowrap:

           \begin{align}
           0 \approx {} &
           f_{x_+} (g_x(g_x \hat{x} + g_u u) + g_u \mathbb{E}_t[u_+]) +
           f_x (g_x \hat{x} + g_u u) + f_{x_-} \hat{x} + f_u u \\
           \approx {} &
           (f_{x_+} g_x g_x + f_x g_x + f_{x_-})\hat{x} +
           (f_{x_+} g_x g_u + f_x g_u + f_u) u
           \end{align}

        For the system to be equal to zero, both coefficient matrices must be zero, which gives us two linear equations
        in the unknowns :math:`g_x` and :math:`g_u`:

        .. math::
           :nowrap:

           \begin{align}
           (f_{x_+} g_x g_x + f_x g_x + f_{x_-}) \hat{x} &= 0 \\
           (f_{x_+} g_x g_u + f_x g_u + f_u) u &= 0
           \end{align}

        Assuming :math:`g_x` has been solved for, the coefficient in the second equation can be directly solved for,
        giving:

        .. math::
           :nowrap:

           \[
           g_u = -(f_{x_+} g_x + f_x)^{-1} f_u = 0
           \]

        The first equation, on the other hand, is a quadratic in :math:`g_x`, and cannot be solved for directly.
        Instead, we employ trickery. Then the equation can be re-written as a linear system in two states:

        .. math::
           :nowrap:

           \begin{align}
           \begin{bmatrix} 0 & f_{x_+} \\ I & 0 \end{bmatrix}
           \begin{bmatrix} g_x g_x \\ g_x \end{bmatrix} \hat{x}
           &=
           \begin{bmatrix} -f_x & -f_{x_-} \\ I & 0 \end{bmatrix}
           \begin{bmatrix} g_x \\ I \end{bmatrix} \hat{x} \\
           D \begin{bmatrix} I \\ g_x \end{bmatrix} g_x \hat{x}
           &=
           E \begin{matrix} g_x \\ I \end{matrix} \hat{x} \\
           QTZ \begin{bmatrix} I \\ g_x \end{bmatrix} g_x \hat{x}
           &=
           QSZ \begin{bmatrix} g_x \\ I \end{bmatrix} \hat{x} \\
           TZ \begin{bmatrix} I \\ g_x \end{bmatrix} g_x \hat{x}
           &=
           SZ \begin{bmatrix} g_x \\ I \end{bmatrix} \hat{x}
           \end{align}

        The last two lines use the QZ decomposition of the pencil :math:`<D, E>` into upper triangular matrix :math:`T`
        and quasi-upper triangular matrix :math:`S`, and the orthogonal matrices :math:`Z` and :math:`Q`. :math:`T` and
        :math:`S` have structure that can be exploited. In particular, they are arranged so that the eigenvalues of the
        pencil :math:`<D, E>` are sorted in modulus from smallest (stable) to largest (unstable).

        Partitioning the rows of the matrices by eign-stability, and the columns by the size of :math:`g_x`, we get:

        .. math::
           :nowrap:

           \[
           \begin{bmatrix} T_{11} & T_{12} \\ 0 & T_{22} \end{bmatrix}
           \begin{bmatrix} Z_{11} & Z_{12} \\ Z_{21} & Z_{22} \end{bmatrix}
           \begin{bmatrix} I \\ g_x \end{bmatrix} g_x \hat{x} =
           \begin{bmatrix} S_{11} & S_{12} \\ 0 & S_{22} \end{bmatrix}
           \begin{bmatrix} Z_{11} & Z_{12} \\ Z_{21} & Z_{22} \end{bmatrix}
           \begin{bmatrix} g_x \\ I \end{bmatrix} \hat{x}
           \]

        For the system to the stable, we require that:

        .. math::
           :nowrap:

           \[
           Z_{21} + Z_{22} g_x = 0
           \]

        And thus:

        .. math::
           :nowrap:

           \[
           g_x = -Z_{22}^{-1} Z_{21}
           \]

        This requires that -Z_{22} is square and invertible, which are known as the *rank* and *stability* conditions of
        Blanchard and Kahn (1980). If these conditions are not met, the model is indeterminate, and a solution is not
        possible.

        """
        if on_failure not in ["error", "ignore"]:
            raise ValueError(f'Parameter on_failure must be one of "error" or "ignore", found {on_failure}')
        if steady_state_kwargs is None:
            steady_state_kwargs = {}

        ss_dict = _maybe_solve_steady_state(self, steady_state, steady_state_kwargs, parameter_updates)
        n_variables = len(self.variables)

        A, B, C, D = self.linearize_model(
            order=order,
            log_linearize=log_linearize,
            not_loglin_variables=not_loglin_variables,
            steady_state=ss_dict.to_string(),
            loglin_negative_ss=loglin_negative_ss,
            verbose=verbose,
            **parameter_updates,
        )

        assert all(x.flags["C_CONTIGUOUS"] for x in [A, B, C, D])

        if self._backward_looking:
            solver = "backward_direct"

        if solver == "gensys":
            T, R = self._solve_with_gensys(A, B, C, D, n_variables, tol, verbose, on_failure)
        elif solver == "cycle_reduction":
            T, R = self._solve_with_cycle_reduction(A, B, C, D, max_iter, tol, verbose, on_failure)
        elif solver == "backward_direct":
            if not self._backward_looking:
                raise ValueError(
                    "Solver 'backward_direct' can only be used for models with no forward-looking variables."
                )
            T, R = solve_policy_function_with_backward_direct(A, B, C, D)
        else:
            raise NotImplementedError(
                'Only "cycle_reduction", "gensys", and "backward_direct" are valid values for solver'
            )

        if T is None:
            return None, None

        if verbose:
            check_perturbation_solution(A, B, C, D, T, R, tol=tol)

        return np.ascontiguousarray(T), np.ascontiguousarray(R)
