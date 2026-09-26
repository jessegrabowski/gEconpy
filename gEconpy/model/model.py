import difflib
import logging

from collections.abc import Callable, Sequence
from copy import deepcopy
from typing import Literal, NamedTuple

import numpy as np
import pytensor
import sympy as sp

from better_optimize import minimize, root
from preliz.distributions.distributions import Distribution
from pymc.distributions.transforms import Interval, Transform, log, logodds
from pytensor import tensor as pt
from pytensor.graph.replace import clone_replace
from pytensor.graph.traversal import explicit_graph_inputs
from pytensor.tensor.variable import TensorVariable
from scipy.optimize import OptimizeResult

from gEconpy.classes.containers import SteadyStateResults, SymbolDictionary
from gEconpy.classes.distributions import CompositeDistribution
from gEconpy.classes.time_aware_symbol import TimeAwareSymbol
from gEconpy.exceptions import GensysFailedException, ModelUnknownParameterError
from gEconpy.model.compile import compile_for_scipy, make_cache_key, pack_and_compile
from gEconpy.model.parameters import compile_param_dict_func
from gEconpy.model.perturbation import check_perturbation_solution, make_not_loglin_flags
from gEconpy.model.perturbation import linearize_model as _linearize_model
from gEconpy.model.statistics.validation import _maybe_solve_steady_state
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
from gEconpy.solvers.gensys import interpret_gensys_output, solve_policy_function_with_gensys
from gEconpy.utilities import get_name, postprocess_optimizer_res, safe_to_ss

_log = logging.getLogger(__name__)

STEADY_STATE_TOL = 1e-8
BOX_BOUND_METHODS = ("trust-constr", "L-BFGS-B", "powell")
GRADIENT_REQUIRED_METHODS = ("trust-ncg", "trust-krylov", "trust-exact", "dogleg", "newton-cg")


def infer_variable_bounds(variable: TimeAwareSymbol | sp.Symbol) -> tuple[float | None, float | None]:
    """
    Read a variable's sign assumptions and return a bound that keeps it strictly signed.

    Parameters
    ----------
    variable : TimeAwareSymbol or Symbol
        Variable whose ``assumptions0`` are inspected.

    Returns
    -------
    lower : float or None
        Lower bound, 1e-8 for a positive variable and None otherwise.
    upper : float or None
        Upper bound, -1e-8 for a negative variable and None otherwise.
    """
    assumptions = variable.assumptions0
    lower = 1e-8 if assumptions.get("positive", False) else None
    upper = -1e-8 if assumptions.get("negative", False) else None

    return lower, upper


def infer_variable_transform(
    variable: TimeAwareSymbol | sp.Symbol,
    user_bound: tuple[float | None, float | None] | None = None,
) -> Transform | None:
    """
    Pick a bijection from the real line onto a variable's feasible region for unconstrained solving.

    The feasible region comes from ``user_bound`` when given, then from the variable's GCN-declared sympy
    assumptions, and is otherwise the whole real line. The returned transform's ``backward`` maps an unconstrained
    real input onto the region: ``log`` for a positive variable, ``logodds`` for the unit interval, and ``Interval``
    for general bounds.

    Parameters
    ----------
    variable : TimeAwareSymbol or Symbol
        The steady-state variable. Its ``assumptions0`` supply the fallback region.
    user_bound : tuple of float or None, optional
        Explicit ``(lower, upper)`` bound. Either side may be None for a one-sided bound. Takes precedence over
        assumptions. Default None.

    Returns
    -------
    transform : Transform or None
        The bijection, or None for an unconstrained variable.
    """
    if user_bound is not None and user_bound != (None, None):
        lower, upper = user_bound
        return Interval(lower, upper)

    assumptions = variable.assumptions0
    if assumptions.get("unit_interval"):
        return logodds
    if assumptions.get("positive"):
        return log
    if assumptions.get("negative"):
        return Interval(None, 0.0)
    return None


def transform_steady_state_system(
    equations: Sequence[TensorVariable],
    ss_nodes: Sequence[TensorVariable],
    transforms: Sequence[Transform | None],
) -> tuple[
    list[TensorVariable], list[TensorVariable], Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray]
]:
    """
    Reparametrize a steady-state system onto unconstrained variables.

    For each steady-state node ``x`` with transform ``t``, substitute ``x -> t.backward(y)`` (identity for an
    unconstrained variable), so the returned equations are functions of fresh unconstrained inputs ``y``.

    Parameters
    ----------
    equations : list of TensorVariable
        Scalar equation graphs, each zero at the steady state.
    ss_nodes : list of TensorVariable
        Scalar input node for each steady-state variable.
    transforms : list of Transform or None
        One transform per variable, as returned by :func:`infer_variable_transform`.

    Returns
    -------
    transformed_equations : list of TensorVariable
        The equations rewritten in terms of the unconstrained inputs.
    y_nodes : list of TensorVariable
        The unconstrained input nodes, one per variable.
    to_unconstrained : callable
        Map a constrained point ``x`` to its unconstrained image ``y``.
    to_constrained : callable
        Map an unconstrained point ``y`` back to the constrained ``x``.
    """
    y_nodes, x_nodes, backward_exprs, forward_exprs, replace = [], [], [], [], {}
    for ss_node, transform in zip(ss_nodes, transforms, strict=True):
        y, x = ss_node.type(), ss_node.type()
        y.name = x.name = ss_node.name
        y_nodes.append(y)
        x_nodes.append(x)

        x_of_y = y if transform is None else transform.backward(y)
        backward_exprs.append(x_of_y)
        forward_exprs.append(x if transform is None else transform.forward(x))
        replace[ss_node] = x_of_y

    transformed_equations = clone_replace(list(equations), replace=replace)
    f_to_x = pytensor.function(y_nodes, backward_exprs, on_unused_input="ignore")
    f_to_y = pytensor.function(x_nodes, forward_exprs, on_unused_input="ignore")

    def to_constrained(y: np.ndarray) -> np.ndarray:
        return np.asarray(f_to_x(*np.asarray(y, dtype=float)), dtype=float)

    def to_unconstrained(x: np.ndarray) -> np.ndarray:
        return np.asarray(f_to_y(*np.asarray(x, dtype=float)), dtype=float)

    return transformed_equations, y_nodes, to_unconstrained, to_constrained


class DROrder(NamedTuple):
    """
    Decision-rule ordering of variables and equations.

    Variables are partitioned into static, predetermined-only, mixed, and forward-only groups by their time-shift
    profile across all equations. Equations are partitioned into static, lag-only, lead-only, and both groups by the
    time shifts they reference. Reordering A, B, C and D by ``var_order`` on the columns and ``eq_order`` on the rows
    places their structural-zero blocks contiguously.

    Apply ``inv_var_order`` to the rows and columns of T and to the rows of R before handing results back to the
    Kalman path or the user, so the user-visible state vector layout is preserved. Equation reordering applies only
    to the rows of A, B, C and D, never to T or R.

    Attributes
    ----------
    var_order : ndarray
        Variable column permutation into static, predetermined-only, mixed, and forward-only order.
    inv_var_order : ndarray
        Inverse of ``var_order``, undoing it on the rows and columns of T and the rows of R.
    eq_order : ndarray
        Equation row permutation into static, lag-only, lead-only, and both order.
    inv_eq_order : ndarray
        Inverse of ``eq_order``, undoing it on the rows of A, B, C and D.
    n_static_var : int
        Number of static variables.
    n_pred_only_var : int
        Number of predetermined-only variables.
    n_mixed_var : int
        Number of variables appearing at both a lag and a lead.
    n_forward_only_var : int
        Number of forward-only variables.
    n_static_eq : int
        Number of equations referencing no time shifts.
    n_lag_only_eq : int
        Number of equations referencing lags but no leads.
    n_lead_only_eq : int
        Number of equations referencing leads but no lags.
    n_both_eq : int
        Number of equations referencing both lags and leads.
    """

    var_order: np.ndarray
    inv_var_order: np.ndarray
    eq_order: np.ndarray
    inv_eq_order: np.ndarray
    n_static_var: int
    n_pred_only_var: int
    n_mixed_var: int
    n_forward_only_var: int
    n_static_eq: int
    n_lag_only_eq: int
    n_lead_only_eq: int
    n_both_eq: int

    @classmethod
    def from_model(cls, variables: list[TimeAwareSymbol], equations: list[sp.Expr]) -> "DROrder":
        """
        Classify variables and equations by the time shifts at which the variables appear.

        Parameters
        ----------
        variables : list of TimeAwareSymbol
            Model variables at time t, in model order.
        equations : list of Expr
            Model equations as sympy expressions.

        Returns
        -------
        order : DROrder
            Permutations and group sizes for the decision-rule ordering.
        """
        lag_index = {variable.set_t(-1): j for j, variable in enumerate(variables)}
        lead_index = {variable.set_t(1): j for j, variable in enumerate(variables)}

        var_has_lag = np.zeros(len(variables), dtype=bool)
        var_has_lead = np.zeros(len(variables), dtype=bool)
        eq_has_lag = np.zeros(len(equations), dtype=bool)
        eq_has_lead = np.zeros(len(equations), dtype=bool)

        for i, equation in enumerate(equations):
            atoms = equation.atoms(TimeAwareSymbol)
            lagged = [lag_index[atom] for atom in atoms if atom in lag_index]
            led = [lead_index[atom] for atom in atoms if atom in lead_index]

            var_has_lag[lagged] = True
            var_has_lead[led] = True
            eq_has_lag[i] = bool(lagged)
            eq_has_lead[i] = bool(led)

        static_vars = np.flatnonzero(~var_has_lag & ~var_has_lead)
        pred_only_vars = np.flatnonzero(var_has_lag & ~var_has_lead)
        mixed_vars = np.flatnonzero(var_has_lag & var_has_lead)
        forward_only_vars = np.flatnonzero(~var_has_lag & var_has_lead)
        var_order = np.concatenate([static_vars, pred_only_vars, mixed_vars, forward_only_vars])

        static_eqs = np.flatnonzero(~eq_has_lag & ~eq_has_lead)
        lag_only_eqs = np.flatnonzero(eq_has_lag & ~eq_has_lead)
        lead_only_eqs = np.flatnonzero(~eq_has_lag & eq_has_lead)
        both_eqs = np.flatnonzero(eq_has_lag & eq_has_lead)
        eq_order = np.concatenate([static_eqs, lag_only_eqs, lead_only_eqs, both_eqs])

        return cls(
            var_order=var_order,
            inv_var_order=np.argsort(var_order),
            eq_order=eq_order,
            inv_eq_order=np.argsort(eq_order),
            n_static_var=len(static_vars),
            n_pred_only_var=len(pred_only_vars),
            n_mixed_var=len(mixed_vars),
            n_forward_only_var=len(forward_only_vars),
            n_static_eq=len(static_eqs),
            n_lag_only_eq=len(lag_only_eqs),
            n_lead_only_eq=len(lead_only_eqs),
            n_both_eq=len(both_eqs),
        )


class Model:
    """
    A Dynamic Stochastic General Equilibrium (DSGE) model.

    Stores the model primitives (variables, parameters, shocks, equations) as sympy objects and builds the pytensor
    graphs and compiled functions for steady-state solving and linearization on first use. Build one with
    :func:`~gEconpy.model.build.model_from_gcn`.

    Parameters
    ----------
    variables : list of TimeAwareSymbol
        Model variables.
    shocks : list of TimeAwareSymbol
        Exogenous shocks.
    equations : list of Expr
        Model equations.
    steady_state_relationships : list of Eq
        Analytical steady-state relationships.
    steady_state_equations : list of Expr
        Steady-state equations in residual form, each equal to zero at the steady state.
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
        Prior distribution dictionaries ``(param_priors, shock_priors)``.
    is_linear : bool, optional
        Whether the model equations are already linear, so the steady state is zero and linearization is the
        identity. Default False.
    mode : str, optional
        Pytensor compilation mode, such as ``'FAST_COMPILE'`` or ``'FAST_RUN'``. Default None uses the pytensor
        default.
    error_func : str, optional
        Error metric for minimize-based steady-state solving. Default ``'squared'``.
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

        self._steady_state_equations = steady_state_equations
        self._ss_solution_dict = ss_solution_dict
        self._param_dict = param_dict
        self._deterministic_dict = deterministic_dict
        self._calib_dict = calib_dict
        self._mode = mode
        self._error_func: ERROR_FUNCTIONS = error_func

        self._cache: dict | None = None
        self._f_params: Callable | None = None
        self._f_ss: Callable | None = None
        self._equation_tensors: list[TensorVariable] | None = None
        self._full_equation_tensors: list[TensorVariable] | None = None
        self._f_full_residual: Callable | None = None

        self._backward_variables: list[TimeAwareSymbol] | None = None
        self._symbolic_forward_variables: list[TimeAwareSymbol] | None = None
        self._lead_var_idx: np.ndarray | None = None
        self._dr_order: DROrder | None = None
        self._symbolic_linearize_cache: dict[frozenset[str], tuple] = {}
        self._linearize_cache: dict[int, Callable] = {}

    @property
    def variables(self) -> list[TimeAwareSymbol]:
        """
        Endogenous variables of the model, as :class:`~gEconpy.classes.time_aware_symbol.TimeAwareSymbol` objects.

        A time subscript identifies a variable as endogenous.
        """
        return self._variables

    @property
    def shocks(self) -> list[TimeAwareSymbol]:
        """
        Exogenous shocks of the model, as :class:`~gEconpy.classes.time_aware_symbol.TimeAwareSymbol` objects.

        Shocks are the source of stochasticity in the model.
        """
        return self._shocks

    @property
    def equations(self) -> list[sp.Expr]:
        """Model equations as sympy expressions, each equal to zero."""
        return self._equations

    @property
    def params(self) -> list[sp.Symbol]:
        """
        Free parameters of the model, as :class:`~sympy.core.symbol.Symbol` objects.

        Parameters are fixed values in the structural equations. They are sometimes called "deep parameters" because
        of their (supposed) microeconomic foundations.
        """
        return self._params

    @property
    def hyper_params(self) -> list[sp.Symbol]:
        """
        Hyperparameters of the model, as :class:`~sympy.core.symbol.Symbol` objects.

        Hyperparameters describe the distribution of the shocks, for example the standard deviation of a normally
        distributed shock.
        """
        return self._hyper_params

    @property
    def deterministic_params(self) -> list[sp.Symbol]:
        """
        Deterministic parameters of the model, as :class:`~sympy.core.symbol.Symbol` objects.

        Deterministic parameters are defined as functions of other parameters and are derived from them at every
        parameter update.
        """
        return self._deterministic_params

    @property
    def calibrated_params(self) -> list[sp.Symbol]:
        """
        Calibrated parameters of the model, as :class:`~sympy.core.symbol.Symbol` objects.

        A calibrated parameter is pinned by a calibration equation in steady-state variables. That equation joins the
        steady-state system, and the parameter is solved for alongside the steady-state variables.
        """
        return self._calibrated_params

    @property
    def param_priors(self) -> dict[str, Distribution]:
        """Prior distributions of the model parameters, keyed by parameter name."""
        return self._priors[0]

    @property
    def shock_priors(self) -> dict[str, CompositeDistribution]:
        """Prior distributions of the model shocks, keyed by shock name."""
        return self._priors[1]

    @property
    def steady_state_relationships(self) -> list[sp.Eq]:
        """Model equations evaluated at the deterministic steady state."""
        return self._steady_state_relationships

    @property
    def n_variables(self) -> int:
        """Number of endogenous variables in the model."""
        return len(self._variables)

    @property
    def backward_variables(self) -> list[TimeAwareSymbol]:
        """Variables that appear at t-1 in at least one equation, the state variables."""
        if self._backward_variables is None:
            lagged = {a.set_t(0) for eq in self._equations for a in eq.atoms(TimeAwareSymbol) if a.time_index == -1}
            self._backward_variables = [v for v in self._variables if v in lagged]
        return self._backward_variables

    @property
    def symbolic_forward_variables(self) -> list[TimeAwareSymbol]:
        """
        Variables that appear at t+1 in at least one equation.

        The linearization can still assign an all-zero column in the lead Jacobian to such a variable, when every
        coefficient on its lead happens to vanish at the current parameters.
        """
        if self._symbolic_forward_variables is None:
            leads = {a.set_t(0) for eq in self._equations for a in eq.atoms(TimeAwareSymbol) if a.time_index == 1}
            self._symbolic_forward_variables = [v for v in self._variables if v in leads]
        return self._symbolic_forward_variables

    @property
    def forward_variables(self) -> list[TimeAwareSymbol]:
        """
        Forward-looking (jump) variables, the set Blanchard-Kahn counting uses.

        Identical to :attr:`symbolic_forward_variables`. The Blanchard-Kahn condition requires the number of
        unstable eigenvalues to equal the length of this list.
        """
        # Measuring this from the lead Jacobian instead would tie the set to the parameters it was measured at,
        # and it is reused at every other parameter vector. Over-counting is safe: an all-zero lead column
        # contributes an infinite generalized eigenvalue, adding one to both sides of the comparison.
        return self.symbolic_forward_variables

    @property
    def n_backward(self) -> int:
        """Number of backward-looking (state) variables."""
        return len(self.backward_variables)

    @property
    def n_forward(self) -> int:
        """Number of forward-looking (jump) variables, the length of :attr:`forward_variables`."""
        return len(self.forward_variables)

    @property
    def n_symbolic_forward(self) -> int:
        """Number of variables appearing at t+1 anywhere in the model equations."""
        return len(self.symbolic_forward_variables)

    @property
    def lead_var_idx(self) -> np.ndarray:
        """Column indices of :attr:`forward_variables` in the Jacobian matrices."""
        if self._lead_var_idx is None:
            forward_set = set(self.forward_variables)
            self._lead_var_idx = np.array([i for i, v in enumerate(self._variables) if v in forward_set], dtype=int)
        return self._lead_var_idx

    @property
    def dr_order(self) -> DROrder:
        """Decision-rule ordering of variables and equations. See :class:`DROrder` for the layout."""
        if self._dr_order is None:
            self._dr_order = DROrder.from_model(self._variables, self._equations)
        return self._dr_order

    @property
    def var_order(self) -> np.ndarray:
        """Column permutation reordering variables as ``[static | pred_only | mixed | forward_only]``."""
        return self.dr_order.var_order

    @property
    def inv_var_order(self) -> np.ndarray:
        """Inverse of :attr:`var_order`. Apply to the rows and columns of T and the rows of R to undo it."""
        return self.dr_order.inv_var_order

    @property
    def eq_order(self) -> np.ndarray:
        """Row permutation reordering equations as ``[static | lag_only | lead_only | both]``."""
        return self.dr_order.eq_order

    @property
    def inv_eq_order(self) -> np.ndarray:
        """Inverse of :attr:`eq_order`. Apply to the rows of A, B, C and D to undo it."""
        return self.dr_order.inv_eq_order

    @property
    def f_params(self) -> Callable:
        """Compiled function mapping free parameter values to the full parameter dictionary."""
        if self._f_params is None:
            self._f_params, _ = compile_param_dict_func(self._param_dict, self._deterministic_dict, mode=self._mode)
        return self._f_params

    @property
    def f_ss(self) -> Callable | None:
        """Compiled function mapping parameters to the known steady-state values, or None without analytic solutions."""
        if self._f_ss is None:
            if not self._ss_solution_dict:
                return None

            _, cache = compile_param_dict_func(self._param_dict, self._deterministic_dict, mode=self._mode)
            all_params = list(self._param_dict.to_sympy().keys()) + list(self._deterministic_dict.to_sympy().keys())
            self._f_ss, _ = compile_known_ss(
                self._ss_solution_dict,
                self._variables,
                all_params,
                mode=self._mode,
                cache=cache,
            )
        return self._f_ss

    @property
    def sympy_to_pytensor_cache(self) -> dict:
        """
        Cache mapping sympy symbol identifiers to pytensor graph nodes.

        Every graph-building call shares this cache, so a model symbol always maps to the same pytensor node.
        """
        return self._ensure_cache()

    def ss_tensors(self, filter_known: bool = False) -> list[TensorVariable]:
        """
        Pytensor scalar variables for the model's steady-state symbols.

        Return one scalar ``TensorVariable`` per model variable and calibrated parameter, in model order. The
        variables live in the shared sympytensor cache, so repeated calls and downstream graph building share the
        same objects.

        Parameters
        ----------
        filter_known : bool, optional
            If True, exclude variables whose steady-state values are analytically known from the ``STEADY_STATE``
            block. Default False.

        Returns
        -------
        ss_nodes : list of TensorVariable
            One scalar node per steady-state symbol.

        Examples
        --------
        List the steady-state inputs that the equation graphs consume:

        .. code-block:: python

            from gEconpy import model_from_gcn
            from gEconpy.data import get_example_gcn

            model = model_from_gcn(get_example_gcn("RBC"), verbose=False)
            ss_nodes = model.ss_tensors()
            print([node.name for node in ss_nodes])
        """
        known_names = self._known_steady_state_names() if filter_known else set()
        return [self._scalar_node(symbol) for symbol in self._steady_state_symbols() if symbol.name not in known_names]

    def param_tensors(
        self,
        include_free: bool = True,
        include_deterministic: bool = True,
        include_calibrated: bool = False,
    ) -> list[TensorVariable]:
        """
        Pytensor scalar variables for the model's parameters.

        Parameters
        ----------
        include_free : bool, optional
            Include free parameters, those with numeric defaults. Default True.
        include_deterministic : bool, optional
            Include deterministic parameters, those defined as functions of other parameters. Default True.
        include_calibrated : bool, optional
            Include calibrated parameters, those pinned by steady-state equations. Default False.

        Returns
        -------
        param_nodes : list of TensorVariable
            One scalar node per selected parameter.

        Examples
        --------
        Collect the free parameter nodes to use as inputs of a hand-built graph:

        .. code-block:: python

            from gEconpy import model_from_gcn
            from gEconpy.data import get_example_gcn

            model = model_from_gcn(get_example_gcn("RBC"), verbose=False)
            param_nodes = model.param_tensors(include_deterministic=False)
            print([node.name for node in param_nodes])
        """
        param_symbols: list[sp.Symbol] = []
        if include_free:
            param_symbols += list(self._param_dict.to_sympy().keys())
        if include_deterministic:
            param_symbols += list(self._deterministic_dict.to_sympy().keys())
        if include_calibrated:
            param_symbols += list(self._calib_dict.to_sympy().keys())

        return [self._scalar_node(symbol) for symbol in param_symbols]

    def equation_tensors(self, filter_known: bool = False) -> list[TensorVariable]:
        """
        Pytensor graphs for the model's steady-state equations.

        Return one scalar ``TensorVariable`` per equation, each equal to zero at the steady state. The graph inputs
        are the nodes in the shared sympytensor cache, so they share identity with :meth:`ss_tensors` and
        :meth:`param_tensors`.

        Each call returns a fresh clone with shared input leaves. Compilation rewrites graphs in place, so handing
        out the memoized master graph would let one consumer corrupt it for the next.

        Parameters
        ----------
        filter_known : bool, optional
            If True, substitute the analytically known steady-state values and drop the equations they fully
            determine. Default False.

        Returns
        -------
        equations : list of TensorVariable
            One scalar residual graph per equation.

        Examples
        --------
        Compile the residual system into a function of the steady-state variables and parameters:

        .. code-block:: python

            import pytensor
            import pytensor.tensor as pt

            from gEconpy import model_from_gcn
            from gEconpy.data import get_example_gcn

            model = model_from_gcn(get_example_gcn("RBC"), verbose=False)
            inputs = model.ss_tensors() + model.param_tensors()
            residual = pt.stack(model.equation_tensors())
            f_residual = pytensor.function(inputs, residual, on_unused_input="ignore")

            steady_state = model.steady_state(verbose=False, progressbar=False)
            values = {**steady_state, **model.parameters()}
            print(f_residual(*[values[node.name] for node in inputs]))
        """
        master = self._ensure_equation_tensors(filter_known=filter_known)
        return clone_replace(master)

    def parameters(self, **updates: float) -> SymbolDictionary[str, float]:
        """
        Compute the full set of free and deterministic parameter values.

        Calibrated parameters are part of the steady-state solution and are not returned here. A parameter absent
        from ``updates`` takes the default value declared in the GCN file.

        Parameters
        ----------
        **updates : float
            New values for free parameters, keyed by parameter name. Deterministic parameter names are ignored, so
            the output of one call can be splatted into the next.

        Returns
        -------
        param_dict : SymbolDictionary
            Parameter values keyed by parameter name.

        Examples
        --------
        Change one parameter and keep the defaults for the rest:

        .. code-block:: python

            from gEconpy import model_from_gcn
            from gEconpy.data import get_example_gcn

            model = model_from_gcn(get_example_gcn("RBC"), verbose=False)
            params = model.parameters(beta=0.95)
            print(params["beta"], params["alpha"])
        """
        deterministic_names = {x.name for x in self.deterministic_params}
        updates = {k: v for k, v in updates.items() if k not in deterministic_names}

        param_dict = self._default_params.copy()
        unknown_updates = set(updates.keys()) - set(param_dict.keys())
        if unknown_updates:
            raise ModelUnknownParameterError(list(unknown_updates))
        param_dict.update(updates)

        return self.f_params(**param_dict).to_string()

    def get(self, name: str) -> sp.Symbol:
        """
        Get a model variable, shock, or parameter by name.

        Variables and shocks are returned as :class:`~gEconpy.classes.time_aware_symbol.TimeAwareSymbol` objects and
        parameters as plain sympy symbols. A name ending in ``_ss`` returns the steady-state form of the symbol.

        Parameters
        ----------
        name : str
            Name of the variable, shock, or parameter to retrieve.

        Returns
        -------
        symbol : Symbol
            The requested symbol.

        Examples
        --------
        Fetch a variable, its steady-state form, and a parameter:

        .. code-block:: python

            from gEconpy import model_from_gcn
            from gEconpy.data import get_example_gcn

            model = model_from_gcn(get_example_gcn("RBC"), verbose=False)
            print(model.get("K"), model.get("K_ss"), model.get("alpha"))
        """
        ss_requested = name.endswith("_ss")
        name = name.removesuffix("_ss")

        symbol = self._all_names_to_symbols.get(name)
        if symbol is None:
            close_matches = difflib.get_close_matches(name, list(self._all_names_to_symbols), n=1)
            hint = f" Did you mean {close_matches[0]}?" if close_matches else ""
            raise IndexError(f"Did not find {name} among model objects.{hint}")
        if ss_requested:
            return symbol.to_ss()
        return symbol

    def steady_state(
        self,
        how: Literal["analytic", "root", "minimize"] = "analytic",
        use_jac: bool = True,
        use_hess: bool = False,
        use_hessp: bool = True,
        progressbar: bool = True,
        optimizer_kwargs: dict | None = None,
        verbose: bool = True,
        bounds: dict[str, tuple[float, float]] | None = None,
        prefer_transform: bool = False,
        fixed_values: dict[str, float] | None = None,
        jitter_x0: bool = False,
        **updates: float,
    ) -> SteadyStateResults:
        r"""
        Solve for the deterministic steady state of the model.

        Given the system of model equations :math:`F(x_{t+1}, x_t, x_{t-1}, \varepsilon_t) = 0`, the steady state is
        the state vector :math:`\bar{x}` such that

        .. math::

            F(\bar{x}, \bar{x}, \bar{x}, 0) = 0.

        Absent an exogenous shock, the system stays at :math:`\bar{x}`. The model is linearized around this point.

        When the GCN file supplies a complete analytic steady state, it is evaluated and returned regardless of
        ``how``. Otherwise the analytic values are substituted into the equations and the remaining variables are
        solved numerically.

        Parameters
        ----------
        how : str, optional
            One of ``'analytic'``, ``'root'``, or ``'minimize'``. ``'root'`` runs a root finder on the residual
            system. ``'minimize'`` minimizes a scalar error over the residuals. ``'analytic'`` falls back to
            ``'minimize'`` when the analytic solution is incomplete. Default ``'analytic'``.
        use_jac : bool, optional
            Use the Jacobian of the residuals (``'root'``) or the gradient of the error (``'minimize'``). Ignored
            when ``how`` is ``'analytic'``. A ``'minimize'`` method that requires a gradient (``'trust-ncg'``,
            ``'trust-krylov'``, ``'trust-exact'``, ``'dogleg'``, ``'newton-cg'``) raises ``ValueError`` when this,
            ``use_hess``, and ``use_hessp`` are all False. Default True.
        use_hess : bool, optional
            Use the Hessian of the error function. Ignored unless ``how`` is ``'minimize'``. Default False.
        use_hessp : bool, optional
            Use the Hessian-vector product of the error function. Prefer this over ``use_hess`` when the method
            supports it, since it scales far better with the number of variables. Ignored unless ``how`` is
            ``'minimize'``. Default True.
        progressbar : bool, optional
            Display a progress bar while solving. Default True.
        optimizer_kwargs : dict, optional
            Keyword arguments passed to :func:`scipy.optimize.root` or :func:`scipy.optimize.minimize`, depending on
            ``how``. ``'method'`` selects the algorithm and defaults to ``'hybr'`` for ``'root'`` and
            ``'trust-ncg'`` for ``'minimize'``. ``'maxiter'`` caps the iterations, defaults to 5000, and is renamed
            to the argument the chosen method expects (``'hybr'`` takes ``maxfev``). Default None.
        verbose : bool, optional
            Log a convergence report. Default True.
        bounds : dict, optional
            Per-variable ``(lower, upper)`` bounds keyed by steady-state variable name. Either side may be None. A
            bounded variable is reparametrized onto the real line and solved unconstrained, except when
            ``prefer_transform`` is False and a bounds-capable method is requested, in which case the bounds go to
            :func:`scipy.optimize.minimize` as box constraints. A variable absent from this dict takes bounds from
            its GCN assumptions. Default None.
        prefer_transform : bool, optional
            If True, enforce bounds by reparametrization even for the bounds-capable methods ``'L-BFGS-B'``,
            ``'trust-constr'``, and ``'powell'``. Default False.
        fixed_values : dict, optional
            Steady-state values to hold fixed, keyed by variable name with or without the ``_ss`` suffix. Equations
            that the fixed values fully determine are checked for consistency. Default None.
        jitter_x0 : bool, optional
            Add ``N(0, 1e-4)`` noise to the initial point. Default False.
        **updates : float
            Parameter values at which to solve, forwarded to :meth:`parameters`. Unspecified parameters take their
            GCN defaults.

        Returns
        -------
        steady_state : SteadyStateResults
            Steady-state values keyed by variable name, with a ``success`` attribute reporting convergence.

        Examples
        --------
        Evaluate the analytic steady state declared in the GCN file:

        .. code-block:: python

            from gEconpy import model_from_gcn
            from gEconpy.data import get_example_gcn

            model = model_from_gcn(get_example_gcn("RBC"), verbose=False)
            steady_state = model.steady_state(verbose=False, progressbar=False)
            print(steady_state["K_ss"], steady_state.success)

        Solve numerically at new parameter values with a root finder:

        .. code-block:: python

            from gEconpy import model_from_gcn
            from gEconpy.data import get_example_gcn

            model = model_from_gcn(get_example_gcn("RBC"), verbose=False)
            steady_state = model.steady_state(how="root", beta=0.95, verbose=False, progressbar=False)
            print(steady_state["K_ss"], steady_state.success)
        """
        if how not in ("analytic", "root", "minimize"):
            raise NotImplementedError(f'how must be one of "analytic", "root", or "minimize", got {how!r}.')

        optimizer_kwargs = {} if optimizer_kwargs is None else optimizer_kwargs
        tol = optimizer_kwargs.get("tol", STEADY_STATE_TOL)
        param_dict = self.parameters(**updates)
        f_ss = self.f_ss

        if self.is_linear:
            return self._linear_steady_state()

        if fixed_values is None:
            analytic_result = self._try_analytic_steady_state(param_dict, f_ss)
            if analytic_result is not None:
                return analytic_result

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
            res, residual_functions = self._solve_steady_state_with_root(
                equations,
                ss_nodes,
                vars_to_solve,
                param_dict,
                use_jac=use_jac,
                progressbar=progressbar,
                optimizer_kwargs=optimizer_kwargs,
                jitter_x0=jitter_x0,
            )
        else:
            res, residual_functions = self._solve_steady_state_with_minimize(
                equations,
                ss_nodes,
                vars_to_solve,
                param_dict,
                use_jac=use_jac,
                use_hess=use_hess,
                use_hessp=use_hessp,
                progressbar=progressbar,
                bounds=bounds,
                prefer_transform=prefer_transform,
                optimizer_kwargs=optimizer_kwargs,
                jitter_x0=jitter_x0,
            )

        return self._postprocess_numerical_result(
            res,
            residual_functions,
            vars_to_solve,
            f_ss,
            param_dict,
            fixed_values,
            tol,
            verbose,
        )

    def evaluate_residual(self, ss_dict: dict[str, float], param_dict: SymbolDictionary) -> np.ndarray:
        """
        Evaluate the steady-state residual at given variable and parameter values.

        Parameters
        ----------
        ss_dict : dict mapping str to float
            Steady-state value of each model variable.
        param_dict : SymbolDictionary
            Value of each model parameter.

        Returns
        -------
        residuals : ndarray
            Residual of each model equation, which is zero at a valid steady state.

        Examples
        --------
        Confirm that a solved steady state satisfies every equation:

        .. code-block:: python

            import numpy as np

            from gEconpy import model_from_gcn
            from gEconpy.data import get_example_gcn

            model = model_from_gcn(get_example_gcn("RBC"), verbose=False)
            steady_state = model.steady_state(verbose=False, progressbar=False)
            residuals = model.evaluate_residual(steady_state, model.parameters())
            print(np.abs(residuals).max())
        """
        f_resid = self._compile_full_residual()
        return np.asarray(f_resid(**ss_dict, **param_dict))

    def symbolic_linearization(
        self,
        order: Literal[1] = 1,
        log_linearize: bool = True,
        not_loglin_variables: list[str] | None = None,
        steady_state: dict | None = None,
        loglin_negative_ss: bool = False,
        verbose: bool = True,
    ) -> tuple[list[TensorVariable], list[TensorVariable], list[TensorVariable], np.ndarray, np.ndarray]:
        r"""
        Build the symbolic pytensor graphs for the linearized Jacobian matrices.

        The cached graphs are the four Jacobian matrices ``A, B, C, D`` of the first-order approximation of the model
        around its steady state:

        .. math::

            A \hat{y}_{t-1} + B \hat{y}_t + C \hat{y}_{t+1} + D \varepsilon_t = 0

        The graphs are returned uncompiled and unevaluated, for the caller to inspect or compile.
        :meth:`linearize_model` compiles and evaluates them.

        Parameters
        ----------
        order : int, optional
            Order of the Taylor expansion. Only ``order=1`` is supported. Default 1.
        log_linearize : bool, optional
            If True, log-linearize every variable whose steady state allows it. If False, leave every variable in
            levels. Default True.
        not_loglin_variables : list of str, optional
            Variable names to leave in levels. Ignored if ``log_linearize`` is False. Default None.
        steady_state : dict, optional
            Steady-state values used to decide which variables have a non-positive steady state and so cannot be
            log-linearized. Solved internally at the default parameters when not given. Default None.
        loglin_negative_ss : bool, optional
            If True, log-linearize variables with a negative steady state as well. Default False.
        verbose : bool, optional
            Log which variables are excluded from log-linearization. Default True.

        Returns
        -------
        jacobians : list of TensorVariable
            Four pytensor matrix graph nodes ``[A, B, C, D]``. Rows are in :attr:`eq_order` and the variable columns
            of A, B and C are in :attr:`var_order`. The columns of D are shocks in model order.
        ss_input_nodes : list of TensorVariable
            Steady-state variable input nodes consumed by the Jacobian graphs.
        param_input_nodes : list of TensorVariable
            Parameter input nodes consumed by the Jacobian graphs.
        eq_order : ndarray of int
            Equation row permutation applied, a copy of :attr:`eq_order`.
        var_order : ndarray of int
            Variable column permutation applied, a copy of :attr:`var_order`.

        See Also
        --------
        linearize_model : Compile and numerically evaluate the linearized system.

        Examples
        --------
        Inspect the graph of the lead Jacobian without compiling it:

        .. code-block:: python

            import pytensor

            from gEconpy import model_from_gcn
            from gEconpy.data import get_example_gcn

            model = model_from_gcn(get_example_gcn("RBC"), verbose=False)
            jacobians, ss_nodes, param_nodes, eq_order, var_order = model.symbolic_linearization(verbose=False)
            A, B, C, D = jacobians
            pytensor.dprint(C, depth=3)
        """
        if order != 1:
            raise NotImplementedError("Only first order linearization is currently supported.")

        if self.is_linear:
            log_linearize = False

        # The steady state only decides the log-linearization flags, so it is solved only when those are needed.
        if steady_state is None and log_linearize:
            steady_state = self.f_ss(**self.parameters()) if self.is_linear else self.steady_state(verbose=verbose)

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

        if loglin_key not in self._symbolic_linearize_cache:
            jacobians, ss_input_nodes, eq_order, var_order = _linearize_model(
                variables=self.variables,
                equations=self.equations,
                shocks=self.shocks,
                cache=self._ensure_cache(),
                loglin_variables=loglin_vars,
                eq_order=self.eq_order,
                var_order=self.var_order,
            )

            ss_names = {n.name for n in ss_input_nodes}
            param_input_nodes = [
                v for v in explicit_graph_inputs(jacobians) if v.name is not None and v.name not in ss_names
            ]

            self._symbolic_linearize_cache[loglin_key] = (
                jacobians,
                ss_input_nodes,
                param_input_nodes,
                eq_order,
                var_order,
            )

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
        **parameter_updates: float,
    ) -> list[np.ndarray]:
        r"""
        Linearize the model around the deterministic steady state.

        Parameters
        ----------
        order : int, optional
            Order of the Taylor expansion. Only ``order=1`` is supported. Default 1.
        log_linearize : bool, optional
            If True, log-linearize every variable whose steady state allows it. If False, leave every variable in
            levels. Default True.
        not_loglin_variables : list of str, optional
            Variable names to leave in levels while the others are log-linearized. Ignored if ``log_linearize`` is
            False. Default None.
        steady_state : dict, optional
            Steady-state values to linearize around. Solved with :meth:`steady_state` when not given. Default None.
        loglin_negative_ss : bool, optional
            If True, log-linearize variables with a negative steady state as well. The result is not a valid
            approximation. Ignored if ``log_linearize`` is False. Default False.
        steady_state_kwargs : dict, optional
            Keyword arguments passed to :meth:`steady_state`. Ignored when ``steady_state`` is given. Default None.
        verbose : bool, optional
            Log the linearization results. Default True.
        **parameter_updates : float
            Parameter values at which to linearize. Unspecified parameters take their GCN defaults.

            .. warning::

                A ``steady_state`` passed in is used as given and is not re-solved at these parameter values. The
                caller is responsible for keeping the two consistent.

        Returns
        -------
        A : ndarray
            Jacobian of the model with respect to :math:`x_{t-1}` at the steady state, right-multiplied by the
            diagonal matrix :math:`T`.
        B : ndarray
            Jacobian of the model with respect to :math:`x_t` at the steady state, right-multiplied by :math:`T`.
        C : ndarray
            Jacobian of the model with respect to :math:`x_{t+1}` at the steady state, right-multiplied by :math:`T`.
        D : ndarray
            Jacobian of the model with respect to :math:`\varepsilon_t` at the steady state.

        Notes
        -----
        Given a DSGE model of the form

        .. math::

            F(x_{t+1}, x_t, x_{t-1}, \varepsilon_t) = 0,

        the solution is a policy function :math:`g(x_t, \varepsilon_t)` such that
        :math:`x_{t+1} = g(x_t, \varepsilon_t)`. Outside of toy models this policy function has no closed form, so
        the model is linearized around the deterministic steady state :math:`\bar{x}` satisfying
        :math:`F(\bar{x}, \bar{x}, \bar{x}, 0) = 0`, and the linear approximation stands in for the policy
        function. A first-order Taylor expansion about :math:`(\bar{x}, \bar{x}, \bar{x}, 0)` yields

        .. math::

            A (x_{t-1} - \bar{x}) + B (x_t - \bar{x}) + C (x_{t+1} - \bar{x}) + D \varepsilon_t = 0,

        where the Jacobian matrices evaluated at the steady state are

        .. math::

            A = \left. \frac{\partial F}{\partial x_{t-1}} \right|_{(\bar{x},\bar{x},\bar{x},0)}, \quad
            B = \left. \frac{\partial F}{\partial x_t} \right|_{(\bar{x},\bar{x},\bar{x},0)}, \quad
            C = \left. \frac{\partial F}{\partial x_{t+1}} \right|_{(\bar{x},\bar{x},\bar{x},0)}, \quad
            D = \left. \frac{\partial F}{\partial \varepsilon_t} \right|_{(\bar{x},\bar{x},\bar{x},0)}.

        Log-linearization is a change of variables. Define the log-state vector :math:`\tilde{x}_t = \log(x_t)`,
        with steady state :math:`\tilde{x}_{ss} = \log(\bar{x})`, so that

        .. math::

            F(\exp(\tilde{x}_{t+1}), \exp(\tilde{x}_t), \exp(\tilde{x}_{t-1}), \varepsilon_t) = 0.

        Differentiating with respect to :math:`\tilde{x}_t` gives the linearized model

        .. math::
            :nowrap:

            \[
            A \exp(\tilde{x}_{ss}) (\tilde{x}_{t-1} - \tilde{x}_{ss}) + B \exp(\tilde{x}_{ss}) (\tilde{x}_t -
            \tilde{x}_{ss}) + C \exp(\tilde{x}_{ss}) (\tilde{x}_{t+1} - \tilde{x}_{ss}) + D \varepsilon_t = 0.
            \]

        Here :math:`\tilde{x} - \tilde{x}_{ss} = \log(x / \bar{x})` is the approximate percent deviation of the
        variable from its steady state.

        The derivation holds variable by variable. Mixing logged and level variables amounts to right-multiplying by
        the diagonal matrix

        .. math::

            T = \text{Diagonal}(\{h(x_1), h(x_2), \ldots, h(x_n)\}),

        where :math:`h(x_i) = 1` if the variable is left in levels and :math:`h(x_i) = \exp(\tilde{x}_{ss})` if it
        is logged. This method returns the matrices :math:`AT`, :math:`BT`, :math:`CT`, and :math:`D`.

        Examples
        --------
        Linearize at the default parameters and read off the lag Jacobian:

        .. code-block:: python

            from gEconpy import model_from_gcn
            from gEconpy.data import get_example_gcn

            model = model_from_gcn(get_example_gcn("RBC"), verbose=False)
            A, B, C, D = model.linearize_model(verbose=False, steady_state_kwargs={"progressbar": False})
            print(A.shape, A.round(3))

        Linearize in levels at a new parameter value:

        .. code-block:: python

            from gEconpy import model_from_gcn
            from gEconpy.data import get_example_gcn

            model = model_from_gcn(get_example_gcn("RBC"), verbose=False)
            A, B, C, D = model.linearize_model(
                log_linearize=False,
                beta=0.95,
                verbose=False,
                steady_state_kwargs={"progressbar": False},
            )
            print(B.round(3))
        """
        if order != 1:
            raise NotImplementedError("Only first order linearization is currently supported.")

        steady_state_kwargs = {"verbose": verbose} | ({} if steady_state_kwargs is None else steady_state_kwargs)

        if self.is_linear:
            log_linearize = False

        param_dict = self.parameters(**parameter_updates)

        if steady_state is None:
            if self.is_linear:
                steady_state = self.f_ss(**param_dict)
            else:
                steady_state = self.steady_state(**param_dict, **steady_state_kwargs)

        jacobians, ss_input_nodes, param_input_nodes, eq_order, var_order = self.symbolic_linearization(
            order=order,
            log_linearize=log_linearize,
            not_loglin_variables=not_loglin_variables,
            steady_state=steady_state,
            loglin_negative_ss=loglin_negative_ss,
            verbose=verbose,
        )

        # symbolic_linearization returns the same graph objects on a cache hit, so the id of the first Jacobian node
        # is a stable key for the compiled function.
        cache_key = id(jacobians[0])
        if cache_key not in self._linearize_cache:
            all_inputs = list(ss_input_nodes) + list(param_input_nodes)
            self._linearize_cache[cache_key] = compile_pytensor_function(
                all_inputs, jacobians, mode=self._mode, on_unused_input="ignore"
            )
        f_jacobians = self._linearize_cache[cache_key]

        ss_values = {k.removesuffix("_ss"): v for k, v in steady_state.items()}
        ss_inputs = [ss_values[v.base_name] for v in self.variables]
        param_inputs = [param_dict[n.name] for n in param_input_nodes]

        A, B, C, D = f_jacobians(*ss_inputs, *param_inputs)

        # The graphs put equations in eq_order (rows) and variables in var_order (columns of A, B and C). Undo both so
        # the caller sees the original equation-by-variable layout.
        if not np.array_equal(eq_order, np.arange(len(eq_order))):
            inv_eq = self.inv_eq_order
            A, B, C, D = (M[inv_eq] for M in (A, B, C, D))
        if not np.array_equal(var_order, np.arange(len(var_order))):
            inv_var = self.inv_var_order
            A, B, C = (M[:, inv_var] for M in (A, B, C))

        return [np.ascontiguousarray(x, dtype=A.dtype) for x in [A, B, C, D]]

    def solve_model(
        self,
        solver: Literal["cycle_reduction", "gensys", "backward_direct"] = "cycle_reduction",
        log_linearize: bool = True,
        not_loglin_variables: list[str] | None = None,
        order: Literal[1] = 1,
        loglin_negative_ss: bool = False,
        steady_state: dict | None = None,
        steady_state_kwargs: dict | None = None,
        tol: float = 1e-8,
        max_iter: int = 1000,
        verbose: bool = True,
        on_failure: Literal["error", "ignore"] = "error",
        **parameter_updates: float,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        r"""
        Solve for the linear approximation to the policy function via perturbation.

        Parameters
        ----------
        solver : str, optional
            Algorithm for the linear solution, one of ``'cycle_reduction'``, ``'gensys'``, or ``'backward_direct'``.
            A model with no forward-looking variables always uses ``'backward_direct'``. Dynare defaults to cycle
            reduction and gEcon to gensys. Default ``'cycle_reduction'``.
        log_linearize : bool, optional
            If True, log-linearize the model. If False, solve it in levels. Default True.
        not_loglin_variables : list of str, optional
            Variable names to leave in levels. Variables with a steady state at or near zero are left in levels
            automatically. Ignored if ``log_linearize`` is False. Default None.
        order : int, optional
            Order of the Taylor expansion. Only ``order=1`` is supported. Default 1.
        loglin_negative_ss : bool, optional
            If True, log-linearize variables with a negative steady state as well. The result is not a valid
            approximation. See :func:`~gEconpy.model.perturbation.linearize_model`. Ignored if ``log_linearize`` is
            False. Default False.
        steady_state : dict, optional
            Steady-state values to linearize around. Solved with :meth:`steady_state` when not given. Default None.
        steady_state_kwargs : dict, optional
            Keyword arguments passed to :meth:`steady_state`. Ignored when ``steady_state`` is given. Default None.
        tol : float, optional
            Floating point tolerance of the solution. Default 1e-8.
        max_iter : int, optional
            Maximum number of cycle reduction iterations. Unused by the other solvers. Default 1000.
        verbose : bool, optional
            Log the solver results and the residual norms of the solution. Default True.
        on_failure : str, optional
            One of ``'error'`` or ``'ignore'``. ``'error'`` raises when the solver fails and ``'ignore'`` returns
            ``(None, None)`` instead, which suits repeated solves such as sampling. Default ``'error'``.
        **parameter_updates : float
            Parameter values at which to solve. Unspecified parameters take their GCN defaults.

        Returns
        -------
        T : ndarray or None
            Transition matrix of the policy function, mapping the lagged state to the current state. None when the
            solver fails and ``on_failure`` is ``'ignore'``.
        R : ndarray or None
            Selection matrix of the policy function, mapping current shocks to the current state. None when the
            solver fails and ``on_failure`` is ``'ignore'``.

        Notes
        -----
        The model is a system of the form

        .. math::
           :nowrap:

           \[
           \mathbb{E} \left [ F(x_{t+1}, x_t, x_{t-1}, \varepsilon_t) \right ] = 0.
           \]

        Its linear approximation is given by the matrices :math:`A`, :math:`B`, :math:`C`, and :math:`D` as

        .. math::
           :nowrap:

           \[
           A \hat{x}_{t-1} + B \hat{x}_t + C \hat{x}_{t+1} + D \varepsilon_t = 0,
           \]

        where :math:`\hat{x}_t = x_t - \bar{x}` is the deviation of the state vector from its steady state,
        possibly in logs. A solution is a function

        .. math::
           :nowrap:

           \[
           x_t = g(x_{t-1}, \varepsilon_t).
           \]

        Since :math:`x_{t+1} = g(x_t, \varepsilon_{t+1})`, the model can be written as

        .. math::
           :nowrap:

           \[
           F_g(x_{t-1}, \varepsilon_t, \varepsilon_{t+1}) =
           f(g(g(x_{t-1}, \varepsilon_t), \varepsilon_{t+1}),
             g(x_{t-1}, \varepsilon_t), x_{t-1}, \varepsilon_t) = 0.
           \]

        Define

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

        so the system reads

        .. math::
           :nowrap:

           \[
           F_g(x_-, u, u_+) = f(g(g(x_-, u), u_+), g(x_-, u), x_-, u) = 0.
           \]

        The unknown function :math:`g` is implicitly defined by this expression and is approximated by a first-order
        Taylor expansion around the steady state:

        .. math::
           :nowrap:

           \[
           0 \approx F_g(x_-, u, u_+) =
           f_{x_+} (g_x (g_x \hat{x} + g_u u) + g_u u_+) +
           f_x (g_x \hat{x} + g_u u) +
           f_{x_-} \hat{x} + f_u u.
           \]

        The Jacobians :math:`f_{x_+}`, :math:`f_x`, :math:`f_{x_-}`, and :math:`f_u` are the matrices :math:`C`,
        :math:`B`, :math:`A`, and :math:`D` evaluated at the steady state, so they are known. The task is to solve
        for :math:`g_x` and :math:`g_u`, the linear approximation to the policy function.

        Taking expectations and imposing :math:`\mathbb{E}_t[u_+] = 0` gives

        .. math::
           :nowrap:

           \begin{aligned}
           0 \approx {} &
           f_{x_+} (g_x(g_x \hat{x} + g_u u) + g_u \mathbb{E}_t[u_+]) +
           f_x (g_x \hat{x} + g_u u) + f_{x_-} \hat{x} + f_u u \\
           \approx {} &
           (f_{x_+} g_x g_x + f_x g_x + f_{x_-})\hat{x} +
           (f_{x_+} g_x g_u + f_x g_u + f_u) u.
           \end{aligned}

        Both coefficient matrices must vanish, which gives two equations in the unknowns :math:`g_x` and
        :math:`g_u`:

        .. math::
           :nowrap:

           \begin{aligned}
           (f_{x_+} g_x g_x + f_x g_x + f_{x_-}) \hat{x} &= 0 \\
           (f_{x_+} g_x g_u + f_x g_u + f_u) u &= 0.
           \end{aligned}

        Given :math:`g_x`, the second equation yields

        .. math::
           :nowrap:

           \[
           g_u = -(f_{x_+} g_x + f_x)^{-1} f_u.
           \]

        The first equation is quadratic in :math:`g_x`. Rewriting it as a linear system in two states,

        .. math::
           :nowrap:

           \begin{aligned}
           \begin{bmatrix} 0 & f_{x_+} \\ I & 0 \end{bmatrix}
           \begin{bmatrix} g_x g_x \\ g_x \end{bmatrix} \hat{x}
           &=
           \begin{bmatrix} -f_x & -f_{x_-} \\ I & 0 \end{bmatrix}
           \begin{bmatrix} g_x \\ I \end{bmatrix} \hat{x} \\
           D \begin{bmatrix} I \\ g_x \end{bmatrix} g_x \hat{x}
           &=
           E \begin{bmatrix} g_x \\ I \end{bmatrix} \hat{x} \\
           QTZ \begin{bmatrix} I \\ g_x \end{bmatrix} g_x \hat{x}
           &=
           QSZ \begin{bmatrix} g_x \\ I \end{bmatrix} \hat{x} \\
           TZ \begin{bmatrix} I \\ g_x \end{bmatrix} g_x \hat{x}
           &=
           SZ \begin{bmatrix} g_x \\ I \end{bmatrix} \hat{x}.
           \end{aligned}

        The last two lines use the QZ decomposition of the pencil :math:`<D, E>` into the upper triangular matrix
        :math:`T`, the quasi-upper triangular matrix :math:`S`, and the orthogonal matrices :math:`Z` and :math:`Q`.
        The decomposition orders the eigenvalues of the pencil by modulus from smallest (stable) to largest
        (unstable). Partitioning the rows by eigenvalue stability and the columns by the size of :math:`g_x` gives

        .. math::
           :nowrap:

           \[
           \begin{bmatrix} T_{11} & T_{12} \\ 0 & T_{22} \end{bmatrix}
           \begin{bmatrix} Z_{11} & Z_{12} \\ Z_{21} & Z_{22} \end{bmatrix}
           \begin{bmatrix} I \\ g_x \end{bmatrix} g_x \hat{x} =
           \begin{bmatrix} S_{11} & S_{12} \\ 0 & S_{22} \end{bmatrix}
           \begin{bmatrix} Z_{11} & Z_{12} \\ Z_{21} & Z_{22} \end{bmatrix}
           \begin{bmatrix} g_x \\ I \end{bmatrix} \hat{x}.
           \]

        Stability requires :math:`Z_{21} + Z_{22} g_x = 0`, so

        .. math::
           :nowrap:

           \[
           g_x = -Z_{22}^{-1} Z_{21}.
           \]

        This requires :math:`Z_{22}` to be square and invertible, the rank and stability conditions of Blanchard and
        Kahn (1980). Too few unstable roots leaves the solution indeterminate. Too many leaves no stable solution.

        Examples
        --------
        Solve the model and read off the transition and selection matrices:

        .. code-block:: python

            from gEconpy import model_from_gcn
            from gEconpy.data import get_example_gcn

            model = model_from_gcn(get_example_gcn("RBC"), verbose=False)
            T, R = model.solve_model(verbose=False, steady_state_kwargs={"progressbar": False})
            print(T.shape, R.shape)

        Solve repeatedly at sampled parameters, skipping draws where the solver fails:

        .. code-block:: python

            import numpy as np

            from gEconpy import model_from_gcn
            from gEconpy.data import get_example_gcn

            model = model_from_gcn(get_example_gcn("RBC"), verbose=False)
            rng = np.random.default_rng(0)

            for beta in rng.uniform(0.9, 0.999, size=3):
                T, R = model.solve_model(
                    beta=beta,
                    solver="gensys",
                    on_failure="ignore",
                    verbose=False,
                    steady_state_kwargs={"progressbar": False},
                )
                print(beta, T is not None)
        """
        if on_failure not in ("error", "ignore"):
            raise ValueError(f'Parameter on_failure must be one of "error" or "ignore", found {on_failure}')
        if steady_state_kwargs is None:
            steady_state_kwargs = {}

        ss_dict = _maybe_solve_steady_state(self, steady_state, steady_state_kwargs, parameter_updates)

        A, B, C, D = self.linearize_model(
            order=order,
            log_linearize=log_linearize,
            not_loglin_variables=not_loglin_variables,
            steady_state=ss_dict.to_string(),
            loglin_negative_ss=loglin_negative_ss,
            verbose=verbose,
            **parameter_updates,
        )

        if self._backward_looking:
            solver = "backward_direct"

        if solver == "gensys":
            T, R = self._solve_with_gensys(A, B, C, D, tol, verbose, on_failure)
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

        if T is None or R is None:
            return None, None

        if verbose:
            check_perturbation_solution(A, B, C, D, T, R, tol=tol)

        return np.ascontiguousarray(T), np.ascontiguousarray(R)

    def _ensure_cache(self) -> dict:
        """
        Return the shared sympytensor cache, creating it on first call.

        Every graph-building operation (residuals, linearization) shares this cache, so a model symbol always maps to
        the same pytensor node.
        """
        if self._cache is None:
            self._cache = {}
            compile_param_dict_func(self._param_dict, self._deterministic_dict, cache=self._cache, return_symbolic=True)
        return self._cache

    def _scalar_node(self, symbol: sp.Symbol) -> TensorVariable:
        cache = self._ensure_cache()
        cache_key = make_cache_key(symbol.name, type(symbol))
        if cache_key not in cache:
            cache[cache_key] = pt.scalar(name=symbol.name, dtype="floatX")
        return cache[cache_key]

    def _cached_nodes(self, symbols: list[sp.Symbol]) -> list[TensorVariable]:
        """Look up the pytensor nodes of ``symbols`` that already exist in the cache, skipping the rest."""
        cache = self._ensure_cache()
        cache_keys = (make_cache_key(symbol.name, type(symbol)) for symbol in symbols)
        return [cache[key] for key in cache_keys if key in cache]

    def _steady_state_symbols(self) -> list[sp.Symbol]:
        return [x.to_ss() for x in self._variables] + list(self.calibrated_params)

    def _known_steady_state_names(self) -> set[str]:
        if not self._ss_solution_dict:
            return set()
        return {safe_to_ss(k).name for k in self._ss_solution_dict.to_sympy()}

    @property
    def _vars_to_solve(self) -> list[sp.Symbol]:
        """Steady-state symbols with no analytic solution."""
        known_names = self._known_steady_state_names()
        return [v for v in self._steady_state_symbols() if v.name not in known_names]

    def _build_equation_tensors(self, known_steady_state: SymbolDictionary) -> list[TensorVariable]:
        equations, _ = _ss_residual_to_pytensor(
            self._steady_state_equations,
            known_steady_state,
            self._variables,
            self._param_dict,
            self._deterministic_dict,
            self._calib_dict,
            cache=self._ensure_cache(),
        )
        return equations

    def _ensure_equation_tensors(self, filter_known: bool = False) -> list[TensorVariable]:
        """
        Build and memoize the master equation graphs.

        With ``filter_known`` the analytically known steady-state values are substituted in and the equations they
        fully determine are dropped. Without it every equation is returned with every steady-state variable as a free
        input. The two variants are memoized separately.
        """
        if filter_known:
            if self._equation_tensors is None:
                self._equation_tensors = self._build_equation_tensors(self._ss_solution_dict)
            return self._equation_tensors

        if self._full_equation_tensors is None:
            self._full_equation_tensors = self._build_equation_tensors(SymbolDictionary())
        return self._full_equation_tensors

    def _compile_full_residual(self) -> Callable:
        if self._f_full_residual is None:
            self._f_full_residual = compile_for_scipy(pt.stack(self.equation_tensors()), mode=self._mode)
        return self._f_full_residual

    def _evaluate_steady_state(self, **updates: float) -> np.ndarray:
        """Evaluate the full residual system at the analytic steady-state values and the given parameters."""
        param_dict = self.parameters(**updates)
        f_resid = self._compile_full_residual()
        ss_dict = self.f_ss(**param_dict) if self.f_ss else {}
        return np.asarray(f_resid(**ss_dict, **param_dict))

    def _validate_provided_steady_state_variables(self, user_fixed_variables: Sequence[str]) -> None:
        normalized_names = [x.removesuffix("_ss") for x in user_fixed_variables]

        # Passing both ``x`` and ``x_ss`` is the only way to produce a duplicate after normalization.
        duplicates = sorted({x for x in normalized_names if normalized_names.count(x) > 1})
        if duplicates:
            raise ValueError(
                "The following variables were provided twice (once with a _ss suffix and once without):\n"
                f"{', '.join(duplicates)}"
            )

        model_variable_names = {x.base_name for x in self.variables}
        unknown_fixed = set(normalized_names) - model_variable_names
        if unknown_fixed:
            raise ValueError(
                f"The following variables or calibrated parameters were given fixed steady state values but are "
                f"unknown to the model: {', '.join(unknown_fixed)}"
            )

    def _linear_steady_state(self) -> SteadyStateResults:
        ss_dict = SteadyStateResults({x.to_ss(): 0.0 for x in self.variables}).to_string()
        ss_dict.success = True
        return ss_dict

    def _try_analytic_steady_state(
        self,
        param_dict: SymbolDictionary,
        f_ss: Callable | None,
    ) -> SteadyStateResults | None:
        """Evaluate the analytic steady state, or return None when it does not cover every variable."""
        if f_ss is None:
            return None
        ss_dict = f_ss(**param_dict)
        if len(ss_dict) != len(self.variables):
            return None

        f_resid = self._compile_full_residual()
        residual = np.asarray(f_resid(**ss_dict, **param_dict))
        success = np.allclose(residual, 0.0, atol=STEADY_STATE_TOL)

        result = SteadyStateResults(ss_dict.to_sympy()).to_string()
        result.success = success
        if not success:
            _log.warning(f"Steady State was not found. Sum of square residuals: {np.square(residual).sum()}")
        return result

    def _provided_steady_state_values(
        self,
        f_ss: Callable | None,
        param_dict: SymbolDictionary,
        fixed_values: dict[str, float] | None,
    ) -> dict[sp.Symbol, float]:
        """Merge the analytic steady-state values with the user-fixed values, keyed by steady-state symbol."""
        provided_ss_values = f_ss(**param_dict).to_sympy() if f_ss is not None else {}
        if fixed_values is not None:
            provided_ss_values.update({safe_to_ss(self.get(k)): v for k, v in fixed_values.items()})
        return provided_ss_values

    def _evaluate_all_resolved(
        self,
        f_ss: Callable | None,
        param_dict: SymbolDictionary,
        fixed_values: dict[str, float] | None,
    ) -> SteadyStateResults:
        """Assemble and validate the steady state when analytic and fixed values together cover every variable."""
        provided_ss_values = self._provided_steady_state_values(f_ss, param_dict, fixed_values)

        f_resid = self._compile_full_residual()
        residual = np.asarray(f_resid(**{str(k): v for k, v in provided_ss_values.items()}, **param_dict))

        result = SteadyStateResults({x: provided_ss_values[x] for x in self._steady_state_symbols()}).to_string()
        result.success = np.allclose(residual, 0.0, atol=STEADY_STATE_TOL)
        return result

    def _build_resid_with_fixed_values(
        self,
        fixed_values: dict[str, float],
        param_dict: SymbolDictionary,
    ) -> tuple[list[TensorVariable], list[sp.Symbol], list[TensorVariable]]:
        """
        Build the residual system with the user-fixed values merged into the known steady state.

        Equations fully determined by the fixed values alone are checked for consistency. The remaining system is
        returned for numerical solving along with the symbols and nodes still to be solved.
        """
        merged_ss = SymbolDictionary(self._ss_solution_dict.copy() if self._ss_solution_dict else {})
        for name, value in fixed_values.items():
            merged_ss[safe_to_ss(self.get(name))] = float(value)

        equations = self._build_equation_tensors(merged_ss)
        self._validate_fixed_value_equations(fixed_values, param_dict)

        merged_names = {safe_to_ss(k).name for k in merged_ss.to_sympy()}
        vars_to_solve = [v for v in self._steady_state_symbols() if v.name not in merged_names]
        ss_nodes = self._cached_nodes(vars_to_solve)

        return equations, vars_to_solve, ss_nodes

    def _validate_fixed_value_equations(self, fixed_values: dict[str, float], param_dict: SymbolDictionary) -> None:
        """
        Check that the equations fully determined by the fixed values have zero residuals.

        An equation that references any unfixed steady-state variable is skipped.
        """
        all_equations = self.equation_tensors()
        fixed_kw = {safe_to_ss(self.get(k)).name: float(v) for k, v in fixed_values.items()}

        unfixed_symbols = [symbol for symbol in self._steady_state_symbols() if symbol.name not in fixed_kw]
        unfixed_node_ids = {id(node) for node in self._cached_nodes(unfixed_symbols)}

        fully_determined_indices = [
            i
            for i, eq in enumerate(all_equations)
            if not any(id(node) in unfixed_node_ids for node in explicit_graph_inputs(eq))
        ]
        if not fully_determined_indices:
            return

        determined_resid = pt.stack([all_equations[i] for i in fully_determined_indices])
        f_resid = compile_for_scipy(determined_resid, mode=self._mode)
        residuals = np.asarray(f_resid(**fixed_kw, **param_dict))
        bad_indices = [
            fully_determined_indices[j] for j, value in enumerate(residuals) if abs(value) > STEADY_STATE_TOL
        ]
        if not bad_indices:
            return

        ss_system = system_to_steady_state(self.equations, self.shocks)
        bad_strs = [str(ss_system[i]) for i in bad_indices if i < len(ss_system)]
        raise ValueError(
            "User-provided steady state is not valid. The following equations had non-zero residuals "
            "after substitution:\n" + "\n".join(bad_strs)
        )

    def _solve_steady_state_with_root(
        self,
        equations: list[TensorVariable],
        ss_nodes: list[TensorVariable],
        vars_to_solve: list[sp.Symbol],
        param_dict: SymbolDictionary,
        use_jac: bool = True,
        progressbar: bool = True,
        optimizer_kwargs: dict | None = None,
        jitter_x0: bool = False,
    ) -> tuple[OptimizeResult, "_ResidualFunctions"]:
        optimizer_kwargs = deepcopy({} if optimizer_kwargs is None else optimizer_kwargs)

        maxiter = optimizer_kwargs.pop("maxiter", 5000)
        method = optimizer_kwargs.pop("method", "hybr")
        options = optimizer_kwargs.setdefault("options", {})
        options["maxfev" if method in ("hybr", "df-sane") else "maxiter"] = maxiter

        x0 = _initialize_x0(optimizer_kwargs, vars_to_solve, jitter_x0)

        resid, jac_graph = build_root_graphs(equations, ss_nodes, use_jac=use_jac)

        f_resid = pack_and_compile(resid, ss_nodes, param_dict=param_dict, mode=self._mode)
        f_jac = None
        f_jac_kw = None
        if jac_graph is not None:
            f_jac = pack_and_compile(jac_graph, ss_nodes, param_dict=param_dict, mode=self._mode)
            f_jac_kw = compile_for_scipy(jac_graph, mode=self._mode)

        with np.errstate(all="ignore"):
            res = root(f=f_resid, x0=x0, jac=f_jac, method=method, progressbar=progressbar, **optimizer_kwargs)

        residual_functions = _ResidualFunctions(resid=compile_for_scipy(resid, mode=self._mode), jac=f_jac_kw)
        return res, residual_functions

    def _solve_steady_state_with_minimize(
        self,
        equations: list[TensorVariable],
        ss_nodes: list[TensorVariable],
        vars_to_solve: list[sp.Symbol],
        param_dict: SymbolDictionary,
        use_jac: bool = True,
        use_hess: bool = False,
        use_hessp: bool = True,
        progressbar: bool = True,
        optimizer_kwargs: dict | None = None,
        jitter_x0: bool = False,
        bounds: dict[str, tuple[float, float]] | None = None,
        prefer_transform: bool = False,
    ) -> tuple[OptimizeResult, "_ResidualFunctions"]:
        optimizer_kwargs = deepcopy({} if optimizer_kwargs is None else optimizer_kwargs)

        x0 = _initialize_x0(optimizer_kwargs, vars_to_solve, jitter_x0)
        tol = optimizer_kwargs.pop("tol", 1e-30)

        bound_dict = {x.name: infer_variable_bounds(x) for x in vars_to_solve}
        bound_dict.update({} if bounds is None else bounds)

        # Box-constrained solvers crawl an ill-conditioned interior-point central path on DSGE steady states, so they
        # run only when the caller names one and leaves ``prefer_transform`` off. Otherwise each bounded variable is
        # reparametrized onto the real line (x = transform.backward(y)) and the unconstrained problem is solved, which
        # also lets a bounds-capable method such as L-BFGS-B run on the transformed space when ``prefer_transform``
        # is set.
        requested_method = optimizer_kwargs.pop("method", None)
        use_box_bounds = (not prefer_transform) and requested_method in BOX_BOUND_METHODS

        if use_box_bounds:
            method = requested_method
            solve_equations, solve_nodes = equations, ss_nodes
            x0_solve = x0
            bounds_arg = [bound_dict[x.name] for x in vars_to_solve]
            to_constrained = None
        else:
            method = requested_method or "trust-ncg"
            transforms = [infer_variable_transform(v, bound_dict[v.name]) for v in vars_to_solve]
            solve_equations, solve_nodes, to_unconstrained, to_constrained = transform_steady_state_system(
                equations, ss_nodes, transforms
            )
            x0_solve = to_unconstrained(x0)
            bounds_arg = None

        gradient_available = use_jac or use_hess or use_hessp
        if not gradient_available and method.lower() in GRADIENT_REQUIRED_METHODS:
            raise ValueError(
                f"Method {method!r} requires a gradient, but use_jac, use_hess, and use_hessp are all False. Pass "
                f"use_jac=True, or choose a gradient-free method such as 'nelder-mead' or 'powell' via "
                f"optimizer_kwargs={{'method': ...}}."
            )

        maxiter = optimizer_kwargs.pop("maxiter", 5000)
        options = optimizer_kwargs.setdefault("options", {})
        options["maxiter"] = maxiter
        if method == "L-BFGS-B":
            options["maxfun"] = maxiter

        if use_hess and use_hessp:
            _log.warning("Both use_hess and use_hessp are set to True. use_hessp will be used.")
            use_hess = False

        error_graph, grad_graph, hess_graph, hessp_graph, hessp_p = build_minimize_graphs(
            solve_equations,
            solve_nodes,
            error_func=self._error_func,
            use_jac=use_jac,
            use_hess=use_hess,
            use_hessp=use_hessp,
        )

        def pack(graph: TensorVariable | None) -> Callable | None:
            if graph is None:
                return None
            return pack_and_compile(graph, solve_nodes, param_dict=param_dict, mode=self._mode)

        f_error = pack(error_graph)
        f_grad = pack(grad_graph)
        f_hess = pack(hess_graph)

        # The Hessian-vector product takes a direction vector at call time, which pack_and_compile cannot bake in.
        f_hessp: Callable | None = None
        if hessp_graph is not None:
            f_hessp_inner = compile_for_scipy(hessp_graph, mode=self._mode)
            var_names = [v.name for v in solve_nodes]

            def hessian_vector_product(x: np.ndarray, p: np.ndarray) -> np.ndarray:
                kw = dict(zip(var_names, x, strict=True))
                kw[hessp_p.name] = p
                return np.asarray(f_hessp_inner(**kw, **param_dict))

            f_hessp = hessian_vector_product

        res = minimize(
            f=f_error,
            x0=x0_solve,
            jac=f_grad,
            hess=f_hess,
            hessp=f_hessp,
            method=method,
            bounds=bounds_arg,
            tol=tol,
            progressbar=progressbar,
            **optimizer_kwargs,
        )
        if to_constrained is not None:
            res.x = to_constrained(res.x)

        # Postprocessing evaluates the original constrained system at the mapped-back solution, so its functions stay
        # in x-space even when the solve ran in y-space.
        resid = pt.stack(equations) if equations else pt.zeros(0)
        if use_box_bounds:
            grad_graph_x = grad_graph
        else:
            _, grad_graph_x, *_ = build_minimize_graphs(
                equations, ss_nodes, error_func=self._error_func, use_jac=use_jac, use_hess=False, use_hessp=False
            )

        residual_functions = _ResidualFunctions(
            resid=compile_for_scipy(resid, mode=self._mode),
            grad=compile_for_scipy(grad_graph_x, mode=self._mode) if grad_graph_x is not None else None,
        )
        return res, residual_functions

    def _postprocess_numerical_result(
        self,
        res: OptimizeResult,
        residual_functions: "_ResidualFunctions",
        vars_to_solve: list[sp.Symbol],
        f_ss: Callable | None,
        param_dict: SymbolDictionary,
        fixed_values: dict[str, float] | None,
        tol: float,
        verbose: bool,
    ) -> SteadyStateResults:
        """Assemble the steady-state dict from the optimizer result and run the convergence diagnostics."""
        provided_ss_values = self._provided_steady_state_values(f_ss, param_dict, fixed_values)
        optimizer_results = SymbolDictionary(dict(zip(vars_to_solve, res.x, strict=True)))
        all_values = optimizer_results | provided_ss_values
        res_dict = SteadyStateResults({x: all_values[x] for x in self._steady_state_symbols()}).to_string()

        def f_resid(**kw: float) -> np.ndarray:
            return np.asarray(residual_functions.resid(**kw, **param_dict))

        def f_grad(**kw: float) -> np.ndarray:
            if residual_functions.grad is not None:
                return np.asarray(residual_functions.grad(**kw, **param_dict))
            if residual_functions.jac is not None:
                resid_val = np.asarray(residual_functions.resid(**kw, **param_dict)).ravel()
                jac_val = np.asarray(residual_functions.jac(**kw, **param_dict))
                return 2.0 * jac_val.T @ resid_val
            return np.zeros(len(vars_to_solve))

        return postprocess_optimizer_res(
            res=res,
            res_dict=res_dict,
            f_resid=f_resid,
            f_grad=f_grad,
            tol=tol,
            verbose=verbose,
        )

    def _solve_with_gensys(
        self,
        A: np.ndarray,
        B: np.ndarray,
        C: np.ndarray,
        D: np.ndarray,
        tol: float,
        verbose: bool,
        on_failure: str,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        G_1, _constant, impact, _f_mat, _f_wt, _y_wt, _gev, eu, _loose = solve_policy_function_with_gensys(
            A, B, C, D, tol
        )

        success = all(x == 1 for x in eu[:2])
        if not success and on_failure == "error":
            raise GensysFailedException(eu)

        if verbose:
            _log.info(interpret_gensys_output(eu))
        if not success:
            return None, None

        T = G_1[: self.n_variables, : self.n_variables]
        R = impact[: self.n_variables, :]
        return T, R

    def _solve_with_cycle_reduction(
        self,
        A: np.ndarray,
        B: np.ndarray,
        C: np.ndarray,
        D: np.ndarray,
        max_iter: int,
        tol: float,
        verbose: bool,
        on_failure: str,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        T, R, result, _log_norm = solve_policy_function_with_cycle_reduction(A, B, C, D, max_iter, tol, verbose)
        if T is None:
            if on_failure == "error":
                raise GensysFailedException(message=result)
            if verbose:
                _log.info(result)
            return None, None
        return T, R


class _ResidualFunctions(NamedTuple):
    """Keyword-argument residual system functions used by the steady-state convergence diagnostics."""

    resid: Callable
    jac: Callable | None = None
    grad: Callable | None = None


def _initialize_x0(optimizer_kwargs: dict, variables: list[sp.Symbol], jitter_x0: bool) -> np.ndarray:
    n_variables = len(variables)
    user_x0 = optimizer_kwargs.pop("x0", None)

    if user_x0 is None:
        x0 = np.full(n_variables, 0.8)
        is_negative = np.array([x.assumptions0.get("negative", False) for x in variables], dtype=bool)
        x0[is_negative] = -x0[is_negative]
    else:
        x0 = np.array(user_x0, dtype=float)

    if jitter_x0:
        rng = np.random.default_rng()
        x0 += rng.normal(scale=1e-4, size=n_variables)

    return x0
