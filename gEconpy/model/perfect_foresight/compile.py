from collections.abc import Callable
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
import pytensor
import pytensor.tensor as pt
import sympy as sp

from pytensor.compile.builders import SymbolicOp
from pytensor.compile.mode import Mode, get_mode
from pytensor.graph.basic import Variable
from pytensor.graph.replace import graph_replace, vectorize_graph
from pytensor.graph.traversal import explicit_graph_inputs
from pytensor.link.numba.linker import NumbaLinker
from pytensor.tensor.blockwise import Blockwise
from pytensor.tensor.utils import safe_signature
from scipy import sparse
from scipy.sparse.linalg import splu
from sympytensor import as_tensor

from gEconpy.classes.containers import SymbolDictionary
from gEconpy.classes.time_aware_symbol import TimeAwareSymbol
from gEconpy.model.compile import build_symbolic_jacobian, make_cache_key
from gEconpy.model.model import Model
from gEconpy.model.perfect_foresight.assemble import (
    StackedJacobianLayout,
    build_stacked_jacobian_layout,
)
from gEconpy.model.steady_state import known_steady_state_graphs
from gEconpy.model.timing import classify_variables_by_timing
from gEconpy.utilities import safe_to_ss

_NATURAL_ORDER_MAX_FILL_RATIO = 4.0


@dataclass(frozen=True)
class PerfectForesightProblem:
    """
    Compiled equations of a model, plus the names that fix their input and output ordering.

    Attributes
    ----------
    f_resid_and_jac : callable
        Function returning the residuals and the Jacobian of every period's equations at once. It is called as
        ``f(Y_tm1, Y_t, Y_tp1, X, P)``, where the first three arguments have shape ``(T, n_vars)``, ``X`` has
        shape ``(T, n_shocks)``, and ``P`` has shape ``(T, n_params)``, all float64. The inputs are not checked
        unless the problem was compiled with ``trust_input=False``. The residuals have shape ``(T, n_eq)`` and the
        Jacobian has shape ``(T, n_eq, 3 * n_vars)``.
    f_resid_only : callable
        Function with the same inputs returning only the residuals.
    jacobian_sparsity : ndarray of bool
        Structural nonzero mask of one period's Jacobian, of shape ``(n_eq, 3 * n_vars)``. It is derived from the
        equations, so it holds at every point the solver can visit.
    var_names : tuple of str
        Variable names, giving the column order of the stacked system.
    shock_names : tuple of str
        Shock names, giving the column order of ``X``.
    param_names : tuple of str
        Parameter names, giving the column order of ``P``.
    deterministic_param_names : tuple of str
        The subset of ``param_names`` that the model derives from other parameters.
    T : int
        Number of periods in the simulation horizon.
    """

    f_resid_and_jac: Callable
    f_resid_only: Callable
    jacobian_sparsity: np.ndarray
    var_names: tuple[str, ...]
    shock_names: tuple[str, ...]
    param_names: tuple[str, ...]
    deterministic_param_names: tuple[str, ...]
    T: int

    @property
    def n_vars(self) -> int:
        """Number of model variables."""
        return len(self.var_names)

    @property
    def n_shocks(self) -> int:
        """Number of model shocks."""
        return len(self.shock_names)

    @property
    def n_eq(self) -> int:
        """Number of model equations, which :func:`compile_perfect_foresight_problem` checks equals ``n_vars``."""
        return self.jacobian_sparsity.shape[0]

    @cached_property
    def jacobian_layout(self) -> StackedJacobianLayout:
        """Index arrays of the stacked Jacobian, resolved once and reused across Newton iterations."""
        return build_stacked_jacobian_layout(
            sparsity_pattern=self.jacobian_sparsity, n_vars=self.n_vars, n_eq=self.n_eq, T=self.T
        )

    @cached_property
    def column_order(self) -> np.ndarray | None:
        """
        Column permutation for factorizing the stacked Jacobian, or None to keep the natural block-tridiagonal order.

        The natural order factorizes fastest unless its fill is far larger than a fill-reducing order's. Both fills
        are measured on the sparsity pattern alone, factorizing without pivoting with placeholder values, and COLAMD's
        order is used when the natural fill is more than four times its fill.
        """
        layout = self.jacobian_layout
        placeholder = np.random.default_rng(0).standard_normal(layout.indices.size)
        pattern = sparse.csc_matrix((placeholder, layout.indices, layout.indptr), shape=layout.shape)
        try:
            natural = splu(pattern, permc_spec="NATURAL", diag_pivot_thresh=0.0)
            colamd = splu(pattern, permc_spec="COLAMD", diag_pivot_thresh=0.0)
        except RuntimeError:
            # A structurally singular pattern has no factorization to compare. The solver reports the failure.
            return None

        natural_fill = natural.L.nnz + natural.U.nnz
        colamd_fill = colamd.L.nnz + colamd.U.nnz
        if natural_fill > _NATURAL_ORDER_MAX_FILL_RATIO * colamd_fill:
            return np.argsort(colamd.perm_c)
        return None


def compile_perfect_foresight_problem(
    model: Model,
    T: int,
    **compile_kwargs,
) -> PerfectForesightProblem:
    """
    Compile the residual and Jacobian functions used by the perfect foresight solver.

    The compiled functions do not depend on ``T``, so they are cached on the model and reused across horizons
    and across calls. The returned problem is cached per horizon as well, so repeated calls with the same ``T`` and
    compile options return the same object and build its stacked Jacobian layout once.

    Parameters
    ----------
    model : Model
        Model to compile. It must be square, every variable must appear at time ``t``, it must have no calibrated
        parameters, and every steady-state variable in its equations must have an analytic solution.
    T : int
        Number of periods in the simulation horizon.
    **compile_kwargs
        Keyword arguments forwarded to :func:`pytensor.function`. ``on_unused_input`` defaults to ``"ignore"`` and
        ``trust_input`` to True, so the compiled functions skip checking their inputs on every call.

    Returns
    -------
    problem : PerfectForesightProblem
        The compiled functions, together with the variable, shock, and parameter names.

    Raises
    ------
    ValueError
        If the model is not square, if a variable never appears at time ``t``, if the model has calibrated
        parameters, or if a steady-state variable has no analytic solution.
    """
    compile_kwargs.setdefault("on_unused_input", "ignore")
    compile_kwargs.setdefault("trust_input", True)

    # Compilation costs more than several Newton iterations, and the natural use of this module is a sweep over
    # shocks or parameter paths, so a repeated call must not pay for it again. repr keys the cache on the kwargs
    # without requiring them to be hashable.
    cache_key = tuple(sorted((name, repr(value)) for name, value in compile_kwargs.items()))
    if cache_key not in model._perfect_foresight_cache:
        model._perfect_foresight_cache[cache_key] = _compile_equations(model, **compile_kwargs)

    compiled = model._perfect_foresight_cache[cache_key]
    if T not in compiled.problems:
        compiled.problems[T] = PerfectForesightProblem(
            f_resid_and_jac=compiled.f_resid_and_jac,
            f_resid_only=compiled.f_resid_only,
            jacobian_sparsity=compiled.graph.sparsity,
            var_names=tuple(compiled.graph.var_names),
            shock_names=tuple(compiled.graph.shock_names),
            param_names=tuple(compiled.graph.param_names),
            deterministic_param_names=tuple(compiled.graph.deterministic_param_names),
            T=T,
        )

    return compiled.problems[T]


@dataclass
class _EquationGraph:
    """Residuals and Jacobian of one period, as functions of ``y_tm1``, ``y_t``, ``y_tp1``, ``x_t``, and ``p``."""

    inputs: list[pt.TensorVariable]
    residuals: pt.TensorVariable
    jacobian: pt.TensorVariable
    sparsity: np.ndarray
    var_names: list[str]
    shock_names: list[str]
    param_names: list[str]
    deterministic_param_names: list[str]


@dataclass(frozen=True)
class _CompiledEquations:
    f_resid_and_jac: Callable
    f_resid_only: Callable
    graph: _EquationGraph
    problems: dict[int, PerfectForesightProblem] = field(default_factory=dict)


def _compile_equations(model: Model, **compile_kwargs) -> _CompiledEquations:
    """Build the horizon-independent half of a :class:`PerfectForesightProblem`."""
    graph = _build_equation_graph(model)
    horizon_inputs = [pt.dmatrix(name) for name in ("Y_tm1", "Y_t", "Y_tp1", "X", "P")]

    batch = _batch_with_blockwise if _uses_numba(compile_kwargs.get("mode")) else _batch_with_vectorize
    residuals_and_jacobian = batch(graph.inputs, [graph.residuals, graph.jacobian], horizon_inputs)
    residuals_only = batch(graph.inputs, [graph.residuals], horizon_inputs)

    return _CompiledEquations(
        f_resid_and_jac=pytensor.function(horizon_inputs, residuals_and_jacobian, **compile_kwargs),
        f_resid_only=pytensor.function(horizon_inputs, residuals_only, **compile_kwargs),
        graph=graph,
    )


class _PeriodEquations(SymbolicOp):
    """The equations of one period, kept opaque so that ``Blockwise`` evaluates the horizon in a compiled loop."""

    def __init__(self, inputs: list[pt.TensorVariable], outputs: list[pt.TensorVariable]):
        self._template = (inputs, outputs)
        super().__init__(input_types=[x.type for x in inputs])
        # The inner graph now holds the equations. A second copy would more than double what each worker unpickles.
        del self._template

    def build_inner_graph(self, *inputs) -> list[Variable]:
        template_inputs, template_outputs = self._template
        return graph_replace(template_outputs, dict(zip(template_inputs, inputs, strict=True)), strict=False)


def _uses_numba(mode: str | Mode | None) -> bool:
    return isinstance(get_mode(mode).linker, NumbaLinker)


def _batch_with_blockwise(
    period_inputs: list[pt.TensorVariable],
    period_outputs: list[pt.TensorVariable],
    horizon_inputs: list[Variable],
) -> list[Variable]:
    # Numba compiles the period graph once and calls it for each period. Fused across the horizon instead, the graph
    # becomes one kernel with an array per Jacobian entry, which numba compiles dozens of times more slowly. The loop
    # costs a few dozen microseconds per period in allocations.
    signature = safe_signature(
        core_inputs_ndim=[x.type.ndim for x in period_inputs],
        core_outputs_ndim=[x.type.ndim for x in period_outputs],
    )
    batched = Blockwise(_PeriodEquations(period_inputs, period_outputs), signature=signature)
    return batched.make_node(*horizon_inputs).outputs


def _batch_with_vectorize(
    period_inputs: list[pt.TensorVariable],
    period_outputs: list[pt.TensorVariable],
    horizon_inputs: list[Variable],
) -> list[Variable]:
    # The other backends would run Blockwise as a Python loop over the periods, and the per-call overhead of a compiled
    # function dominates the arithmetic of one period, so the graph is vectorized over the horizon instead.
    return vectorize_graph(period_outputs, replace=dict(zip(period_inputs, horizon_inputs, strict=True)))


def _build_equation_graph(model: Model) -> _EquationGraph:
    """
    Build the symbolic residual and Jacobian of the model equations in one period.

    The equations are converted to pytensor once, differentiated with respect to every variable at ``t-1``, ``t``, and
    ``t+1``, and rewritten so that each scalar variable reads from an element of ``y_tm1``, ``y_t``, ``y_tp1``, ``x_t``,
    or the parameter vector ``p``. A variable that never appears at some time index gets a structurally zero Jacobian
    column there.
    """
    shock_names = [s.base_name for s in model.shocks]

    equations = model.equations
    ss_values = _steady_state_values(equations, model._ss_solution_dict)
    vars_tm1, vars_t, vars_tp1, shocks_t = classify_variables_by_timing(
        _hide_steady_state_symbols(equations, ss_values), shock_names
    )
    _check_system_is_solvable(equations=equations, vars_tm1=vars_tm1, vars_t=vars_t, vars_tp1=vars_tp1)

    cache: dict = {}
    equations_pt = [as_tensor(eq, cache=cache) for eq in equations]

    def get_pt_var(sym: TimeAwareSymbol) -> pt.TensorVariable:
        return cache[make_cache_key(sym.name, cls=TimeAwareSymbol)]

    # Steady-state variables stay symbols through differentiation and are bound to their parameter-only graphs after
    # it. Substituting the closed forms in sympy would copy each into every Jacobian entry it touches, and
    # differentiating those copies takes minutes. The graphs are built in a copy of the cache so that parameters only
    # the steady state block references are not mistaken for parameters of the equations.
    ss_cache = dict(cache)
    ss_graphs = known_steady_state_graphs(ss_values, model._ss_definitions, ss_cache)
    ss_replacements = {get_pt_var(symbol): graph for symbol, graph in zip(ss_values, ss_graphs, strict=True)}

    var_names = [x.base_name for x in vars_t]
    vars_t_pt = [get_pt_var(x) for x in vars_t]
    shocks_pt = [get_pt_var(x) for x in shocks_t]
    n_vars = len(vars_t_pt)
    n_shocks = len(shocks_pt)

    # Variables absent at t-1 or t+1 get a dummy scalar so every time index has one node per variable.
    tm1_by_name = {x.base_name: get_pt_var(x) for x in vars_tm1}
    tp1_by_name = {x.base_name: get_pt_var(x) for x in vars_tp1}
    vars_tm1_pt = [tm1_by_name.get(name, pt.dscalar(f"{name}_tm1_dummy")) for name in var_names]
    vars_tp1_pt = [tp1_by_name.get(name, pt.dscalar(f"{name}_tp1_dummy")) for name in var_names]

    # The shared cache makes the Jacobian's column nodes the same objects as the vars_*_pt lists wherever the
    # variable appears at that time index, so the vector replacement below lines up.
    jacobian_wrt = [v.set_t(-1) for v in vars_t] + list(vars_t) + [v.set_t(1) for v in vars_t]
    jacobian = build_symbolic_jacobian(equations, jacobian_wrt, cache, to_ss=False)
    sparsity = _jacobian_sparsity(equations, jacobian_wrt)
    residuals = pt.stack(equations_pt)

    y_tm1 = pt.dvector("y_tm1", shape=(n_vars,))
    y_t = pt.dvector("y_t", shape=(n_vars,))
    y_tp1 = pt.dvector("y_tp1", shape=(n_vars,))
    x_t = pt.dvector("x_t", shape=(n_shocks,))

    stacked = ((y_tm1, vars_tm1_pt), (y_t, vars_t_pt), (y_tp1, vars_tp1_pt), (x_t, shocks_pt))
    replacements = {node: vector[i] for vector, nodes in stacked for i, node in enumerate(nodes)}
    replacements.update(ss_replacements)

    residuals_vec, jacobian_vec = graph_replace([residuals, jacobian], replacements, strict=False)

    graph_inputs = set(explicit_graph_inputs([residuals_vec, jacobian_vec]))
    _check_no_calibrated_parameters(calibrated_params=model.calibrated_params, cache=ss_cache, inputs=graph_inputs)
    params_in_equations, params_pt = _collect_parameters_in_equations(model, cache=ss_cache, inputs=graph_inputs)
    deterministic_names = {p.name for p in model.deterministic_params}

    p = pt.dvector("p", shape=(len(params_pt),))
    residuals_period, jacobian_period = graph_replace(
        [residuals_vec, jacobian_vec], {param: p[i] for i, param in enumerate(params_pt)}, strict=False
    )

    return _EquationGraph(
        inputs=[y_tm1, y_t, y_tp1, x_t, p],
        residuals=residuals_period,
        jacobian=jacobian_period,
        sparsity=sparsity,
        var_names=var_names,
        shock_names=[x.base_name for x in shocks_t],
        param_names=[p.name for p in params_in_equations],
        deterministic_param_names=[p.name for p in params_in_equations if p.name in deterministic_names],
    )


def _check_system_is_solvable(
    equations: list[sp.Expr],
    vars_tm1: list[TimeAwareSymbol],
    vars_t: list[TimeAwareSymbol],
    vars_tp1: list[TimeAwareSymbol],
) -> None:
    """
    Check that the system is square and that every variable appears at time ``t``.

    Raises
    ------
    ValueError
        If the system is not square, or if a variable appears at ``t-1`` or ``t+1`` but never at ``t``.
    """
    # Both checks fail far from their cause if left to the compiler: a non-square system reaches
    # build_stacked_jacobian_layout as a shape mismatch, and a variable missing at t is never given an input
    # slot, so pytensor reports it as a missing input to an anonymous subgraph.
    names_at_t = {x.base_name for x in vars_t}
    only_at_other_times = sorted({x.base_name for x in vars_tm1 + vars_tp1} - names_at_t)
    if only_at_other_times:
        raise ValueError(
            f"Perfect foresight simulation requires every variable to appear at time t, but the following "
            f"appear only at t-1 or t+1: {', '.join(only_at_other_times)}."
        )

    if len(equations) != len(vars_t):
        raise ValueError(
            f"Perfect foresight simulation requires a square system, but the model has {len(equations)} "
            f"equations and {len(vars_t)} variables."
        )


def _collect_parameters_in_equations(
    model: Model,
    cache: dict,
    inputs: set[Variable],
) -> tuple[list[sp.Symbol], list[pt.TensorVariable]]:
    """
    Pick out the parameters the equations actually use, in the order the compiled functions expect them.

    Parameters
    ----------
    model : Model
        Model whose free and deterministic parameters are candidates.
    cache : dict
        Sympytensor cache holding the node of every parameter the equations or their steady-state values reference.
    inputs : set of Variable
        Root inputs of the equation graph, after the steady-state values are bound.

    Returns
    -------
    parameters : list of sympy Symbol
        The free and deterministic parameters appearing in the equations.
    nodes : list of TensorVariable
        The pytensor node of each, in the same order.
    """
    parameters, nodes = [], []
    for param in model.params + model.deterministic_params:
        node = cache.get(make_cache_key(param.name, cls=sp.Symbol))
        if node in inputs:
            parameters.append(param)
            nodes.append(node)

    return parameters, nodes


def _check_no_calibrated_parameters(
    calibrated_params: list[sp.Symbol],
    cache: dict,
    inputs: set[Variable],
) -> None:
    """
    Check that no calibrated parameter appears in the equations.

    Raises
    ------
    ValueError
        If a calibrated parameter is used by the equations.
    """
    # A calibrated parameter is solved for alongside the steady state, so its value is not known when the
    # equations are compiled and it never gets an input slot. Caught here, the user learns which parameter and
    # why; left to pytensor, it surfaces as a MissingInputError naming an arbitrary node that uses it.
    in_equations = [p.name for p in calibrated_params if cache.get(make_cache_key(p.name, cls=sp.Symbol)) in inputs]
    if in_equations:
        raise ValueError(
            f"Perfect foresight simulation does not support calibrated parameters, but the equations use: "
            f"{', '.join(sorted(in_equations))}. Replace the calibration equation with a fixed value in the "
            f"calibration block of your GCN file."
        )


def _jacobian_sparsity(equations: list[sp.Expr], wrt: list[TimeAwareSymbol]) -> np.ndarray:
    """
    Structural nonzero mask of ``d(equations)/d(wrt)``.

    Parameters
    ----------
    equations : list of sympy expression
        Equations to differentiate, one per row of the mask.
    wrt : list of TimeAwareSymbol
        Symbols to differentiate with respect to, one per column of the mask.

    Returns
    -------
    sparsity : ndarray of bool
        Mask of shape ``(len(equations), len(wrt))``, True wherever the derivative can be nonzero.
    """
    # Membership rather than differentiation: a symbol absent from an equation has an identically zero
    # derivative there, so the mask is a superset of the true nonzero set at a fraction of the cost. The
    # superset direction is the safe one, since a missing entry would silently corrupt the Newton step.
    symbols_by_equation = [equation.free_symbols for equation in equations]

    return np.array([[symbol in symbols for symbol in wrt] for symbols in symbols_by_equation], dtype=bool)


def _steady_state_values(
    equations: list[sp.Expr],
    ss_solution_dict: SymbolDictionary | None,
) -> dict[sp.Symbol, sp.Expr | float]:
    """
    Look up the analytic expression of every steady-state variable in the equations.

    The expression is in model parameters, so under a parameter path the bound term takes the parameter values of
    each period rather than staying fixed at the initial steady state.

    Parameters
    ----------
    equations : list of sympy.Expr
        Model equations that may contain ``X_ss`` symbols.
    ss_solution_dict : SymbolDictionary, optional
        Analytically known steady-state solutions, keyed by steady-state variable.

    Returns
    -------
    values : dict mapping sympy Symbol to sympy expression or float
        Analytic value of each steady-state symbol in the equations, in a deterministic order.
    """
    ss_atoms = {a for eq in equations for a in eq.atoms(TimeAwareSymbol) if a.time_index == "ss"}
    if not ss_atoms:
        return {}

    values_by_name = {}
    if ss_solution_dict:
        values_by_name = {safe_to_ss(key).name: value for key, value in ss_solution_dict.to_sympy().items()}

    remaining = sorted(atom.name for atom in ss_atoms if atom.name not in values_by_name)
    if remaining:
        raise ValueError(
            f"Perfect foresight simulation requires all steady-state variables to have analytic "
            f"solutions, but the following do not: {', '.join(remaining)}. Provide analytic steady-state values "
            f"in the Steady_State block of your GCN file."
        )

    return {atom: values_by_name[atom.name] for atom in sorted(ss_atoms, key=lambda atom: atom.name)}


def _hide_steady_state_symbols(
    equations: list[sp.Expr],
    ss_values: dict[sp.Symbol, sp.Expr | float],
) -> list[sp.Expr]:
    """Swap steady-state symbols for plain ones, so that timing classification sees only t-1, t, and t+1."""
    placeholders = {symbol: sp.Symbol(symbol.name) for symbol in ss_values}
    return [eq.xreplace(placeholders) for eq in equations]
