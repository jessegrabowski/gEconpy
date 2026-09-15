from collections.abc import Callable
from functools import wraps

import numpy as np
import pytensor
import sympy as sp

from pytensor import tensor as pt
from pytensor.graph.replace import graph_replace
from pytensor.graph.traversal import explicit_graph_inputs
from pytensor.tensor import TensorVariable
from sympytensor import as_tensor

from gEconpy.classes.containers import SteadyStateResults, SymbolDictionary
from gEconpy.classes.time_aware_symbol import TimeAwareSymbol
from gEconpy.pytensorf.compile import compile_pytensor_function
from gEconpy.utilities import eq_to_ss

_CSE_TEMPORARY_PREFIX = "__cse_tmp_"


def compile_function(
    inputs: list[sp.Symbol],
    outputs: list[sp.Symbol | sp.Expr] | sp.MutableDenseMatrix,
    cache: dict | None = None,
    stack_return: bool = False,
    pop_return: bool = False,
    return_symbolic: bool = False,
    **kwargs,
) -> tuple[Callable | TensorVariable | list[TensorVariable], dict]:
    """
    Compile sympy expressions into a PyTensor function of the given symbols.

    Parameters
    ----------
    inputs : list of sympy Symbol
        Inputs of the function.
    outputs : list of sympy expression, or sympy Matrix
        Outputs of the function.
    cache : dict, optional
        Sympytensor cache mapping cache keys to PyTensor variables. Pass the same cache to every compilation that
        should share input variables. Default is a new empty cache.
    stack_return : bool, optional
        Stack the outputs into a single array. Default is False.
    pop_return : bool, optional
        Return the single output directly instead of a one-element list. Default is False.
    return_symbolic : bool, optional
        Return the PyTensor output graph instead of a compiled function. Default is False.
    **kwargs
        Forwarded to :func:`~gEconpy.pytensorf.compile.compile_pytensor_function`.

    Returns
    -------
    f : callable or TensorVariable
        The compiled function, or the output graph when ``return_symbolic`` is True.
    cache : dict
        The cache, extended with every variable created during conversion.
    """
    return compile_to_pytensor_function(
        inputs,
        outputs,
        cache,
        stack_return=stack_return,
        pop_return=pop_return,
        return_symbolic=return_symbolic,
        **kwargs,
    )


def compile_to_pytensor_function(
    inputs: list[sp.Symbol],
    outputs: list[sp.Symbol | sp.Expr] | sp.MutableDenseMatrix,
    cache: dict | None,
    stack_return: bool,
    pop_return: bool,
    return_symbolic: bool,
    **kwargs,
) -> tuple[Callable | TensorVariable | list[TensorVariable], dict]:
    """
    Convert sympy expressions to a PyTensor graph and compile it with :func:`pytensor.function`.

    Parameters
    ----------
    inputs : list of sympy Symbol
        Inputs of the function.
    outputs : list of sympy expression, or sympy Matrix
        Outputs of the function.
    cache : dict or None
        Sympytensor cache mapping cache keys to PyTensor variables, or None for a new empty cache.
    stack_return : bool
        Stack the outputs into a single array.
    pop_return : bool
        Return the single output directly instead of a one-element list.
    return_symbolic : bool
        Return the PyTensor output graph instead of a compiled function.
    **kwargs
        Forwarded to :func:`~gEconpy.pytensorf.compile.compile_pytensor_function`. Ignored when ``return_symbolic``
        is True.

    Returns
    -------
    f : callable or TensorVariable
        The compiled function, or the output graph when ``return_symbolic`` is True.
    cache : dict
        The cache, extended with every variable created during conversion.
    """
    input_pt, output_pt, cache = sympy_to_pytensor(inputs, outputs, cache)

    if stack_return:
        output_pt = pt.stack(output_pt)
    if pop_return and isinstance(output_pt, list) and len(output_pt) == 1:
        output_pt = output_pt[0]

    if return_symbolic:
        return output_pt, cache

    f = compile_pytensor_function(input_pt, output_pt, **kwargs)

    return f, cache


def sympy_to_pytensor(
    inputs: list[sp.Symbol],
    outputs: list[sp.Symbol | sp.Expr] | sp.MutableDenseMatrix,
    cache: dict | None = None,
    cse: bool = False,
) -> tuple[list[TensorVariable], list[TensorVariable], dict]:
    """
    Convert sympy symbols and expressions to PyTensor variables.

    Parameters
    ----------
    inputs : list of sympy Symbol
        Input symbols.
    outputs : list of sympy expression, or sympy Matrix
        Output expressions.
    cache : dict, optional
        Sympytensor cache mapping cache keys to PyTensor variables. Pass the same cache to every conversion that
        should share variables. Default is a new empty cache.
    cse : bool, optional
        Run :func:`sympy.cse` over the outputs before converting them. This shrinks the forward graph but inflates
        gradient compilation, so enable it only for forward-only compiles. Default is False.

    Returns
    -------
    input_nodes : list of TensorVariable
        PyTensor variables for ``inputs``.
    output_nodes : list of TensorVariable
        PyTensor graphs for ``outputs``.
    cache : dict
        The cache, extended with every variable created during conversion.
    """
    cache = {} if cache is None else cache
    outputs = [outputs] if not isinstance(outputs, list) else outputs
    input_nodes = [as_tensor(x, cache) for x in inputs]

    cse_candidates = [(i, output) for i, output in enumerate(outputs) if isinstance(output, sp.Basic)]
    if not cse or len(cse_candidates) < 2:
        output_nodes = [output_to_tensor(x, cache) for x in outputs]
        return input_nodes, output_nodes, cache

    indices, expressions = zip(*cse_candidates, strict=True)
    reduced = _plant_cse_intermediates(list(expressions), cache)
    reduced_by_index = dict(zip(indices, reduced, strict=True))
    output_nodes = [output_to_tensor(reduced_by_index.get(i, x), cache) for i, x in enumerate(outputs)]

    return input_nodes, output_nodes, cache


def output_to_tensor(x: sp.Basic | int | float, cache: dict) -> TensorVariable:
    """
    Convert one sympy expression to a PyTensor variable, mapping plain numbers to constants.

    Parameters
    ----------
    x : sympy expression, int, or float
        Expression to convert.
    cache : dict
        Sympytensor cache mapping cache keys to PyTensor variables.

    Returns
    -------
    x_pt : TensorVariable
        PyTensor variable for ``x``.
    """
    if isinstance(x, int | float | sp.Float | sp.Integer):
        return pt.constant(x, dtype=pytensor.config.floatX)

    return as_tensor(x, cache)


def build_symbolic_jacobians(
    specs: list[tuple[list[sp.Expr], list[sp.Symbol]]],
    cache: dict,
    to_ss: bool = False,
    shocks: list[TimeAwareSymbol] | None = None,
) -> list[TensorVariable]:
    """
    Build several dense Jacobians by symbolic differentiation, sharing one :func:`sympy.cse` pass.

    Parameters
    ----------
    specs : list of (list of sympy expression, list of sympy Symbol)
        ``(equations, wrt)`` pairs. One dense Jacobian is built per pair.
    cache : dict
        Sympytensor cache mapping cache keys to PyTensor variables.
    to_ss : bool, optional
        Evaluate each entry at the steady state: time-indexed variables map to their steady-state symbols and
        shocks map to zero. Default is False.
    shocks : list of TimeAwareSymbol, optional
        Shocks to zero out when ``to_ss`` is True. Default is None.

    Returns
    -------
    jacobians : list of TensorVariable
        One PyTensor matrix per spec, of shape ``(len(equations), len(wrt))``.
    """
    shapes = [(len(equations), len(wrt)) for equations, wrt in specs]
    entries = [
        eq_to_ss(eq.diff(symbol), shocks=shocks) if to_ss else eq.diff(symbol)
        for equations, wrt in specs
        for eq in equations
        for symbol in wrt
    ]

    n_symbolic = sum(1 for entry in entries if isinstance(entry, sp.Basic) and not entry.is_number)
    if n_symbolic > 1:
        entries = _plant_cse_intermediates(entries, cache)

    jacobians = []
    offset = 0
    for n_eq, n_wrt in shapes:
        if n_eq == 0 or n_wrt == 0:
            jacobians.append(pt.zeros((n_eq, n_wrt)))
            continue

        rows = [entries[offset + i * n_wrt : offset + (i + 1) * n_wrt] for i in range(n_eq)]
        offset += n_eq * n_wrt
        jacobians.append(as_tensor(sp.ImmutableMatrix(rows), cache))

    return jacobians


def build_symbolic_jacobian(
    equations: list[sp.Expr],
    wrt: list[sp.Symbol],
    cache: dict,
    to_ss: bool = False,
    shocks: list[TimeAwareSymbol] | None = None,
) -> TensorVariable:
    """
    Build one dense Jacobian by symbolic differentiation.

    Parameters
    ----------
    equations : list of sympy expression
        Equations to differentiate, one per Jacobian row.
    wrt : list of sympy Symbol
        Symbols to differentiate with respect to, one per Jacobian column.
    cache : dict
        Sympytensor cache mapping cache keys to PyTensor variables.
    to_ss : bool, optional
        Evaluate each entry at the steady state. Default is False.
    shocks : list of TimeAwareSymbol, optional
        Shocks to zero out when ``to_ss`` is True. Default is None.

    Returns
    -------
    jacobian : TensorVariable
        PyTensor matrix of shape ``(len(equations), len(wrt))``.
    """
    return build_symbolic_jacobians([(equations, wrt)], cache, to_ss=to_ss, shocks=shocks)[0]


def make_cache_key(name: str, cls: type[sp.Symbol]) -> tuple:
    """
    Build the sympytensor cache key for a scalar float symbol.

    Parameters
    ----------
    name : str
        Name of the sympy symbol.
    cls : type
        Class of the sympy symbol, such as ``sp.Symbol`` or :class:`~gEconpy.classes.time_aware_symbol.TimeAwareSymbol`.

    Returns
    -------
    key : tuple
        The key sympytensor uses for this symbol.
    """
    return name, cls, (), "floatX", ()


def make_return_dict_and_update_cache(
    input_symbols: list[sp.Symbol],
    output_tensors: TensorVariable | list[TensorVariable],
    cache: dict[tuple, TensorVariable],
    cls: type[sp.Symbol] | None = None,
) -> tuple[dict[TensorVariable, TensorVariable], dict]:
    """
    Map each input symbol's PyTensor variable to its output graph, creating cache entries for missing symbols.

    Parameters
    ----------
    input_symbols : list of sympy Symbol
        Symbols whose values the output graphs compute.
    output_tensors : TensorVariable or list of TensorVariable
        Output graphs, in the same order as ``input_symbols``.
    cache : dict
        Sympytensor cache mapping cache keys to PyTensor variables.
    cls : type, optional
        Class of the sympy symbols, used to build their cache keys. Default is ``sp.Symbol``.

    Returns
    -------
    out_dict : dict
        Mapping from each symbol's PyTensor variable to its output graph.
    cache : dict
        The cache, extended with a variable for every symbol it did not already hold.
    """
    if cls is None:
        cls = sp.Symbol

    out_dict = {}
    for symbol, value in zip(input_symbols, output_tensors, strict=False):
        cache_key = make_cache_key(symbol.name, cls)
        if cache_key not in cache:
            cache[cache_key] = pt.scalar(name=symbol.name, dtype="floatX")

        out_dict[cache[cache_key]] = value

    return out_dict, cache


def dictionary_return_wrapper(f: Callable, outputs: list[sp.Symbol]) -> Callable:
    """
    Wrap a function so that it returns a string-keyed :class:`~gEconpy.classes.containers.SteadyStateResults`.

    Parameters
    ----------
    f : callable
        Function returning a sequence of values.
    outputs : list of sympy Symbol
        Symbols naming the outputs, in the order ``f`` returns them.

    Returns
    -------
    inner : callable
        The wrapped function.
    """

    @wraps(f, updated=())
    def inner(*args, **kwargs):
        values = f(*args, **kwargs)
        return SteadyStateResults(zip(outputs, values, strict=False)).to_string()

    return inner


def compile_for_scipy(
    outputs: TensorVariable | list[TensorVariable],
    mode: str | None = None,
) -> Callable:
    """
    Compile a PyTensor graph into a callable that takes every input by keyword and ignores unknown keywords.

    Use it for one-off evaluations. For hot-loop scipy calls, use :func:`~gEconpy.model.compile.pack_and_compile`.

    Parameters
    ----------
    outputs : TensorVariable or list of TensorVariable
        Graph outputs to compile.
    mode : str, optional
        PyTensor compilation mode. Default is None, which uses the PyTensor default mode.

    Returns
    -------
    wrapper : callable
        Compiled function accepting its inputs as keyword arguments.
    """
    inputs = list(explicit_graph_inputs(outputs))
    f = compile_pytensor_function(inputs, outputs, mode=mode, on_unused_input="ignore")
    accepted_names = frozenset(inp.name for inp in f.input_storage)

    @wraps(f, updated=())
    def wrapper(*args, **kwargs):
        return f(*args, **{k: v for k, v in kwargs.items() if k in accepted_names})

    return wrapper


def pack_and_compile(
    outputs: TensorVariable | list[TensorVariable],
    ss_nodes: list[TensorVariable],
    param_dict: SymbolDictionary | None = None,
    mode: str | None = None,
) -> Callable:
    """
    Compile a graph whose steady-state inputs are read from one flat vector ``x_flat``.

    Only steady-state nodes that are graph inputs are replaced by ``x_flat[i]``. With ``param_dict`` given, the
    parameter values are frozen into the closure and the returned function has signature ``f(x_flat)``. Without it,
    the signature is ``f(x_flat, *param_args)``.

    Parameters
    ----------
    outputs : TensorVariable or list of TensorVariable
        Graph outputs to compile.
    ss_nodes : list of TensorVariable
        Every scalar steady-state variable node, in the positional order of ``x_flat``.
    param_dict : SymbolDictionary, optional
        Parameter values to freeze into the closure. Default is None.
    mode : str, optional
        PyTensor compilation mode. Default is None, which uses the PyTensor default mode.

    Returns
    -------
    f : callable
        Compiled function of ``x_flat``.
    """
    graph_inputs = set(explicit_graph_inputs(outputs))
    active_indices = [i for i, node in enumerate(ss_nodes) if node in graph_inputs]
    active_nodes = [ss_nodes[i] for i in active_indices]

    if not active_nodes:
        return _compile_without_ss_inputs(outputs, param_dict, mode)

    x_flat = pt.dvector("x_flat")
    replacements = {node: x_flat[i] for i, node in enumerate(active_nodes)}
    new_outputs = graph_replace(outputs, replacements, strict=False)

    active_set = set(active_nodes)
    param_inputs = [inp for inp in explicit_graph_inputs(outputs) if inp not in active_set]
    inner = compile_pytensor_function([x_flat, *param_inputs], new_outputs, mode=mode, on_unused_input="ignore")

    if len(active_nodes) == len(ss_nodes):

        def select_active(x_flat: np.ndarray) -> np.ndarray:
            return x_flat

    else:
        active_index_array = np.array(active_indices)

        def select_active(x_flat: np.ndarray) -> np.ndarray:
            return x_flat[active_index_array]

    if param_dict is None:

        def f(x_flat: np.ndarray, *args) -> np.ndarray:
            return inner(select_active(x_flat), *args)

        return f

    frozen_args = tuple(float(param_dict[inp.name]) for inp in param_inputs)

    def f_frozen(x_flat: np.ndarray) -> np.ndarray:
        return inner(select_active(x_flat), *frozen_args)

    return f_frozen


def _compile_without_ss_inputs(
    outputs: TensorVariable | list[TensorVariable],
    param_dict: SymbolDictionary | None,
    mode: str | None,
) -> Callable:
    inner = compile_pytensor_function(
        list(explicit_graph_inputs(outputs)), outputs, mode=mode, on_unused_input="ignore"
    )

    if param_dict is None:

        def f(_x_flat: np.ndarray, *args) -> np.ndarray:
            return inner(*args)

        return f

    accepted = frozenset(inp.name for inp in inner.input_storage)
    frozen_params = {name: value for name, value in param_dict.items() if name in accepted}

    def f_frozen(_x_flat: np.ndarray) -> np.ndarray:
        return inner(**frozen_params)

    return f_frozen


def _plant_cse_intermediates(expressions: list[sp.Basic], cache: dict) -> list[sp.Basic]:
    """
    Run :func:`sympy.cse` over ``expressions`` and convert each intermediate into the cache.

    The intermediates get a dunder-prefixed name so that neither :func:`sympy.cse` nor the shared sympytensor cache can
    confuse them with a model variable or parameter. Each is converted in topological order and planted in the cache
    under sympytensor's key shape, so the reduced expressions resolve to the planted graphs when converted.
    """
    substitutions, reduced = sp.cse(
        expressions, symbols=sp.numbered_symbols(_CSE_TEMPORARY_PREFIX), optimizations="basic"
    )
    for intermediate, definition in substitutions:
        cache[make_cache_key(intermediate.name, type(intermediate))] = as_tensor(definition, cache)

    return reduced
