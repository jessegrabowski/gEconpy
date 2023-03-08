from typing import Callable, List, Optional, Union

import numpy as np
import sympy as sp
from numba import njit
from numba.core import types
from numba.core.errors import TypingError


def _get_underlying_float(dtype):
    s_dtype = str(dtype)
    out_type = s_dtype
    if s_dtype == "complex64":
        out_type = "float32"
    elif s_dtype == "complex128":
        out_type = "float64"

    return np.dtype(out_type)


def _check_scipy_linalg_matrix(a, func_name):
    prefix = "scipy.linalg"
    interp = (prefix, func_name)
    # Unpack optional type
    if isinstance(a, types.Optional):
        a = a.type
    if not isinstance(a, types.Array):
        msg = "%s.%s() only supported for array types" % interp
        raise TypingError(msg, highlighting=False)
    if not a.ndim == 2:
        msg = "%s.%s() only supported on 2-D arrays." % interp
        raise TypingError(msg, highlighting=False)
    if not isinstance(a.dtype, (types.Float, types.Complex)):
        msg = "%s.%s() only supported on " "float and complex arrays." % interp
        raise TypingError(msg, highlighting=False)


@njit
def direct_lyapunov_solution(A, B):
    lhs = np.kron(A, A.conj())
    lhs = np.eye(lhs.shape[0]) - lhs
    x = np.linalg.solve(lhs, B.flatten())

    return np.reshape(x, B.shape)


@njit
def _lhp(alpha, beta):
    out = np.empty(alpha.shape, dtype=np.int32)
    nonzero = beta != 0
    # handles (x, y) = (0, 0) too
    out[~nonzero] = False
    out[nonzero] = np.real(alpha[nonzero] / beta[nonzero]) < 0.0
    return out


@njit
def _rhp(alpha, beta):
    out = np.empty(alpha.shape, dtype=np.int32)
    nonzero = beta != 0
    # handles (x, y) = (0, 0) too
    out[~nonzero] = False
    out[nonzero] = np.real(alpha[nonzero] / beta[nonzero]) > 0.0
    return out


@njit
def _iuc(alpha, beta):
    out = np.empty(alpha.shape, dtype=np.int32)
    nonzero = beta != 0
    # handles (x, y) = (0, 0) too
    out[~nonzero] = False
    out[nonzero] = np.abs(alpha[nonzero] / beta[nonzero]) < 1.0

    return out


@njit
def _ouc(alpha, beta):
    """
    Jit-aware version of the function scipy.linalg._decomp_qz._ouc, creates the mask needed for ztgsen to sort
    eigenvalues from stable to unstable.

    Parameters
    ----------
    alpha: Array, complex
        alpha vector, as returned by zgges
    beta: Array, complex
        beta vector, as return by zgges
    Returns
    -------
    out: Array, bool
        Boolean mask indicating which eigenvalues are unstable
    """

    out = np.empty(alpha.shape, dtype=np.int32)
    alpha_zero = alpha == 0
    beta_zero = beta == 0

    out[alpha_zero & beta_zero] = False
    out[~alpha_zero & beta_zero] = True
    out[~beta_zero] = np.abs(alpha[~beta_zero] / beta[~beta_zero]) > 1.0

    return out


def numba_lambdify(
    exog_vars: List[sp.Symbol],
    expr: Union[List[sp.Expr], sp.Matrix, List[sp.Matrix]],
    endog_vars: Optional[List[sp.Symbol]] = None,
    func_signature: Optional[str] = None,
) -> Callable:
    """
    Convert a sympy expression into a Numba-compiled function.  Unlike sp.lambdify, the resulting function can be
    pickled. In addition, common sub-expressions are gathered using sp.cse and assigned to local variables,
    giving a (very) modest performance boost. A signature can optionally be provided for numba.njit.

    Finally, the resulting function always returns a numpy array, rather than a list.

    Parameters
    ----------
    exog_vars: list of sympy.Symbol
        A list of "exogenous" variables. The distinction between "exogenous" and "enodgenous" is
        useful when passing the resulting function to a scipy.optimize optimizer. In this context, exogenous
        variables should be the choice varibles used to minimize the function.
    expr : list of sympy.Expr or sp.Matrix
        The sympy expression(s) to be converted. Expects a list of expressions (in the case that we're compiling a
        system to be stacked into a single output vector), a single matrix (which is returned as a single nd-array)
        or a list of matrices (which are returned as a list of nd-arrays)
    endog_vars : Optional, list of sympy.Symbol
        A list of "exogenous" variables, passed as a second argument to the function.
    func_signature: str
        A numba function signature, passed to the numba.njit decorator on the generated function.

    Returns
    -------
    numba.types.function
        A Numba-compiled function equivalent to the input expression.

    Notes
    -----
    The function returned by this function is pickleable.
    """

    FLOAT_SUBS = {
        sp.core.numbers.Zero(): sp.Float(0),
        sp.core.numbers.One(): sp.Float(1),
        sp.core.numbers.NegativeOne(): sp.Float(-1),
    }

    if func_signature is None:
        decorator = "@nb.njit"
    else:
        decorator = f"@nb.njit({func_signature})"

    len_checks = []
    len_checks.append(
        f'    assert len(exog_inputs) == {len(exog_vars)}, "Expected {len(exog_vars)} exog_inputs"'
    )
    if endog_vars is not None:
        len_checks.append(
            f'    assert len(endog_inputs) == {len(endog_vars)}, "Expected {len(endog_vars)} exog_inputs"'
        )
    len_checks = "\n".join(len_checks)

    # Special case: expr is [[]]. This can occur if no user-defined steady-state values were provided.
    # It shouldn't happen otherwise.
    if expr == [[]]:
        sub_dict = ()
        code = ""
        retvals = ["[None]"]

    else:
        # Need to make the float substitutions so that numba can correctly interpret everything, but we have to handle
        # several cases:
        # Case 1: expr is just a single Sympy thing
        if isinstance(expr, (sp.Matrix, sp.Expr)):
            expr = expr.subs(FLOAT_SUBS)

        # Case 2: expr is a list. Items in the list are either lists of expressions (systems of equations),
        # single equations, or matrices.
        elif isinstance(expr, list):
            new_expr = []
            for item in expr:
                # Case 2a: It's a simple list of sympy things
                if isinstance(item, (sp.Matrix, sp.Expr)):
                    new_expr.append(item.subs(FLOAT_SUBS))
                # Case 2b: It's a system of equations, List[List[sp.Expr]]
                elif isinstance(item, list):
                    if all([isinstance(x, (sp.Matrix, sp.Expr)) for x in item]):
                        new_expr.append([x.subs(FLOAT_SUBS) for x in item])
                    else:
                        raise ValueError("Unexpected input type for expr")
            expr = new_expr
        else:
            raise ValueError("Unexpected input type for expr")

        sub_dict, expr = sp.cse(expr)

        # Converting matrices to a list of lists is convenient because NumPyPrinter() won't wrap them in np.array
        exprs = []
        for ex in expr:
            if hasattr(ex, "tolist"):
                exprs.append(ex.tolist())
            else:
                exprs.append(ex)

        codes = []
        retvals = []
        for i, expr in enumerate(exprs):
            code = sp.printing.numpy.NumPyPrinter().doprint(expr)
            delimiter = "]," if "]," in code else ","
            code = code.split(delimiter)
            code = [" " * 8 + eq.strip() for eq in code]
            code = f"{delimiter}\n".join(code)
            code = code.replace("numpy.", "np.")

            code_name = f"retval_{i}"
            retvals.append(code_name)
            code = f"    {code_name} = np.array(\n{code}\n    )"
            codes.append(code)
        code = "\n".join(codes)

    input_signature = "exog_inputs"
    unpacked_inputs = "\n".join(
        [f"    {x.name} = exog_inputs[{i}]" for i, x in enumerate(exog_vars)]
    )
    if endog_vars is not None:
        input_signature += ", endog_inputs"
        exog_unpacked = "\n".join(
            [f"    {x.name} = endog_inputs[{i}]" for i, x in enumerate(endog_vars)]
        )
        unpacked_inputs += "\n" + exog_unpacked

    assignments = "\n".join(
        [
            f"    {x} = {sp.printing.numpy.NumPyPrinter().doprint(y).replace('numpy.', 'np.')}"
            for x, y in sub_dict
        ]
    )
    returns = f'[{",".join(retvals)}]' if len(retvals) > 1 else retvals[0]
    full_code = f"{decorator}\ndef f({input_signature}):\n{unpacked_inputs}\n\n{assignments}\n\n{code}\n\n    return {returns}"

    docstring = f"'''Automatically generated code:\n{full_code}'''"
    code = f"{decorator}\ndef f({input_signature}):\n    {docstring}\n{len_checks}\n{unpacked_inputs}\n\n{assignments}\n\n{code}\n\n    return {returns}"

    exec(code)
    return locals()["f"]
