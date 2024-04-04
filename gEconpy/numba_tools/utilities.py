import re

from typing import Callable, Optional, Union

import numba as nb
import numpy as np
import sympy as sp

from numba.core import types
from numba.core.errors import TypingError
from sympy.printing.numpy import NumPyPrinter, _known_functions_numpy

_known_functions_numpy.update({"DiracDelta": lambda x: 0.0, "log": "log"})

# Pattern needs to hit "0," and "0]". but not "x0" or "6.0", and return the
# close-bracket (if any).
ZERO_PATTERN = re.compile(r"(?<![\.\w])0([ ,\]])")


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

    # Unpack optional type
    if isinstance(a, types.Optional):
        a = a.type
    if not isinstance(a, types.Array):
        msg = f"{prefix}.{func_name} only supported for array types"
        raise TypingError(msg, highlighting=False)
    if not a.ndim == 2:
        msg = f"{prefix}.{func_name} only supported on 2-D arrays."
        raise TypingError(msg, highlighting=False)
    if not isinstance(a.dtype, (types.Float, types.Complex)):
        msg = f"{prefix}.{func_name} only supported on " "float and complex arrays."
        raise TypingError(msg, highlighting=False)


@nb.njit
def direct_lyapunov_solution(A, B):
    lhs = np.kron(A, A.conj())
    lhs = np.eye(lhs.shape[0]) - lhs
    x = np.linalg.solve(lhs, B.flatten())

    return np.reshape(x, B.shape)


@nb.njit
def _lhp(alpha, beta):
    out = np.empty(alpha.shape, dtype=np.int32)
    nonzero = beta != 0
    # handles (x, y) = (0, 0) too
    out[~nonzero] = False
    out[nonzero] = np.real(alpha[nonzero] / beta[nonzero]) < 0.0
    return out


@nb.njit
def _rhp(alpha, beta):
    out = np.empty(alpha.shape, dtype=np.int32)
    nonzero = beta != 0
    # handles (x, y) = (0, 0) too
    out[~nonzero] = False
    out[nonzero] = np.real(alpha[nonzero] / beta[nonzero]) > 0.0
    return out


@nb.njit
def _iuc(alpha, beta):
    out = np.empty(alpha.shape, dtype=np.int32)
    nonzero = beta != 0
    # handles (x, y) = (0, 0) too
    out[~nonzero] = False
    out[nonzero] = np.abs(alpha[nonzero] / beta[nonzero]) < 1.0

    return out


@nb.njit
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


class NumbaFriendlyNumPyPrinter(NumPyPrinter):
    _kf = _known_functions_numpy

    def _print_Max(self, expr):
        # Use maximum instead of amax, because 1) we only expect scalars, and 2) numba doesn't accept amax
        return "{}({})".format(
            self._module_format(self._module + ".maximum"),
            ",".join(self._print(i) for i in expr.args),
        )

    def _print_Piecewise(self, expr):
        # Use the default python Piecewise instead of the numpy one -- looping with if conditions is faster in numba
        # anyway.
        result = []
        i = 0
        for arg in expr.args:
            e = arg.expr
            c = arg.cond
            if i == 0:
                result.append("(")
            result.append("(")
            result.append(self._print(sp.Float(e)))
            result.append(")")
            result.append(" if ")
            result.append(self._print(c))
            result.append(" else ")
            i += 1
        result = result[:-1]
        if result[-1] == "True":
            result = result[:-2]
            result.append(")")
        else:
            result.append(" else None)")
        return "".join(result)

    def _print_DiracDelta(self, expr):
        # The proper function should return infinity at one point, but the measure of that point is zero so this should
        # be fine. Pytensor defines grad(grad(max(0, x), x), x) to be zero everywhere.
        return "0.0"

    def _print_log(self, expr):
        return "{}({})".format(
            self._module_format(self._module + ".log"),
            ",".join(self._print(i) for i in expr.args),
        )

    def _print_exp(self, expr):
        return "{}({})".format(
            self._module_format(self._module + ".exp"),
            ",".join(self._print(i) for i in expr.args),
        )


def numba_lambdify(
    inputs: list[sp.Symbol],
    expr: Union[list[sp.Expr], sp.Matrix, list[sp.Matrix]],
    func_signature: Optional[str] = None,
    ravel_outputs=False,
) -> Callable:
    """
    Convert a sympy expression into a Numba-compiled function.  Unlike sp.lambdify, the resulting function can be
    pickled. In addition, common sub-expressions are gathered using sp.cse and assigned to local variables,
    giving a (very) modest performance boost. A signature can optionally be provided for numba.njit.

    Finally, the resulting function always returns a numpy array, rather than a list.

    Parameters
    ----------
    inputs: list of sympy.Symbol
        A list of "exogenous" variables. The distinction between "exogenous" and "enodgenous" is
        useful when passing the resulting function to a scipy.optimize optimizer. In this context, exogenous
        variables should be the choice varibles used to minimize the function.
    expr : list of sympy.Expr or sp.Matrix
        The sympy expression(s) to be converted. Expects a list of expressions (in the case that we're compiling a
        system to be stacked into a single output vector), a single matrix (which is returned as a single nd-array)
        or a list of matrices (which are returned as a list of nd-arrays)
    func_signature: str
        A numba function signature, passed to the numba.njit decorator on the generated function.
    ravel_outputs: bool, default False
        If true, all outputs of the jitted function will be raveled before they are returned. This is useful for
        removing size-1 dimensions from sympy vectors.

    Returns
    -------
    numba.types.function
        A Numba-compiled function equivalent to the input expression.

    Notes
    -----
    The function returned by this function is pickleable.
    """
    ZERO_PATTERN = re.compile(r"(?<![\.\w])0([ ,\]])")
    ZERO_ONE_INDEX_PATTERN = re.compile(r"((?<=\[)(([0,1])\.0)(?=\]))")
    FLOAT_SUBS = {
        sp.core.numbers.One(): sp.Float(1),
        sp.core.numbers.NegativeOne(): sp.Float(-1),
    }
    printer = NumbaFriendlyNumPyPrinter()

    if func_signature is None:
        decorator = "@nb.njit"
    else:
        decorator = f"@nb.njit({func_signature})"

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

                # Case 2c: It's a constant -- just pass it along unchanged.
                elif isinstance(item, (int, float)):
                    new_expr.append(item)
                else:
                    raise ValueError(f"Unexpected input type for expr: {expr}")

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
            code = printer.doprint(expr)

            delimiter = "]," if "]," in code else ","
            delimiter = ","
            code = code.split(delimiter)
            code = [" " * 8 + eq.strip() for eq in code]
            code = f"{delimiter}\n".join(code)
            code = code.replace("numpy.", "np.")

            # Handle conversion of 0 to 0.0
            code = re.sub(ZERO_PATTERN, r"0.0\g<1>", code)

            # Repair indexing -- we might have converted x[0] to x[0.0] or x[1] to x[1.0]
            code = re.sub(ZERO_ONE_INDEX_PATTERN, r"\g<3>", code)

            code_name = f"retval_{i}"
            retvals.append(code_name)
            code = f"    {code_name} = np.array(\n{code}\n    )"
            if ravel_outputs:
                code += ".ravel()"

            codes.append(code)
        code = "\n".join(codes)

    input_signature = ", ".join([getattr(x, "safe_name", x.name) for x in inputs])

    assignments = "\n".join(
        [f"    {x} = {printer.doprint(y).replace('numpy.', 'np.')}" for x, y in sub_dict]
    )
    assignments = re.sub(ZERO_ONE_INDEX_PATTERN, r"\g<3>", assignments)

    returns = f'[{",".join(retvals)}]' if len(retvals) > 1 else retvals[0]
    full_code = (
        f"{decorator}\ndef f({input_signature}):\n\n{assignments}\n\n{code}\n\n    return {returns}"
    )

    docstring = f"'''Automatically generated code:\n{full_code}'''"
    code = f"{decorator}\ndef f({input_signature}):\n    {docstring}\n\n{assignments}\n\n{code}\n\n    return {returns}"

    exec(code)
    return locals()["f"]
