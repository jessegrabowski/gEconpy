import re

from collections.abc import Callable

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
    if not isinstance(a.dtype, types.Float | types.Complex):
        msg = f"{prefix}.{func_name} only supported on float and complex arrays."
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
