from numba import njit
import numpy as np
from numba.extending import get_cython_function_address
import ctypes

from numpy.typing._array_like import ArrayLike


# Datatype pointers to give to the cython LAPACK functions
_PTR = ctypes.POINTER

_dbl = ctypes.c_double
_int = ctypes.c_int64

_ptr_dbl = _PTR(_dbl)
_ptr_int = _PTR(_int)

# zgges is the complex QZ-decomposition
zgges_addr = get_cython_function_address('scipy.linalg.cython_lapack', 'zgges')
zgges_functype = ctypes.CFUNCTYPE(None,
                                  _ptr_int,  # JOBVSL
                                  _ptr_int,  # JOBVSR
                                  _ptr_int,  # SORT
                                  _ptr_int,  # SELCTG
                                  _ptr_int,  # N
                                  _ptr_dbl,  # A, complex
                                  _ptr_int,  # LDA
                                  _ptr_dbl,  # B, complex
                                  _ptr_int,  # LDB
                                  _ptr_int,  # SDIM
                                  _ptr_dbl,  # ALPHA, complex
                                  _ptr_dbl,  # BETA, complex
                                  _ptr_dbl,  # VSL, complex
                                  _ptr_int,  # LDVSL
                                  _ptr_dbl,  # VSR, complex
                                  _ptr_int,  # LDVSR
                                  _ptr_dbl,  # WORK, complex
                                  _ptr_int,  # LWORK
                                  _ptr_dbl,  # RWORK
                                  _ptr_int,  # BWORK
                                  _ptr_int)  # INFO
zgges_fn = zgges_functype(zgges_addr)

# ztgsen re-organizes the eigenvalues in the generalized Schur matrices to be in some order (stable to unstable, in our
# case here, although this code could be generalized)
ztgsen_addr = get_cython_function_address('scipy.linalg.cython_lapack', 'ztgsen')
ztgsen_functype = ctypes.CFUNCTYPE(None,
                                   _ptr_int,  # IJOB
                                   _ptr_int,  # WANTQ
                                   _ptr_int,  # WANTZ
                                   _ptr_int,  # SELECT
                                   _ptr_int,  # N
                                   _ptr_dbl,  # A
                                   _ptr_int,  # LDA
                                   _ptr_dbl,  # B
                                   _ptr_int,  # LDB
                                   _ptr_dbl,  # ALPHA
                                   _ptr_dbl,  # BETA
                                   _ptr_dbl,  # Q
                                   _ptr_int,  # LDQ
                                   _ptr_dbl,  # Z
                                   _ptr_int,  # LDZ
                                   _ptr_int,  # M
                                   _ptr_dbl,  # PL
                                   _ptr_dbl,  # PR
                                   _ptr_dbl,  # DIF
                                   _ptr_dbl,  # WORK
                                   _ptr_int,  # LWORK
                                   _ptr_int,  # IWORK
                                   _ptr_int)  # LIWORK

ztgsen_fn = ztgsen_functype(ztgsen_addr)

@njit
def numba_block_diagonal(nd_array):
    n, rows, cols = nd_array.shape

    out = np.zeros((n * rows, n * cols))

    r, c = 0, 0
    for i, (rr, cc) in enumerate([(rows, cols)] * n):
        out[r:r + rr, c:c + cc] = nd_array[i]
        r += rr
        c += cc
    return out

@njit
def _ouc(alpha: ArrayLike, beta: ArrayLike):
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
    alpha_zero = (alpha == 0)
    beta_zero = (beta == 0)

    out[alpha_zero & beta_zero] = False
    out[~alpha_zero & beta_zero] = True
    out[~beta_zero] = (np.abs(alpha[~beta_zero]/beta[~beta_zero]) > 1.0)

    return out

@njit
def numba_zgges(x, y):
    _M, _N = x.shape

    A = x
    B = y

    JOBVSL = np.array([ord('V')], dtype=np.int64)
    JOBVSR = np.array([ord('V')], dtype=np.int64)
    SORT = np.array([ord('N')], dtype=np.int64)
    SELCTG = np.empty(1, dtype=np.int64)

    N = np.array(_N, dtype=np.int64)
    LDA = np.array(_N, dtype=np.int64)
    LDB = np.array(_N, dtype=np.int64)
    SDIM = np.array(0, dtype=np.int64) # out

    ALPHA = np.empty(_N, dtype=np.complex128) # out
    BETA = np.empty(_N, dtype=np.complex128) # out
    LDVSL = np.array(_N, dtype=np.int64)
    VSL = np.empty((_N, _N), dtype=np.complex128) # out
    LDVSR = np.array(_N, dtype=np.int64)
    VSR = np.empty((_N, _N), dtype=np.complex128) # out

    WORK = np.empty((1,), dtype=np.complex128) #out
    LWORK = np.array(-1, dtype=np.int64)
    RWORK = np.empty(8*_N, dtype=np.float64)
    BWORK = np.empty(_N, dtype=np.int64)
    INFO = np.empty(1, dtype=np.int64)

    zgges_fn(JOBVSL.ctypes,
             JOBVSR.ctypes,
             SORT.ctypes,
             SELCTG.ctypes,
             N.ctypes,
             A.view(np.float64).ctypes,
             LDA.ctypes,
             B.view(np.float64).ctypes,
             LDB.ctypes,
             SDIM.ctypes,
             ALPHA.view(np.float64).ctypes,
             BETA.view(np.float64).ctypes,
             VSL.view(np.float64).ctypes,
             LDVSL.ctypes,
             VSR.view(np.float64).ctypes,
             LDVSR.ctypes,
             WORK.view(np.float64).ctypes,
             LWORK.ctypes,
             RWORK.ctypes,
             BWORK.ctypes,
             INFO.ctypes)

    print("Calculated workspace size as", WORK[0])
    WS_SIZE = np.int64(WORK[0].real)
    LWORK = np.array(WS_SIZE, np.int64)
    WORK = np.empty(WS_SIZE, dtype=np.complex128)
    zgges_fn(JOBVSL.ctypes,
             JOBVSR.ctypes,
             SORT.ctypes,
             SELCTG.ctypes,
             N.ctypes,
             A.view(np.float64).ctypes,
             LDA.ctypes,
             B.view(np.float64).ctypes,
             LDB.ctypes,
             SDIM.ctypes,
             ALPHA.view(np.float64).ctypes,
             BETA.view(np.float64).ctypes,
             VSL.view(np.float64).ctypes,
             LDVSL.ctypes,
             VSR.view(np.float64).ctypes,
             LDVSR.ctypes,
             WORK.view(np.float64).ctypes,
             LWORK.ctypes,
             RWORK.ctypes,
             BWORK.ctypes,
             INFO.ctypes)

    # The LAPACK function also returns SDIM, WORK, BWORK, but I don't need them here.
    return A, B, ALPHA, BETA, VSL.T, VSR.T, INFO


@njit
def numba_ztgsen(A, B, ALPHA, BETA, Q, Z, SELECT):
    _, _N = A.shape

    # These values come from the scipy defaults
    IJOB = np.array(0, np.int32)
    WANTQ = np.array(1, np.int32)
    WANTZ = np.array(1, np.int32)

    # Shapes, only consider the case of 2 square matrices, so everything is equal
    N = np.array(_N, np.int32)
    LDA = np.array(_N, np.int32)
    LDB = np.array(_N, np.int32)
    LDQ = np.array(_N, np.int32)
    LDZ = np.array(_N, np.int32)
    M = np.array(_N, np.int32)

    # These are computed by ztgsen, so the defaults shouldn't matter (i think?)
    PL = np.array(1, np.float64)
    PR = np.array(1, np.float64)
    DIF = np.array([1.0, 1.0], np.float64)

    WORK = np.empty((1,), dtype=np.complex128) #out
    LWORK = np.array(1, dtype=np.int32)
    IWORK = np.empty(_N, dtype=np.int32)
    LIWORK = np.array(1, dtype=np.int32)

    ztgsen_fn(IJOB.ctypes,
              WANTQ.ctypes,
              WANTZ.ctypes,
              SELECT.view(np.int32).ctypes,
              N.ctypes,
              A.view(np.float64).ctypes,
              LDA.ctypes,
              B.view(np.float64).ctypes,
              LDB.ctypes,
              ALPHA.view(np.float64).ctypes,
              BETA.view(np.float64).ctypes,
              Q.view(np.float64).ctypes,
              LDQ.ctypes,
              Z.view(np.float64).ctypes,
              LDZ.ctypes,
              M.ctypes,
              PL.ctypes,
              PR.ctypes,
              DIF.ctypes,
              WORK.view(np.float64).ctypes,
              LWORK.ctypes,
              IWORK.ctypes,
              LIWORK.ctypes)

    print("Calculated workspace size as", WORK[0].real)
    WS_SIZE = np.int32(WORK[0].real)
    LWORK = np.array(WS_SIZE, np.int32)
    WORK = np.empty(WS_SIZE, dtype=np.complex128)



