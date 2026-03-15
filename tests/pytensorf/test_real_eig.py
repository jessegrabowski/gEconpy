import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest

from numpy.testing import assert_allclose
from pytensor.gradient import verify_grad

from gEconpy.pytensorf.real_eig import RealEig, real_eig


@pytest.fixture
def rng():
    return np.random.default_rng()


@pytest.fixture
def test_matrix(rng):
    return rng.standard_normal((4, 4))


class TestRealEig:
    def test_output_static_shapes(self):
        M = pt.dmatrix("M", shape=(4, 4))
        re, im = real_eig(M)
        assert re.type.shape == (4,)
        assert im.type.shape == (4,)

    def test_matches_numpy_eig(self, test_matrix):
        M = pt.dmatrix("M")
        re, im = real_eig(M)
        f = pytensor.function([M], [re, im])
        r, i = f(test_matrix)

        np_eigvals = np.linalg.eig(test_matrix)[0]
        np_eigvals = np_eigvals[np.argsort(np.abs(np_eigvals))]

        # Moduli must match exactly
        assert_allclose(np.sqrt(r**2 + i**2), np.abs(np_eigvals))
        # Real parts match (conjugate pairs have equal real parts)
        assert_allclose(r, np.real(np_eigvals))
        # Imaginary parts match up to sign within conjugate pairs
        assert_allclose(np.abs(i), np.abs(np.imag(np_eigvals)))

    @pytest.mark.parametrize("case", ["real", "imag", "combined"], ids=str)
    def test_grad(self, test_matrix, rng, case):
        def f(M):
            re, im = real_eig(M)
            if case == "real":
                return re.sum()
            if case == "imag":
                return im.sum()
            # case == 'combined'
            return (re + im).sum()

        verify_grad(f, [test_matrix], rng=rng)

    def test_numba_dispatch(self, test_matrix):
        """Numba dispatch produces the same moduli; conjugate pair ordering may differ."""
        M = pt.dmatrix("M")
        re, im = real_eig(M)

        f_py = pytensor.function([M], [re, im])
        f_numba = pytensor.function([M], [re, im], mode="NUMBA")

        r_py, i_py = f_py(test_matrix)
        r_nb, i_nb = f_numba(test_matrix)

        mod_py = np.sqrt(r_py**2 + i_py**2)
        mod_nb = np.sqrt(r_nb**2 + i_nb**2)
        assert_allclose(mod_nb, mod_py)

    def test_jax_dispatch(self, test_matrix):
        """JAX dispatch produces the same moduli; conjugate pair ordering may differ."""
        M = pt.dmatrix("M")
        re, im = real_eig(M)

        f_py = pytensor.function([M], [re, im])
        f_jax = pytensor.function([M], [re, im], mode="JAX")

        r_py, i_py = f_py(test_matrix)
        r_jax, i_jax = f_jax(test_matrix)

        mod_py = np.sqrt(r_py**2 + i_py**2)
        mod_jax = np.sqrt(r_jax**2 + i_jax**2)
        assert_allclose(mod_jax, mod_py)
