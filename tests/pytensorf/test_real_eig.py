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

    def test_rejects_non_matrix(self):
        with pytest.raises(ValueError, match="requires a square matrix"):
            RealEig()(pt.dvector("v"))

    def test_matches_numpy_eig(self, test_matrix):
        M = pt.dmatrix("M")
        re, im = real_eig(M)
        f = pytensor.function([M], [re, im])
        r, i = f(test_matrix)

        np_eigvals = np.linalg.eig(test_matrix)[0]
        np_eigvals = np_eigvals[np.argsort(np.abs(np_eigvals))]

        assert_allclose(np.sqrt(r**2 + i**2), np.abs(np_eigvals), atol=1e-12)
        assert_allclose(r, np.real(np_eigvals), atol=1e-12)
        # Imaginary parts match up to sign within a conjugate pair, and the sort may order the pair either way.
        assert_allclose(np.abs(i), np.abs(np.imag(np_eigvals)), atol=1e-12)

    @pytest.mark.parametrize(
        "reduce_outputs",
        [lambda re, _im: re.sum(), lambda _re, im: im.sum(), lambda re, im: (re + im).sum()],
        ids=["real", "imag", "combined"],
    )
    def test_grad(self, test_matrix, rng, reduce_outputs):
        def f(M):
            return reduce_outputs(*real_eig(M))

        verify_grad(f, [test_matrix], rng=rng)

    @pytest.mark.parametrize("mode", ["NUMBA", "JAX"])
    def test_backend_dispatch_matches_python(self, test_matrix, mode):
        """The backends agree on the moduli. Conjugate pair ordering may differ between them."""
        M = pt.dmatrix("M")
        re, im = real_eig(M)

        f_py = pytensor.function([M], [re, im])
        f_backend = pytensor.function([M], [re, im], mode=mode)

        r_py, i_py = f_py(test_matrix)
        r_backend, i_backend = f_backend(test_matrix)

        assert_allclose(np.sqrt(r_backend**2 + i_backend**2), np.sqrt(r_py**2 + i_py**2))
