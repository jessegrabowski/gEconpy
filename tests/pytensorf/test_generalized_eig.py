import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest

from numpy.testing import assert_allclose
from pytensor.gradient import verify_grad
from pytensor.tensor.linalg.decomposition.schur import QZ
from scipy import linalg

from gEconpy.pytensorf.generalized_eig import SchurEigenvectors, generalized_eig

TOL = 1e-8


@pytest.fixture
def rng():
    return np.random.default_rng(1234)


@pytest.fixture
def pencil(rng):
    return rng.standard_normal((5, 5)), rng.standard_normal((5, 5)) + 3 * np.eye(5)


@pytest.fixture
def singular_pencil(pencil):
    a, b = pencil
    b = b.copy()
    b[:, 0] = 0.0
    return a, b


@pytest.fixture
def repeated_eigenvalue_pencil(rng):
    """Build a pencil whose eigenvalue 0.5 is semisimple with multiplicity two."""
    P = rng.standard_normal((4, 4)) + 4 * np.eye(4)
    b = np.eye(4) + 0.1 * rng.standard_normal((4, 4))
    return b @ P @ np.diag([0.5, 0.5, 2.0, -1.0]) @ np.linalg.inv(P), b


@pytest.fixture
def degenerate_pencil(pencil):
    a, b = (matrix.copy() for matrix in pencil)
    a[0] = 0.0
    b[0] = 0.0
    return a, b


class TestGeneralizedEig:
    def test_output_static_shapes(self):
        a = pt.dmatrix("a", shape=(4, 4))
        b = pt.dmatrix("b", shape=(4, 4))
        re, im = generalized_eig(a, b, tol=TOL)
        assert re.type.shape == im.type.shape == (4,)

    def test_rejects_non_matrix(self):
        with pytest.raises(ValueError, match="requires two square matrices"):
            generalized_eig(pt.dvector("a"), pt.dmatrix("b"), tol=TOL)

    @pytest.mark.parametrize("pencil_name", ["pencil", "singular_pencil"])
    def test_matches_scipy(self, pencil_name, request):
        a, b = request.getfixturevalue(pencil_name)
        A, B = pt.dmatrices("a", "b")
        r, i = pytensor.function([A, B], generalized_eig(A, B, tol=TOL))(a, b)

        expected = linalg.eigvals(a, b)
        expected = expected[np.argsort(np.abs(expected))]
        assert_allclose(np.hypot(r, i), np.abs(expected), rtol=1e-10)
        finite = np.isfinite(expected)
        assert_allclose(np.sort(r[finite]), np.sort(expected[finite].real), rtol=1e-10)
        assert_allclose(np.sort(i[finite]), np.sort(expected[finite].imag), atol=1e-12)

    @pytest.mark.parametrize(
        ("dtype", "expected_dtype"),
        [
            pytest.param(
                "float32",
                "float32",
                marks=pytest.mark.skip(reason="PyTensor's Numba real QZ with eigenvalues fails to type float32 input"),
            ),
            ("int64", "float64"),
        ],
    )
    def test_output_dtype_follows_qz(self, dtype, expected_dtype):
        a = np.array([[2, 1], [0, 3]], dtype=dtype)
        b = np.array([[1, 0], [1, 2]], dtype=dtype)
        A, B = pt.matrix("a", dtype=dtype), pt.matrix("b", dtype=dtype)
        r, i = pytensor.function([A, B], generalized_eig(A, B, tol=TOL))(a, b)

        expected = linalg.eigvals(a, b)
        assert r.dtype == i.dtype == expected_dtype
        assert_allclose(np.sort(r), np.sort(expected.real), rtol=1e-6)

    @pytest.mark.parametrize(
        "reduce_outputs",
        [lambda re, _im: re.sum(), lambda _re, im: (im**2).sum(), lambda re, im: (re**2 + im**2).sum()],
        ids=["real", "imag", "modulus"],
    )
    def test_grad(self, pencil, rng, reduce_outputs):
        def f(a, b):
            return reduce_outputs(*generalized_eig(a, b, tol=TOL))

        verify_grad(f, list(pencil), rng=rng)

    def test_grad_ignores_infinite_eigenvalues(self, singular_pencil, rng):
        """Perturbing the zero column of b would make the infinite eigenvalue finite, so that column stays fixed."""
        a, b = singular_pencil
        zero_column = np.zeros((b.shape[0], 1))

        def finite_modulus_sum(a, b_rest):
            re, im = generalized_eig(a, pt.concatenate([zero_column, b_rest], axis=1), tol=TOL)
            return pt.switch(pt.isfinite(re), re**2 + im**2, 0.0).sum()

        verify_grad(finite_modulus_sum, [a, b[:, 1:]], rng=rng)

    def test_grad_of_a_repeated_eigenvalue_sum(self, repeated_eigenvalue_pencil, rng):
        """Single eigenvalues of a repeated cluster have no derivative, but their sum does."""

        def eigenvalue_sum(a, b):
            re, _ = generalized_eig(a, b, tol=TOL)
            return re.sum()

        verify_grad(eigenvalue_sum, list(repeated_eigenvalue_pencil), rng=rng)

    @pytest.mark.parametrize("mode", ["NUMBA", "CVM"])
    @pytest.mark.parametrize(("with_gradient", "expected_qz"), [(False, [False]), (True, [False, True])])
    def test_complex_qz_only_in_the_gradient(self, mode, with_gradient, expected_qz):
        A, B = pt.dmatrices("a", "b")
        re, _ = generalized_eig(A, B, tol=TOL)
        outputs = [re, pt.grad(re.sum(), A)] if with_gradient else [re]
        f = pytensor.function([A, B], outputs, mode=mode)

        core_ops = [getattr(node.op, "core_op", node.op) for node in f.maker.fgraph.apply_nodes]
        assert sorted(op.complex_output for op in core_ops if isinstance(op, QZ)) == expected_qz

    @pytest.mark.parametrize("mode", ["NUMBA", "CVM"])
    def test_backends_match_python(self, singular_pencil, mode):
        A, B = pt.dmatrices("a", "b")
        re, im = generalized_eig(A, B, tol=TOL)
        finite = pt.isfinite(re)
        outputs = [re, im, *pt.grad(pt.switch(finite, re**2 + im**2, 0.0).sum(), [A, B])]

        python_results = pytensor.function([A, B], outputs, mode="FAST_COMPILE")(*singular_pencil)
        backend_results = pytensor.function([A, B], outputs, mode=mode)(*singular_pencil)

        for python_result, backend_result in zip(python_results, backend_results, strict=True):
            assert_allclose(backend_result, python_result, atol=1e-12)


@pytest.mark.parametrize("pencil_name", ["singular_pencil", "repeated_eigenvalue_pencil"])
@pytest.mark.parametrize("mode", ["FAST_COMPILE", "NUMBA"])
def test_schur_eigenvectors_solve_the_pencil(pencil_name, mode, request):
    a, b = request.getfixturevalue(pencil_name)
    A, B = pt.dmatrices("a", "b")
    re, im = generalized_eig(A, B, tol=TOL)
    S, T, Q, Z = pt.linalg.qz(A, B, output="complex")
    re, im, X, Y = pytensor.function([A, B], [re, im, *SchurEigenvectors()(S, T, Q, Z, re, im)], mode=mode)(a, b)

    finite = np.isfinite(re)
    eigvals = re[finite] + 1j * im[finite]
    X_f, Y_f = X[:, finite], Y[:, finite]

    assert_allclose(a @ X_f, (b @ X_f) * eigvals, atol=1e-12)
    assert_allclose(Y_f.conj().T @ a, eigvals[:, None] * (Y_f.conj().T @ b), atol=1e-12)
    assert_allclose(Y_f.conj().T @ b @ X_f, np.eye(finite.sum()), atol=1e-12)
    assert np.all(X[:, ~finite] == 0)


@pytest.mark.parametrize("mode", ["FAST_COMPILE", "NUMBA"])
def test_schur_eigenvectors_never_match_a_zero_diagonal_pair(degenerate_pencil, mode):
    """A zero row in both matrices puts a (0, 0) pair on the Schur diagonal, which has no eigenvector to offer."""
    A, B = pt.dmatrices("a", "b")
    re, im = generalized_eig(A, B, tol=TOL)
    S, T, Q, Z = pt.linalg.qz(A, B, output="complex")
    X, Y = pytensor.function([A, B], SchurEigenvectors()(S, T, Q, Z, re, im), mode=mode)(*degenerate_pencil)

    assert np.all(np.isfinite(X))
    assert np.all(np.isfinite(Y))
