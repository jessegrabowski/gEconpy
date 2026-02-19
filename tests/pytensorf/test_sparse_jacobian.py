import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest

from pytensor import function
from pytensor import sparse as pts
from pytensor.tensor import TensorVariable

from gEconpy.pytensorf.sparse_jacobian import sparse_jacobian


class TestSparseJacobian:
    def test_jacobian_values(self):
        x, y = pt.dscalars("x", "y")
        eq1 = x**2 + y
        eq2 = x - y**2

        jac = sparse_jacobian([eq1, eq2], [x, y], return_sparse=False)
        f = function([x, y], jac)

        # At x=2, y=3:
        # d(eq1)/dx=2x=4
        # d(eq1)/dy=1
        # d(eq2)/dx=1
        # d(eq2)/dy=-2y=-6
        expected = np.array([[4.0, 1.0], [1.0, -6.0]])
        np.testing.assert_allclose(f(2.0, 3.0), expected)

    def test_dense_and_sparse_agree(self):
        """Dense and sparse outputs must produce identical values."""
        x1, x2, y1, y2 = pt.dscalars("x1", "x2", "y1", "y2")
        eq1 = x1**2 + 2 * y1
        eq2 = 3 * x2 - y2**2

        jac_dense = sparse_jacobian([eq1, eq2], [x1, x2, y1, y2], return_sparse=False)
        jac_sparse = sparse_jacobian([eq1, eq2], [x1, x2, y1, y2], return_sparse=True)

        f_dense = function([x1, x2, y1, y2], jac_dense)
        f_sparse = function([x1, x2, y1, y2], pts.dense_from_sparse(jac_sparse))

        vals = (1.0, 2.0, 3.0, 4.0)
        np.testing.assert_allclose(f_dense(*vals), f_sparse(*vals))

    def test_coloring_compression_preserves_sparsity(self):
        x, y, z = pt.dscalars("x", "y", "z")
        # Each equation depends on exactly one variable - maximally sparse
        eq1 = x**2
        eq2 = y**2
        eq3 = z**2

        jac = sparse_jacobian([eq1, eq2, eq3], [x, y, z], return_sparse=False)
        f = function([x, y, z], jac)

        # Should produce diagonal matrix with correct derivatives
        expected = np.diag([2.0, 4.0, 6.0])
        np.testing.assert_allclose(f(1.0, 2.0, 3.0), expected)

    def test_vector_element_variables(self):
        x = pt.vector("x", shape=(4,))
        eq1 = x[0] ** 2 + x[1]  # depends on x[0], x[1]
        eq2 = x[2] - x[3] ** 2  # depends on x[2], x[3]
        eq3 = x[0] + x[2]  # depends on x[0], x[2]

        variables = [x[i] for i in range(4)]
        equations = [eq1, eq2, eq3]

        jac = sparse_jacobian(equations, variables, return_sparse=False, use_vectorized_jacobian=True)
        f = function([x], jac)

        # At x=[1,2,3,4]:
        # eq1: d/dx0=2*1=2, d/dx1=1, d/dx2=0, d/dx3=0
        # eq2: d/dx0=0, d/dx1=0, d/dx2=1, d/dx3=-2*4=-8
        # eq3: d/dx0=1, d/dx1=0, d/dx2=1, d/dx3=0
        expected = np.array([[2.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, -8.0], [1.0, 0.0, 1.0, 0.0]])
        result = f(np.array([1.0, 2.0, 3.0, 4.0]))
        np.testing.assert_allclose(result, expected)


class TestSparseJacobianBenchmark:
    def build_system(self, n_eq: int, seed: int = 42) -> tuple[list[TensorVariable], list[TensorVariable]]:
        rng = np.random.default_rng(seed)
        n_vars = n_eq + 1
        variables = [pt.dscalar(f"x{i}") for i in range(n_vars)]

        ops = [pt.sin, pt.cos, pt.exp, lambda x: x**2, lambda x: x**3]
        equations = []

        for i in range(n_eq):
            term_vars = [variables[i], variables[i + 1]]

            eq = 0
            for var in term_vars:
                op = rng.choice(ops)
                coef = rng.uniform(0.5, 2.0)
                eq = eq + coef * op(var)
            equations.append(eq)

        return variables, equations

    @pytest.mark.parametrize("n_eq", [5, 10, 20, 50, 100, 500])
    def test_sparse_jacobian(self, n_eq, benchmark):
        variables, equations = self.build_system(n_eq=n_eq)
        jac = sparse_jacobian(equations, variables, return_sparse=False, use_vectorized_jacobian=True)
        fn = pytensor.function(variables, jac, mode="JAX")

        test_values = dict.fromkeys([var.name for var in variables], 1.0)

        def fn_with_inputs():
            return fn(**test_values)

        fn_with_inputs()
        benchmark(fn_with_inputs)

    @pytest.mark.parametrize("n_eq", [5, 10, 20, 50, 100, 500])
    def test_dense_jacobian(self, n_eq, benchmark):
        variables, equations = self.build_system(n_eq=n_eq)
        dense_jac = pt.stack(pt.jacobian(pt.stack(equations), variables, vectorize=True), axis=-1)
        fn = pytensor.function(variables, dense_jac, mode="JAX")

        test_values = dict.fromkeys([var.name for var in variables], 1.0)

        def fn_with_inputs():
            return fn(**test_values)

        benchmark(fn_with_inputs)
