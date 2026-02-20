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

        f_dense = function([x1, x2, y1, y2], jac_dense, on_unused_input="ignore")
        f_sparse = function([x1, x2, y1, y2], pts.dense_from_sparse(jac_sparse), on_unused_input="ignore")

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


class TestSparseJacobianBenchmark:
    @staticmethod
    def build_system(
        n_eq: int,
        rng=None,
        min_terms: int = 1,
        max_terms: int = 4,
        coef_range=(0.5, 2.0),
        ops=(pt.identity, pt.sin, pt.cos, pt.exp, lambda x: x**2, lambda x: x**3),
    ):
        rng = np.random.default_rng(rng)

        n_vars = n_eq
        variables = [pt.dscalar(f"x{i}") for i in range(n_vars)]

        vars_per_eq = rng.integers(min_terms, max_terms + 1, size=n_eq)

        seed_vars = rng.permutation(n_vars)

        equations = []
        all_vars = np.arange(n_vars)

        lo, hi = coef_range
        for i in range(n_eq):
            chosen = [int(seed_vars[i])]
            n_extra = int(vars_per_eq[i]) - 1

            if n_extra > 0:
                candidates = all_vars[all_vars != chosen[0]]
                extra_idx = rng.choice(candidates, size=min(n_extra, n_vars - 1), replace=False)
                chosen.extend(map(int, extra_idx))

            eq = pt.constant(0.0)
            for j in chosen:
                op = rng.choice(ops)
                eq = eq + rng.uniform(lo, hi) * op(variables[j])
            equations.append(eq)

        return variables, equations

    @pytest.mark.parametrize("n_eq", [5, 10, 20, 50, 100])
    def test_sparse_jacobian(self, n_eq, benchmark):
        variables, equations = self.build_system(n_eq=n_eq)
        jac = sparse_jacobian(equations, variables, return_sparse=False)
        fn = pytensor.function(variables, jac, on_unused_input="ignore", mode="JAX")

        test_values = dict.fromkeys([var.name for var in variables], 1.0)

        def fn_with_inputs():
            return fn(**test_values)

        fn_with_inputs()
        benchmark(fn_with_inputs)

    @pytest.mark.parametrize("n_eq", [5, 10, 20, 50, 100])
    def test_dense_jacobian(self, n_eq, benchmark):
        variables, equations = self.build_system(n_eq=n_eq)
        dense_jac = pt.stack(pt.jacobian(pt.stack(equations), variables, vectorize=True), axis=-1)
        fn = pytensor.function(variables, dense_jac, on_unused_input="ignore", mode="JAX")

        test_values = dict.fromkeys([var.name for var in variables], 1.0)

        def fn_with_inputs():
            return fn(**test_values)

        benchmark(fn_with_inputs)
