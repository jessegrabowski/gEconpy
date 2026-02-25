import pytest

from gEconpy.model.perfect_foresight import compile_perfect_foresight_problem
from tests._resources.cache_compiled_models import load_and_cache_model


@pytest.fixture
def rbc_model():
    return load_and_cache_model("one_block_1.gcn")


class TestCompile:
    def test_problem_has_correct_dimensions(self, rbc_model):
        T = 25
        problem = compile_perfect_foresight_problem(rbc_model, T)

        assert problem.T == T
        assert problem.n_vars == len(rbc_model.variables)
        assert problem.n_shocks == len(rbc_model.shocks)
        assert problem.n_eq == problem.n_vars
