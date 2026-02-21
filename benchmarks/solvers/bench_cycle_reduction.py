import numpy as np

# Solver parameters
MAX_ITER = 1000
SOLVER_TOL = 1e-7


class CycleReduction:
    params = [["RBC", "New_Keynesian"]]
    param_names = ["model"]

    def setup(self, model):
        from gEconpy.data import get_example_gcn
        from gEconpy.model.build import model_from_gcn
        from gEconpy.solvers.cycle_reduction import nb_cycle_reduction

        self.nb_cycle_reduction = nb_cycle_reduction

        # Build model and extract linearization matrices
        m = model_from_gcn(get_example_gcn(model), verbose=False, backend="numpy")
        m.steady_state(verbose=False)
        A, B, C, _D = m.linearize_model(verbose=False)

        # Cycle reduction solves A0*X^2 + A1*X + A2 = 0
        # where A0=C (leads), A1=B (current), A2=A (lags)
        self.A0 = np.ascontiguousarray(C)
        self.A1 = np.ascontiguousarray(B)
        self.A2 = np.ascontiguousarray(A)

    def time_solve(self, model):
        self.nb_cycle_reduction(self.A0, self.A1, self.A2, max_iter=MAX_ITER, tol=SOLVER_TOL)
