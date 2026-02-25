import numpy as np

SOLVER_TOL = 1e-8


class Gensys:
    params = [["RBC", "New_Keynesian"]]
    param_names = ["model"]

    def setup(self, model):
        from gEconpy.data import get_example_gcn
        from gEconpy.model.build import model_from_gcn
        from gEconpy.solvers.gensys import solve_policy_function_with_gensys

        self.solve_gensys = solve_policy_function_with_gensys

        # Build model and extract linearization matrices
        m = model_from_gcn(get_example_gcn(model), verbose=False)
        m.steady_state(verbose=False)
        A, B, C, D = m.linearize_model(verbose=False)

        self.A = np.ascontiguousarray(A)
        self.B = np.ascontiguousarray(B)
        self.C = np.ascontiguousarray(C)
        self.D = np.ascontiguousarray(D)

    def time_solve(self, model):
        self.solve_gensys(self.A, self.B, self.C, self.D, tol=SOLVER_TOL)
