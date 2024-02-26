from itertools import product

import numpy as np
import sympy as sp

from gEconpy.classes.time_aware_symbol import TimeAwareSymbol


class SymbolicSteadyStateSolver:
    def __init__(self):
        pass

    @staticmethod
    def score_eq(
        eq: sp.Expr,
        var_list: list[sp.Symbol],
        state_vars: list[sp.Symbol],
        var_penalty_factor: float = 25,
        state_var_penalty_factor: float = 5,
        length_penalty_factor: float = 1,
    ) -> float:
        """
        Compute an "unfitness" score for an equation using three simple heuristics:
            1. The number of jumper variables in the expression
            2. The number of state variables in the expression
            3. The total length of the expression

        Expressions with the lowest unfitness will be selected. Setting a lower penalty for state variables will
        push the system towards finding solutions expressed in state variables if a steady state is parameters only
        cannot be found.

        Parameters
        ----------
        eq: sp.Expr
            A sympy expression representing a steady-state equation
        var_list: list of sp.Symbol
            A list of sympy symbols representing all variables in the model (state and jumper)
        state_vars: list of sp.Symbol
            A list of symbol symbols representing all state variables in the model
        var_penalty_factor: float, default: 25
            A penalty factor applied to unfitness for each jumper variable in the expression.
        state_var_penalty_factor: float, default: 5
            A penalty factor applied to unfitness for each control variable in the expression.
        length_penalty_factor: float, default: 1
            A penalty factor applied to each term in the expression

        Returns
        -------
        unfitness: float
            An unfitness score used to select potential substitutions between system equations
        """

        # If the equation is length zero, it's been reduced away and should never be selected.
        if eq == 0:
            return 10000

        var_list = list(set(var_list) - set(state_vars))

        # The equation with the LOWEST score will be chosen to substitute, so punishing state variables less
        # ensures that equations that have only state variables will be chosen more often.
        var_penalty = len([x for x in eq.atoms() if x in var_list]) * var_penalty_factor
        state_var_penalty = (
            len([x for x in eq.atoms() if x in state_vars]) * state_var_penalty_factor
        )

        # Prefer shorter equations
        length_penalty = eq.count_ops() * length_penalty_factor

        return var_penalty + state_var_penalty + length_penalty

    @staticmethod
    def solve_and_return(eq: sp.Expr, v: sp.Symbol) -> sp.Expr:
        """
        Attempt to solve an expression for a given variable. Returns 0 if the expression is not solvable or if the
        given variable does not appear in the expression. If multiple solutions are found, only the first one is
        returned.

        Parameters
        ----------
        eq: sp.Expr
            A sympy expression
        v: sp.Symbol
            A sympy symbol

        Returns
        -------
        solution: sp.Expr
            Given f(x, ...) =  0, returns x = g(...) if possible, or 0 if not.
        """

        if v not in eq.atoms():
            return sp.Float(0)
        try:
            solution = sp.solve(eq, v)
        except Exception:
            return sp.Float(0)

        if len(solution) > 0:
            return solution[0]

        return sp.Float(0)

    @staticmethod
    def clean_substitutions(
        sub_dict: dict[sp.Symbol, sp.Expr],
    ) -> dict[sp.Symbol, sp.Expr]:
        """
        "Cleans" a dictionary of substitutions by:
            1. Delete substitutions in the form of x=x or x=0 (x=0 implies the substitution is redundant with other
                substitutions in sub_dict)
            2. If a substitution is of the form x = f(x, ...), attempts to solve the expression x - f(x, ...) = 0 for x,
                and deletes the substitution if no solution exists.
            3. Apply all substitutions in sub_dict to expressions in sub_dict to ensure older solutions remain up to
                date with newly found solutions.

        Parameters
        ----------
        sub_dict: dict
            Dictionary of sp.Symbol keys and sp.Expr values to be passed to the subs method of sympy expressions.

        Returns
        -------
        sub_dict: dict
            Cleaned dictionary of sympy substitutions
        """
        result = sub_dict.copy()

        for k, eq in sub_dict.items():
            # Remove invalid or useless substitutions
            if eq == 0 or k == eq:
                del result[k]
                continue

            # Solve for the sub variable if necessary
            elif k in eq.atoms():
                try:
                    eq = sp.solve(k - eq, k)[0]
                except Exception:
                    del result[k]
                    continue
            result[k] = eq

        # Substitute subs into the sub dict
        result = {k: v.subs(result) for k, v in result.items()}
        return result

    def get_candidates(
        self,
        system: list[sp.Expr],
        variables: list[sp.Symbol],
        state_variables: list[sp.Symbol],
        var_penalty_factor: float = 25,
        state_var_penalty_factor: float = 5,
        length_penalty_factor: float = 1,
        cores: int = -1,
    ) -> dict[sp.Symbol, tuple[sp.Expr, float]]:
        """
        Attempt to solve every equation in the system for every variable. Scores the results using the score_eq
        function, and returns (solution, score) pairs with the highest fitness (lowest unfitness).

        Solving equations is parallelized using joblib.

        Parameters
        ----------
        system: list of sp.Expr
            List of steady state equations to be scored
        variables: list of sp.Symbol
            List of all variables among all steady state equations
        state_variables: list of Sp.Symbol
            List of all state variables among all steady state equations
        var_penalty_factor: float, default: 25
            A penalty factor applied to unfitness for each jumper variable in the expression.
        state_var_penalty_factor: float, default: 5
            A penalty factor applied to unfitness for each control variable in the expression.
        length_penalty_factor: float, default: 1
            A penalty factor applied to each term in the expression
        cores: int, default -1
            Number of cores over which to parallelize computation. Passed to joblib.Parallel. -1 for all available
            cores.

        Returns
        -------
        candidates: dict
            A dictionary of candidate substitutions to simplify the steady state system. One candidate is produced
            for each variable in the system. Keys are sp.Symbol, and values are (sp.Expr, float) tuples with the
            candidate substitution and its fitness.
        """
        try:
            from joblib import Parallel, delayed
        except ImportError:
            raise ImportError(
                "This function requires the python library joblib"
                "The easiest way to install all of this is by running\n\n"
                "\tconda install -c conda-forge joblib"
            )

        eq_vars = product(system, variables)

        n = len(system)
        k = len(variables)
        args = (
            variables,
            state_variables,
            var_penalty_factor,
            state_var_penalty_factor,
            length_penalty_factor,
        )

        with Parallel(cores) as pool:
            solutions = pool(delayed(self.solve_and_return)(eq, v) for eq, v in eq_vars)
            scores = np.array(pool(delayed(self.score_eq)(eq, *args) for eq in solutions))

        score_matrix = scores.reshape(n, k)
        idx_matrix = np.arange(n * k).reshape(n, k)
        best_idx = idx_matrix[score_matrix.argmin(axis=0), np.arange(k)]

        return dict(zip(variables, [(solutions[idx], scores[idx]) for idx in best_idx]))

    @staticmethod
    def make_solved_subs(sub_dict, assumptions):
        res = {}
        for k, v in sub_dict.items():
            if not any([isinstance(x, TimeAwareSymbol) for x in v.atoms()]):
                if v == 1:
                    continue
                res[v] = sp.Symbol(k.name + r"^\star", **assumptions[k.base_name])

        return res

    def solve_symbolic_steady_state(
        self,
        mod,
        top_k=3,
        var_penalty_factor=25,
        state_var_penalty_factor=5,
        length_penalty_factor=1,
        cores=-1,
        zero_tol=12,
    ):
        ss_vars = [x.to_ss() for x in mod.variables]
        state_vars = [x for x in mod.variables if x.base_name == "Y"]
        ss_system = mod.steady_state_system

        system = ss_system.copy()
        calib_eqs = [
            var - eq for var, eq in zip(mod.params_to_calibrate, mod.calibrating_equations)
        ]
        system.extend(calib_eqs)

        params = list(mod.free_param_dict.to_sympy().keys())
        sub_dict = {}
        unsolved_dict = {}

        while True:
            candidates = self.get_candidates(
                system,
                ss_vars,
                state_vars,
                var_penalty_factor=var_penalty_factor,
                state_var_penalty_factor=state_var_penalty_factor,
                length_penalty_factor=length_penalty_factor,
                cores=cores,
            )

            scores = np.array([score for eq, score in candidates.values()])
            print(scores)
            top_k_score_idxs = scores.argsort()[:top_k]
            for idx in top_k_score_idxs:
                key = list(candidates.keys())[idx]
                if candidates[key][0] == 0:
                    continue
                sub_dict[key] = candidates[key][0]

            sub_dict = self.clean_substitutions(sub_dict)

            system = [eq.subs(sub_dict) for eq in system]
            system = [
                eq
                for eq in system
                if not self.test_expr_is_zero(eq.subs(unsolved_dict), params, tol=zero_tol)
            ]
            solved_dict = self.make_solved_subs(sub_dict, mod.assumptions)
            unsolved_dict = {v: k.subs(unsolved_dict) for k, v in solved_dict.items()}
            system = [eq.subs(solved_dict) for eq in system]

            if len(system) == 0:
                break

            if min(scores) > 100:
                break

        to_solve = {x for eq in system for x in eq.atoms() if isinstance(x, TimeAwareSymbol)}
        system = [eq.simplify() for eq in system]
        try:
            final_solutions = sp.solve(system, to_solve, dict=True)
        except NotImplementedError:
            final_solutions = [{}]

        return [sub_dict.update(d) for d in final_solutions]
