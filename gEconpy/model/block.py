from collections import defaultdict

import sympy as sp

from gEconpy.classes.containers import SymbolDictionary
from gEconpy.classes.time_aware_symbol import TimeAwareSymbol
from gEconpy.exceptions import (
    ControlVariableNotFoundException,
    DuplicateParameterError,
    DynamicCalibratingEquationException,
    MultipleObjectiveFunctionsException,
    OptimizationProblemNotDefinedException,
)
from gEconpy.utilities import (
    diff_through_time,
    expand_subs_for_all_times,
    set_equality_equals_zero,
    substitute_repeatedly,
    unpack_keys_and_values,
)


class Block:
    """
    The Block class holds equations and parameters associated with each block of the DSGE model. They hold methods
    to solve their associated optimization problem. Blocks should be created by a Model.
    """

    #     TODO: Split components out into their own class/protocol and let them handle their own parsing?
    #    TODO: Refactor this into an abstract class with basic functionality, then create some child classes for specific
    #     problems, e.g. IdentityBlock, OptimizationBlock, CRRABlock, etc, each with their own optimization machinery.

    def __init__(
        self,
        name: str,
        block_dict: dict[str, list[list[str]]],
        assumptions: dict[str, dict] | None = None,
    ) -> None:
        """
        Initialize a block object

        Parameters
        ----------
        name: str
            The name of the block
        block_dict: dict
            Dictionary of component:list[equations] key-value pairs created by gEcon_parser.parsed_block_to_dict.
        """

        self.name = name
        self.short_name = "".join(word[0] for word in name.split("_"))

        self.definitions: dict[int, sp.Eq] | None = None
        self.controls: list[TimeAwareSymbol] | None = None
        self.objective: dict[int, sp.Expr] | None = None
        self.constraints: dict[int, sp.Eq] | None = None
        self.identities: dict[int, sp.Eq] | None = None
        self.shocks: dict[int, TimeAwareSymbol] | None = None
        self.calibration: dict[int, sp.Expr] | None = None

        self.variables: list[TimeAwareSymbol] = []
        self.param_dict: SymbolDictionary[str, float] = SymbolDictionary()

        self.calib_dict: SymbolDictionary[str, float] = SymbolDictionary()
        self.deterministic_dict: SymbolDictionary[str, float] = SymbolDictionary()

        self.system_equations: list[sp.Expr] = []
        self.multipliers: dict[int, TimeAwareSymbol] = {}
        self.eliminated_variables: list[sp.Symbol] = []

        self.equation_flags: dict[int, dict[str, bool]] = {}

        self.n_equations = 0
        self.initialized = False

        if assumptions is None:
            assumptions = defaultdict(dict)

        self.initialize_from_dictionary(block_dict, assumptions)
        self._consolidate_definitions()

        self._get_variable_list()
        self._get_param_dict_and_calibrating_equations()

    def __str__(self):
        return (
            f"{self.name} Block of {self.n_equations} equations, initialized: {self.initialized}, "
            f"solved: {self.system_equations is not None}"
        )

    @property
    def deterministic_params(self) -> list[sp.Symbol]:
        return list(self.deterministic_dict.to_sympy().keys())

    @property
    def deterministic_relationships(self) -> list[sp.Expr]:
        return list(self.deterministic_dict.values())

    @property
    def params_to_calibrate(self) -> list[sp.Symbol]:
        return list(self.calib_dict.to_sympy().keys())

    @property
    def calibrating_equations(self) -> list[sp.Expr]:
        return list(self.calib_dict.values())

    def initialize_from_dictionary(self, block_dict: dict, assumptions: dict) -> None:
        """
        Initialize the model block with the provided definitions, objective, constraints, identities, and calibration
        equations. The model block's controls and shocks will also be extracted from the provided block dictionary.

        Parameters
        ----------
        block_dict: dict
            A dictionary of component: list[equations] key-value pairs created by gEcon_parser.parsed_block_to_dict
        assumptions: dict
            A dictionary of user-provided Sympy assumptions about variables in the model.

        Returns
        -------
        None
        """

        self.controls = self._parse_variable_list(block_dict, "controls", assumptions)
        self.shocks = self._parse_variable_list(block_dict, "shocks", assumptions)

        self.definitions = self._parse_equation_list(
            block_dict, "definitions", assumptions
        )
        self.objective = self._parse_equation_list(block_dict, "objective", assumptions)
        self.constraints = self._parse_equation_list(
            block_dict, "constraints", assumptions
        )
        self.identities = self._parse_equation_list(
            block_dict, "identities", assumptions
        )
        self.calibration = self._parse_equation_list(
            block_dict, "calibration", assumptions
        )

        self.initialized = self._validate_initialization()

    def _validate_initialization(self) -> bool:
        """
        Check whether the block has been successfully initialized.

        At a high level, gEcon allows for two kinds of blocks: those with and those without an optimization problem.
        To have an optimization problem, the block needs both the `controls` and `objective` components to be present.
        Additionally, all control variables need to be represented among the equations in `objective`, `definitions`, and
        `constraints`.

        Parameters
        ----------
        self: Block
            The block to be checked

        Returns
        -------
        bool
            Indicates whether the block has been successfully initialized.

        Raises
        ------
        OptimizationProblemNotDefinedException
            If either the `controls` or `objective` component is missing
        MultipleObjectiveFunctionsException
            If there is more than one objective function defined
        ControlVariableNotFoundException
            If a control variable is not found in any of the `objective`, `definitions`, or `constraints` equations.
        """

        if self.objective is not None and self.controls is None:
            raise OptimizationProblemNotDefinedException(
                block_name=self.name, missing="controls"
            )

        if self.objective is None and self.controls is not None:
            raise OptimizationProblemNotDefinedException(
                block_name=self.name, missing="objective"
            )

        if self.objective is not None and len(list(self.objective.values())) > 1:
            raise MultipleObjectiveFunctionsException(
                block_name=self.name, eqs=list(self.objective.values())
            )

        if self.controls is not None:
            for control in self.controls:
                control_found = False
                eq_dicts = [
                    x
                    for x in [self.definitions, self.objective, self.constraints]
                    if x is not None
                ]
                for eq_dict in eq_dicts:
                    for eq in list(eq_dict.values()):
                        if control in eq.atoms():
                            control_found = True
                            break
                if not control_found:
                    raise ControlVariableNotFoundException(self.name, control)

        # Validate equation flags
        # - the "is_calibrating" key can only occur in the calibration block
        # - the "exclude" key can only occur in the constraints block
        valid_flags = {
            "is_calibrating": ["calibration"],
            "exclude": ["constraints"],
        }

        for name, eq_block in zip(
            ["definitions", "objective", "constraints", "identities"],
            [self.definitions, self.objective, self.constraints, self.identities],
        ):
            if eq_block is not None:
                for key, eq in eq_block.items():
                    if (
                        self.equation_flags[key].get("is_calibrating", False)
                        and name not in valid_flags["is_calibrating"]
                    ):
                        raise ValueError(
                            f"Equation {eq} in {name} block of {self.name} has an invalid decorator: is_calibrating. "
                            f"This flag should only appear in the calibration block."
                        )
                    if (
                        self.equation_flags[key].get("exclude", False)
                        and name not in valid_flags["exclude"]
                    ):
                        raise ValueError(
                            f"Equation {eq} in {name} block of {self.name} has an invalid decorator: exclude. "
                            f"This flag should only appear in the constraints block."
                        )

        return True

    def _validate_key(self, block_dict: dict, key: str) -> bool:
        """
        Check whether a block component is present in the block_dict, and a valid component name. For valid component
        names, see gEcon_parser.BLOCK_COMPONENTS.

        Parameters
        ----------
        block_dict : dict
            Dictionary of component:list[equations] key-value pairs created by gEcon_parser.parsed_block_to_dict.
        key : str
            A component name.

        Returns
        -------
        bool
        """
        return key in block_dict and hasattr(self, key) and block_dict[key] is not None

    def _consolidate_definitions(self):
        """
        Combine definitions that refer to other definitions via subsitution
        """
        if self.definitions is None:
            return

        sub_dict = {eq.lhs: eq.rhs for eq in self.definitions.values()}

        for var, eq in sub_dict.items():
            if not hasattr(eq, "subs"):
                continue
            sub_dict[var] = substitute_repeatedly(eq, sub_dict)

        self.definitions = {
            k: sp.Eq(v.lhs, v.rhs.subs(sub_dict)) for k, v in self.definitions.items()
        }

    def _extract_lagrange_multipliers(
        self, equations: list[list[str]], assumptions: dict
    ) -> tuple[list[list[str]], list[TimeAwareSymbol | None]]:
        """
        gEcon allows the user to name lagrange multipliers in the GCN file. These multiplier variables need to be saved
        and used once the optimization problem is solved. This function removes the ": muliplier[]" from each equation
        and returns them as a list, along with the new equations. A None is placed in the list for each equation
        with no associated multiplier.

        Parameters
        ----------
        equations : list
            A list of lists of strings, each list representing a model equation. Created by the
            gEcon_parser.parsed_block_to_dict function.
        assumptions : dict
            Assumptions for the model.

        Returns
        -------
        list
            List of lists of strings.
        list
            List of Union[TimeAwareSymbols, None].
        """
        from gEconpy.parser import parse_equations

        result, multipliers = [], []
        for eq in equations:
            if ":" in eq:
                colon_idx = eq.index(":")
                multiplier = eq[-1]
                multiplier = parse_equations.single_symbol_to_sympy(
                    multiplier, assumptions
                )
                eq = eq[:colon_idx].copy()

                result.append(eq)
                multipliers.append(multiplier)
            else:
                result.append(eq)
                multipliers.append(None)

        return result, multipliers

    def _extract_decorators(
        self, equations: list[list[str]], assumptions: dict
    ) -> tuple[list[list[str]], list[dict[str, bool]]]:
        """
        Extract decorators from the equations in the block. Decorators are flags that indicate special properties of the
        equation, such as whether it should be excluded from the final system of equations.

        Parameters
        ----------
        equations : list
            A list of lists of strings, each list representing a model equation. Created by the
            gEcon_parser.parsed_block_to_dict function.

        assumptions : dict
            Assumptions for the model.

        Returns
        -------
        equations: list
            List of lists of strings. All decorator strings have been removed.

        flags: dict
            A dictionary of flags for each equation, indexed by equation number.
        """

        result, decorator_flags = [], []
        for i, eq in enumerate(equations):
            new_eq = []
            flags = {}
            for token in eq:
                if token.startswith("@"):
                    decorator = token.removeprefix("@")
                    flags[decorator] = True
                else:
                    new_eq.append(token)
            result.append(new_eq)
            decorator_flags.append(flags)

        return result, decorator_flags

    def _parse_variable_list(
        self, block_dict: dict, key: str, assumptions: dict | None = None
    ) -> list[sp.Symbol] | None:
        from gEconpy.parser import parse_equations

        """
        Two components -- controls and shocks -- expect a simple list of variables, which is a case the
        gEcon_parser.build_sympy_equations cannot handle.

        Parameters
        ----------
        block_dict : list
            A list of lists of strings, each list representing a model equation. Created by the
            gEcon_parser.parsed_block_to_dict function.
        key : str
            A component name.
        assumptions : dict, optional
            Assumptions for the model.

        Returns
        -------
        list
            A list of variables, represented as Sympy objects, or None if the block does not exist.
        """
        if not self._validate_key(block_dict, key):
            return

        raw_list = [var for var_list in block_dict[key] for var in var_list]
        output = []
        for variable in raw_list:
            variable = parse_equations.single_symbol_to_sympy(variable, assumptions)
            output.append(variable)

        return output

    def _get_variable_list(self) -> None:
        """
        :return: None
        Get a list of all unique variables in the Block and store it in the class attribute "variables"
        """
        objective, constraints, identities, multipliers = [], [], [], []
        sub_dict = {}
        if self.definitions is not None:
            _, definitions = unpack_keys_and_values(self.definitions)
            sub_dict = {eq.lhs: eq.rhs for eq in definitions}

        if self.objective is not None:
            _, objective = unpack_keys_and_values(self.objective)

        if self.constraints is not None:
            _, constraints = unpack_keys_and_values(self.constraints)

        if self.identities is not None:
            _, identities = unpack_keys_and_values(self.identities)

        all_equations = [
            eq for eq_list in [objective, constraints, identities] for eq in eq_list
        ]

        if self.multipliers is not None:
            _, multipliers = unpack_keys_and_values(self.multipliers)
            multipliers = [x for x in multipliers if x is not None]

        all_equations = [
            eq for eqs_list in [objective, constraints, identities] for eq in eqs_list
        ]
        for eq in all_equations:
            eq = substitute_repeatedly(eq, sub_dict)
            atoms = eq.atoms()
            variables = [x for x in atoms if isinstance(x, TimeAwareSymbol)]
            for variable in variables:
                if variable.to_ss() not in self.variables:
                    self.variables.append(variable.to_ss())

        if self.variables is None:
            return

        # Can't directly check if variables are not in shocks, because shocks will be None if there are none in the
        # model
        shocks = self.shocks or []
        self.variables = [*self.variables, *multipliers]
        self.variables = sorted(
            list({x for x in self.variables if x.set_t(0) not in shocks}),
            key=lambda x: x.name,
        )

    def _get_and_record_equation_numbers(self, equations: list[sp.Eq]) -> list[int]:
        """
        Get a list of all unique variables in the Block and store it in the class attribute "variables".

        Returns
        -------
        list
            A list of equation number indices
        """
        n_equations = len(equations)
        equation_numbers = range(self.n_equations, self.n_equations + n_equations)
        self.n_equations += n_equations

        return equation_numbers

    def _parse_equation_list(
        self, block_dict: dict, key: str, assumptions: dict[str, str]
    ) -> dict[int, sp.Eq] | None:
        """
        Convert a list of equations represented as strings into a dictionary of sympy equations, indexed by their
        equation number.

        Parameters
        ----------
        block_dict : list
            A list of lists of strings, each list representing a model equation. Created by the
            gEcon_parser.parsed_block_to_dict function.
        key : str
            A component name.
        assumptions : dict
            A dictionary with assumptions for each variable.

        Returns
        -------
        dict
            A dictionary of sympy equations, indexed by their equation number, or None if the block does not exist.
        """
        from gEconpy.parser import parse_equations

        if not self._validate_key(block_dict, key):
            return

        equations = block_dict[key]
        equations, lagrange_multipliers = self._extract_lagrange_multipliers(
            equations, assumptions
        )
        equations, decorators = self._extract_decorators(equations, assumptions)

        parser_output = parse_equations.build_sympy_equations(equations, assumptions)

        if len(parser_output) > 0:
            equations, flags = list(zip(*parser_output))
        else:
            equations, flags = [], {}

        equation_numbers = self._get_and_record_equation_numbers(equations)

        equations = dict(zip(equation_numbers, equations))
        flags = dict(zip(equation_numbers, flags))
        decorator_flags = dict(zip(equation_numbers, decorators))

        lagrange_multipliers = dict(zip(equation_numbers, lagrange_multipliers))
        self.multipliers.update(lagrange_multipliers)
        for k in equation_numbers:
            self.equation_flags[k] = flags[k]
            self.equation_flags[k].update(decorator_flags[k])

        return equations

    def _get_param_dict_and_calibrating_equations(self) -> None:
        """
        The calibration block, as implemented in gEcon, mixes together parameters, which are fixed values with a
        user-provided value, with calibrating equations, which are extra conditions added to the steady-state system.
        This function divides these out so that the Model instance can ask for only one or the other.

        These are divided heuristically: a parameter is assumed to be an equation of the form x = y, where x is a
        Sympy symbol (NOT a TimeAwareSymbol) and y is a Sympy number.

        Calibrating equations are identified by the use of the "->" operator in the GCN file, and are flagged during
        parsing. All TimeAwareSymbols in calibrating equations must be in the steady state, or else a Exception will be
        raised.

        Deterministic equations are parameters defined as functions of other parameters. For example, in a linear model,
        the user will need to define steady state values as model parameters. These parameters are analogous to
        equations beginning with # in Dynare.
        """
        # It is possible that an initialized block will not have a calibration component
        if self.calibration is None:
            return

        eq_idxs, equations = unpack_keys_and_values(self.calibration)
        duplicates = []

        # Main parameter processing loop
        for idx, eq in zip(eq_idxs, equations):
            atoms = eq.atoms()
            lhs, rhs = eq.lhs, eq.rhs
            if not lhs.is_symbol:
                raise ValueError(
                    "Left-hand side of calibrating expressions should be the single parameter to be "
                    f"computed. Found multiple argumnets: {eq.lhs.args}"
                )

            param = eq.lhs

            # Check if the RHS is just a number (most common case). If so, convert it to a float (rather than
            # an sp.Float, which won't play nice with lambdify later)
            if eq.rhs.is_number:
                value = eq.rhs.evalf()
                if param in self.param_dict.keys():
                    duplicates.append(param)
                else:
                    self.param_dict[param] = value

            # If the RHS was not a number, its either a calibrating equation or a deterministic relationship of other
            # parameters.

            # Calibrating equations are tagged in the equation_flags dictionary during parsing.
            elif self.equation_flags[idx]["is_calibrating"]:
                # Calibrating equations can have variables, but they must be in the steady state
                if not all(
                    [
                        x.time_index == "ss"
                        for x in atoms
                        if isinstance(x, TimeAwareSymbol)
                    ]
                ):
                    raise DynamicCalibratingEquationException(
                        eq=eq, block_name=self.name
                    )

                if param in self.calib_dict:
                    duplicates.append(param)
                else:
                    self.calib_dict[param] = rhs

            else:
                # What is left should only be "deterministic relationships", parameters that are defined as
                # functions of other parameters that the user wants to keep track of.

                # Check that these are functions of numbers and parameters only
                if any([isinstance(x, TimeAwareSymbol) for x in atoms]):
                    raise ValueError(
                        "Parameters defined as functions in the calibration sub-block cannot be functions "
                        f"of variables. Found:\n\n {eq} in {self.name}"
                    )
                if eq.lhs in self.deterministic_dict:
                    duplicates.append(lhs)
                else:
                    self.deterministic_dict[lhs] = rhs.doit()

        if len(duplicates) > 0:
            raise DuplicateParameterError(duplicates, self.name)

    def _build_lagrangian(self) -> sp.Add:
        """
        Split the calibration block into a dictionary of fixed parameters and a list of equations to be used for
        calibration.

        A parameter is assumed to be an equation with up to three atoms, all of which are of class sp.Symbol or
        sp.Number. Calibrating equations, on the other hand, are comprised of Symbols, TimeAwareSymbols, and numbers.
        In this second case, all TimeAwareSymbols must be in the steady state.

        Returns
        -------
        None
        """
        objective = next(iter(self.objective.values()))
        constraints = self.constraints
        multipliers = self.multipliers
        sub_dict = dict()

        if self.definitions is not None:
            definitions = list(self.definitions.values())
            sub_dict = {eq.lhs: eq.rhs for eq in definitions}

        i = 1

        lagrange = objective.rhs.subs(sub_dict)
        for key, constraint in constraints.items():
            if multipliers[key] is not None:
                lm = multipliers[key]
            else:
                lm = TimeAwareSymbol(f"lambda__{self.short_name}_{i}", 0)
                self.multipliers[i] = lm
                i += 1

            lagrange = lagrange - lm * (
                constraint.lhs.subs(sub_dict) - constraint.rhs.subs(sub_dict)
            )

        return lagrange

    def _get_discount_factor(self) -> sp.Symbol | None:
        """
        Calculate the discount factor of a Bellman equation.

        A Bellman equation has the form X[] = a[] + b * E[][X[1]], where `a[]` is the value of the objective function at
        time `t`, and `E[][X[1]]` is the expected continuation value conditioned on the current information set. The
        parameter `b` (0 < b < 1) is the discount factor that ensures the equation converges to a fixed point. This
        function extracts `b` from the objective function and returns it as a sympy symbol.

        For single period optimizations, the discount factor is 1.

        TODO: This function currently assumes the continuation value is a single variable, it will fail in the case of
        TODO: something like X[] = a[] + b * E[][Y[1] + Z[1]], although i don't know how such a function could arise?

        Returns
        -------
        sp.Symbol
            The discount factor of the Bellman equation.

        Raises
        ------
        ValueError
            If the block has multiple t+1 variables in the Bellman equation.
        """

        _, objective = unpack_keys_and_values(self.objective)
        objective = objective[0]

        variables = [x for x in objective.atoms() if isinstance(x, TimeAwareSymbol)]

        # Return 1 if there is no continuation value -- static optimization
        if all([x.time_index in [0, -1] for x in variables]):
            return sp.Float(1.0)

        else:
            # We expect a bellman equation of the form X[] = a[] + E[][f(a[1]]. Step one is to identify a[], the
            # instantaneous value function at time t. It should be a term isolated on the RHS of the equation.
            current_value = objective.lhs
            continuation_value = [
                x for x in objective.rhs.args if x.has(current_value.set_t(1))
            ]

            # continuation_value = [x for x in variables if x.time_index == 1 and x.set_t(0) in variables]
            if len(continuation_value) == 0:
                raise ValueError(
                    f"Block {self.name} did not find the continuation value of the current state value in the following"
                    f"objective function: {objective}. Objectives should be written in the form "
                    f"``V[t] = f(x[t]) + b[t] * E[V[t+1]]``, where V[t] is the current state value, f(x[t]) is the "
                    f"instantaneous value function, and b[t] is the discount factor."
                )

            continuation_value = continuation_value[0]
            discount_factor = continuation_value.subs({current_value.set_t(1): 1})
            return discount_factor

    def simplify_system_equations(self) -> None:
        """
        Simplify the system of equations that define the first-order conditions (FoCs) in the model. This function
        currently applies a heuristic to remove redundant Lagrange multipliers generated by the solver. User-named
        lagrange multipliers are not removed, following the example of gEcon.

        TODO: Add solution patterns for CES, CRRA, and CD functions. Check parameter values to allow CES to collapse
        TODO: to CD, and CRRA to log-utility.
        """

        system = self.system_equations
        simplified_system = system.copy()
        variables = [
            x for eq in system for x in eq.atoms() if isinstance(x, TimeAwareSymbol)
        ]
        generated_multipliers = list(
            {x for x in variables if "lambda__" in x.base_name}
        )

        # Strictly heuristic simplification: look for an equation of the form x = y and use it to substitute away
        # the generated multipliers.

        eliminated_variables = []
        for x in generated_multipliers:
            candidates = [eq for eq in simplified_system if x in eq.atoms()]
            for eq in candidates:
                # x = y will have 2 atoms, x = -y will have 3
                if len(eq.atoms()) <= 3:
                    sub_dict = sp.solve(eq, x, dict=True)[0]
                    sub_dict = expand_subs_for_all_times(sub_dict)
                    eliminated_variables.extend(list(sub_dict.keys()))
                    simplified_system = [eq.subs(sub_dict) for eq in simplified_system]
                    break

        simplified_system = [eq for eq in simplified_system if eq != 0]

        self.system_equations = simplified_system
        self.eliminated_variables = eliminated_variables

        for key, value in self.multipliers.items():
            if value in eliminated_variables:
                self.multipliers[key] = None

    def solve_optimization(self, try_simplify: bool = True) -> None:
        r"""
        Solve the optimization problem implied by the block structure:

        .. math::

            \begin{aligned}
            \max_{\text{[Controls]}} \quad & \sum_{t=0}^{\infty} \text{[Objective]} \\
            \text{subject to} \quad & \text{[Constraints]}
            \end{aligned}

        By setting up the following Lagrangian:

        ..math::

            L = Sum_{t=0}^\infty \text{Objective} - lambda_1 * \text{constraint}_1 - ... - \lambda_n * \text{constraint}_n

        And taking the derivative with respect to each control variable in turn.

        Parameters
        ----------
        try_simplify : bool
            Whether to apply simplifications to the FoCs.

        Returns
        -------
        None

        Notes
        -----
        All first order conditions, along with the constraints and objective are stored in the .system_equations method.
        No attempt is made to simplify the resulting system if try_simplify = False.
        """

        # TODO: Add helper functions to simplify common setups, including CRRA/log-utility (extract Euler equation,
        #     labor supply curve, etc), and common production functions (CES, CD -- extract demand curves, prices, or
        #     marginal costs)
        sub_dict = dict()
        if self.system_equations is None:
            self.system_equations = []

        if self.definitions is not None:
            _, definitions = unpack_keys_and_values(self.definitions)
            sub_dict = {eq.lhs: eq.rhs for eq in definitions}

        if self.identities is not None:
            _, identities = unpack_keys_and_values(self.identities)
            for eq in identities:
                self.system_equations.append(
                    set_equality_equals_zero(eq.subs(sub_dict))
                )

        if self.constraints is not None:
            eq_idx, constraints = unpack_keys_and_values(self.constraints)
            for idx, eq in zip(eq_idx, constraints):
                if not self.equation_flags[idx].get("exclude", False):
                    self.system_equations.append(
                        set_equality_equals_zero(eq.subs(sub_dict))
                    )

        if self.controls is None and self.objective is None:
            return

        # Solve Lagrangian
        controls = self.controls
        obj_idx, objective = unpack_keys_and_values(self.objective)
        obj_idx, objective = obj_idx[0], objective[0]

        self.system_equations.append(set_equality_equals_zero(objective.subs(sub_dict)))

        _, multipliers = unpack_keys_and_values(self.multipliers)

        discount_factor = self._get_discount_factor()
        lagrange = self._build_lagrangian()

        # Corner case, if the objective function has a named lagrange multiplier
        # (pointless? but done in some gEcon example GCN files)
        if multipliers[obj_idx] is not None:
            raise NotImplementedError(
                "Lagrange multipliers in the objective block is not currently supported. This may break backwards"
                "compatibility with some gEcon example GCN files. Re-write the model to directly define the stochastic"
                "discount factor."
            )
            # self.system_equations.append(
            #     multipliers[obj_idx] - diff_through_time(lagrange, objective.lhs, discount_factor)
            # )

        for control in controls:
            foc = diff_through_time(lagrange, control, discount_factor)
            self.system_equations.append(foc.powsimp())

        if try_simplify:
            self.simplify_system_equations()

        # Update the variable list
        self._get_variable_list()

    def __html_repr__(self) -> str:
        """
        Return an HTML representation of the block.

        The block is rendered as a collapsible section with collapsible sub-sections for each component (definitions,
        controls, objective, constraints, identities, shocks, calibration).
        """

        html_parts = []
        html_parts.append(
            f"<details class='block-info'><summary class='block-title'>Block: {self.name}</summary>"
        )
        html_parts.append("<div class='block-content'>")
        prop_names = [
            "definitions",
            "controls",
            "objective",
            "constraints",
            "identities",
            "shocks",
            "calibration",
        ]
        properties = {}
        for prop in prop_names:
            value = getattr(self, prop)
            prop = prop.title()
            if value is None:
                continue
            elif isinstance(value, list):
                properties[prop] = [sp.Set([sp.cancel(x) for x in value])]
            elif isinstance(value, dict):
                properties[prop] = [sp.cancel(x) for x in value.values()]
            else:
                raise ValueError(f"Unexpected type for property {prop}")

        for prop_label, prop in properties.items():
            html_parts.append(
                f"<details class='property-details'><summary>{prop_label}</summary>"
            )
            for item in prop:
                latex_repr = f"\\({sp.latex(item)}\\)"
                html_parts.append(f"<p>{latex_repr}</p>")
            html_parts.append("</details>")

        html_parts.append("</div>")  # close block-content
        html_parts.append("</details>")  # close block-info

        return "\n".join(html_parts)
