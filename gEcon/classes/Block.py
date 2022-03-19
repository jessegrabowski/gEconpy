from gEcon.parser import gEcon_parser, parse_equations
from gEcon.classes.TimeAwareSymbol import TimeAwareSymbol
from gEcon.classes.utilities import diff_through_time, unpack_keys_and_values, set_equality_equals_zero
from gEcon.exceptions.exceptions import BlockNotInitializedException, DynamicCalibratingEquationException, \
    OptimizationProblemNotDefinedException, MultipleObjectiveFunctionsException, ControlVariableNotFoundException

import sympy as sp
from typing import List, Tuple, Optional, Union, Dict


class Block:
    """
    The Block class holds equations and parameters associated with each block of the DSGE model. They hold methods
    to solve their associated optimization problem. Blocks should be created by a Model.

    TODO: Refactor this into an abstract class with basic functionality, then create some child classes for specific
    problems, e.g. IdentityBlock, OptimizationBlock, CRRABlock, etc, each with their own optimization machinery.

    TODO: Split components out into their own class/protocol and let them handle their own parsing?
    """

    def __init__(self, name: str, block_dict: dict, solution_hints: Optional[Dict[str, str]] = None,
                 allow_incomplete_initialization: bool = False) -> None:
        """
        :param name: str, the name of the block
        :param block_dict: dict, dictionary of component:List[equations] key-value pairs created by
                           gEcon_parser.parsed_block_to_dict.
        :param solution_hints: dict, if not None, a dictionary of flags that help the solve_optimization method combine
                               the FoC into the "expected" solution. Currently unused.

        :param allow_incomplete_initialization: bool, default: False. If True, the block will not raise an exception if
                an error in the block's implementation is encountered.

        TODO: Implement solution hints as a dictionary of functions with fixed FoC "patterns", such as "CES", "CRRA",
        and "CD", that will allow the solver_optimization method to return the solved system in the "expected" way.
        """
        self.name = name
        self.short_name = ''.join(word[0] for word in name.split('_'))

        self.definitions: Optional[Dict[int, sp.Add]] = None
        self.controls: Optional[List[TimeAwareSymbol]] = None
        self.objective: Optional[Dict[int, sp.Add]] = None
        self.constraints: Optional[Dict[int, sp.Add]] = None
        self.identities: Optional[Dict[int, sp.Add]] = None
        self.shocks: Optional[Dict[int, TimeAwareSymbol]] = None
        self.calibration: Optional[Dict[int, sp.Add]] = None

        self.variables: List[TimeAwareSymbol] = []
        self.param_dict: Dict[str, float] = {}

        self.params_to_calibrate: Optional[List[sp.Symbol]] = None
        self.calibrating_equations: Optional[List[sp.Add]] = None

        self.system_equations: List[sp.Add] = []
        self.multipliers: Dict[int, TimeAwareSymbol] = {}

        self.n_equations = 0
        self.initialized = False

        self.initialize_from_dictionary(block_dict)
        self._get_variable_list()
        self._get_param_dict_and_calibrating_equations()

    def __str__(self):
        return f'{self.name} Block of {self.n_equations} equations, initialized: {self.initialized}, ' \
               f'solved: {self.system_equations is not None}'

    def initialize_from_dictionary(self, block_dict: dict) -> None:
        """
        :param block_dict: dict, dictionary of component:List[equations] key-value pairs created by
                           gEcon_parser.parsed_block_to_dict.
        :return: None
        """
        self.controls = self._parse_variable_list(block_dict, 'controls')
        self.shocks = self._parse_variable_list(block_dict, 'shocks')

        self.definitions = self._parse_equation_list(block_dict, 'definitions')
        self.objective = self._parse_equation_list(block_dict, 'objective')
        self.constraints = self._parse_equation_list(block_dict, 'constraints')
        self.identities = self._parse_equation_list(block_dict, 'identities')
        self.calibration = self._parse_equation_list(block_dict, 'calibration')

        self.initialized = self._validate_initialization()

    def _validate_initialization(self) -> bool:
        """
        :return: bool, indicates whether the block has been successfully initialized.

        At a high level, gEcon allows for two kinds of blocks, those with and those without an optimization problem.
        To have an optimization problem, we need the controls and objective components to be present. In addition, all
        control variables need to be represented among the equations in objective, definitions, and
        """

        if self.objective is not None and self.controls is None:
            raise OptimizationProblemNotDefinedException(block_name=self.name, missing='controls')

        if self.objective is None and self.controls is not None:
            raise OptimizationProblemNotDefinedException(block_name=self.name, missing='objective')

        if self.objective is not None and len(list(self.objective.values())) > 1:
            raise MultipleObjectiveFunctionsException(block_name=self.name, eqs=list(self.objective.values()))

        if self.controls is not None:
            for control in self.controls:
                control_found = False
                eq_dicts = [x for x in [self.definitions, self.objective, self.constraints] if x is not None]
                for eq_dict in eq_dicts:
                    for eq in list(eq_dict.values()):
                        if control in eq.atoms():
                            control_found = True
                            break
                if not control_found:
                    raise ControlVariableNotFoundException(self.name, control)

        return True

    def _validate_key(self, block_dict: dict, key: str) -> bool:
        """
        :param block_dict: dict, dictionary of component:List[equations] key-value pairs created by
                           gEcon_parser.parsed_block_to_dict.
        :param key: str, a component name.
        :return: bool

        Check whether a block component is present in the block_dict, and a valid component name. For valid component
        names, see gEcon_parser.BLOCK_COMPONENTS.
        """

        return key in block_dict and hasattr(self, key) and block_dict[key] is not None

    def _extract_lagrange_multipliers(self, equations: List[List[str]]) -> Tuple[List[List[str]],
                                                                                 List[Union[TimeAwareSymbol, None]]]:
        """
        :param equations: list, a List of Lists of strings, each list representing a model equation. Created by the
                          gEcon_parser.parsed_block_to_dict function.
        :return: tuple, List of List of strings, and a list of Union[TimeAwareSymbols, None].

        gEcon allows the user to name lagrange multipliers in the GCN file. These multiplier variables need to be saved
        and used once the optimization problem is solved. This function removes the ": muliplier[]" from each equation
        and returns them as a list, along with the new equations. A None is placed in the list for each equation
        with no associated multiplier.
        """

        result, multipliers = [], []
        for eq in equations:
            if ':' in eq:
                colon_idx = eq.index(':')
                multiplier = eq[-1]
                multiplier = parse_equations.single_symbol_to_sympy(multiplier)
                eq = eq[:colon_idx].copy()

                result.append(eq)
                multipliers.append(multiplier)
            else:
                result.append(eq)
                multipliers.append(None)

        return result, multipliers

    def _parse_variable_list(self, block_dict: dict, key: str) -> Optional[List[sp.Symbol]]:
        """
        :param block_dict: list, a List of Lists of strings, each list representing a model equation. Created by the
                          gEcon_parser.parsed_block_to_dict function.
        :param key: str, a component name.
        :return: list, a list of variables, represented as Sympy objects, or None if the block does not exist.

        Two components -- controls and shocks -- expect a simple list of variables, which is a case the
        gEcon_parser.build_sympy_equations cannot handle.
        """
        if not self._validate_key(block_dict, key):
            return

        raw_list = block_dict[key][0]
        output = []
        for variable in raw_list:
            variable = parse_equations.single_symbol_to_sympy(variable)
            output.append(variable)

        return output

    def _get_variable_list(self) -> None:
        """
        :return: None
        Get a list of all unique variables in the Block and store it in the class attribute "variables"
        """
        objective, constraints, identities = [], [], []
        sub_dict = {}
        if self.definitions is not None:
            _, definitions = unpack_keys_and_values(self.definitions)
            sub_dict = {eq.lhs:eq.rhs for eq in definitions}

        if self.objective is not None:
            _, objective = unpack_keys_and_values(self.objective)

        if self.constraints is not None:
            _, constraints = unpack_keys_and_values(self.constraints)

        if self.identities is not None:
            _, identities = unpack_keys_and_values(self.identities)

        all_equations = [eq for l in [objective, constraints, identities] for eq in l]
        for eq in all_equations:
            eq = eq.subs(sub_dict)
            atoms = eq.atoms()
            variables = [x for x in atoms if isinstance(x, TimeAwareSymbol)]
            for variable in variables:
                if variable.to_ss() not in self.variables:
                    self.variables.append(variable.to_ss())

    def _get_and_record_equation_numbers(self, equations: List[sp.Eq]) -> range:
        """
        :param equations:  list of Sympy equations representing a block
        :return: range, a python range object with integers corresponding to these equations

        In addition to numbering the equations, this function also maintains the internal count of how many equations
        are in the block, to allow consistent numbers across components.
        """
        n_equations = len(equations)
        equation_numbers = range(self.n_equations, self.n_equations + n_equations)
        self.n_equations += n_equations

        return equation_numbers

    def _parse_equation_list(self, block_dict: dict, key: str) -> Optional[Dict[int, sp.Eq]]:
        """
        :param block_dict: list, a List of Lists of strings, each list representing a model equation. Created by the
                          gEcon_parser.parsed_block_to_dict function.
        :param key: str, a component name.
        :return: list, a list of equations, represented as Sympy objects, or None if the block does not exist.
        """
        if not self._validate_key(block_dict, key):
            return

        equations = block_dict[key]
        equations, lagrange_multipliers = self._extract_lagrange_multipliers(equations)

        equations = parse_equations.build_sympy_equations(equations)
        equation_numbers = self._get_and_record_equation_numbers(equations)

        equations = dict(zip(equation_numbers, equations))
        lagrange_multipliers = dict(zip(equation_numbers, lagrange_multipliers))
        self.multipliers.update(lagrange_multipliers)

        return equations

    def _get_param_dict_and_calibrating_equations(self) -> None:
        """
        :return: None

        The calibration block, as implemented in gEcon, mixes together parameters, which are fixed values with a
        user-provided value, with calibrating equations, which are extra conditions added to the steady-state system.
        This function divides these out so that the Model instance can ask for only one or the other.

        These are divided heuristically: a parameter is assumed to be an equation with up to three atoms, all of which
        are of class sp.Symbol or sp.Number. Calibrating equations, on the other hand, are comprised of Symbols,
        TimeAwareSymbols, and numbers. All TimeAwareSymbols must be in the steady state, or else a Exception will be
        raised.
        """
        if not self.initialized:
            raise BlockNotInitializedException(block_name=self.name)

        # It is possible that an initialized block will not have a calibration component
        if self.calibration is None:
            return

        _, equations = unpack_keys_and_values(self.calibration)
        for eq in equations:
            atoms = eq.atoms()

            # Check if this equation is a normal parameter definition
            if len(atoms) <= 3 and all([not isinstance(x, TimeAwareSymbol) for x in atoms]):
                param = eq.lhs
                value = eq.rhs
                self.param_dict.update({param: value})

            # Check if this equation is a valid calibrating equation
            elif all([isinstance(x, (sp.Number, sp.Symbol, TimeAwareSymbol)) for x in atoms]):
                if not all([x.time_index == 'ss' for x in atoms if isinstance(x, TimeAwareSymbol)]):
                    raise DynamicCalibratingEquationException(eq=eq, block_name=self.name)

                if self.params_to_calibrate is None:
                    self.params_to_calibrate = [eq.lhs]
                else:
                    self.params_to_calibrate.append(eq.lhs)

                if self.calibrating_equations is not None:
                    self.calibrating_equations.append(set_equality_equals_zero(eq.rhs))
                else:
                    self.calibrating_equations = [set_equality_equals_zero(eq.rhs)]

    def _build_lagrangian(self) -> sp.Add:
        """
        :return: sp.Add, a SymPy equation representing the Lagrangian of the optimization problem

        The lagrangian function is built from the objective function, by recursively subtracting all constraints times
        the respective lagrange multiplier. If a lagrange multiplier is not specified, then it is assigned one
        automatically.
        """
        objective = list(self.objective.values())[0]
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
                lm = TimeAwareSymbol(f'lambda__{self.short_name}_{i}', 0)
                i += 1

            lagrange = lagrange - lm * (constraint.lhs.subs(sub_dict) - constraint.rhs.subs(sub_dict))

        return lagrange

    def _get_discount_factor(self) -> Optional[sp.Symbol]:
        """
        :return: sp.Symbol, the Bellman equation discount factor
        Optimization problems that are done over an infinite time horizon, for example that of the household, are
        represented in the GCN file as Bellman equations of the form:
        X[] = a[] + b * E[][X[1]];

        Where a[] is the value of the objective function at time t, and E[][X[1]] is the expected continuation value
        conditioned on the current information set. The parameter 0 < b < 1, then, is a discount factor that ensures
        the Bellman equation converges to a fixed point. This parameter should be extracted and used by the
        diff_through_time function to correctly build the lagrange in cases where controls appear inside the
        continuation value, for example capital stock at time t+1 in the RBC model.

        For single period optimizations, the discount factor should be 1.

        TODO: This function currently assumes the continuation value is a single variable, it will fail in the case of
        TODO: something like X[] = a[] + b * E[][Y[1] + Z[1]], although i don't know how such a function could arise?
        """

        _, objective = unpack_keys_and_values(self.objective)
        objective = objective[0]

        variables = [x for x in objective.atoms() if isinstance(x, TimeAwareSymbol)]

        # Return 1 if there is no continuation value
        if all([x.time_index in [0, -1] for x in variables]):
            return 1

        else:
            continuation_value = [x for x in variables if x.time_index == 1]
            if len(continuation_value) > 1:
                raise ValueError(f'Block {self.name} has multiple t+1 variables in the Bellman equation, this is not'
                                 f'currently supported. Rewrite the equation in the form X[] = a[] + b * E[][X[1]],'
                                 f'where a[] is the instantaneous value function at time t, defined in the'
                                 f'"definitions" component of the block.')
            discount_factor = objective.rhs.coeff(continuation_value[0])
            return discount_factor

    def simplify_system_equations(self) -> None:
        """
        :return: None
        Apply simplifications to the FoCs. For now this is just a heuristic check to eliminate redundant Lagrange
        multipliers generated by the solver. User-named lagrange multipliers might also be redundant, but they aren't
        removed, following the example of gEcon.

        TODO: Add solution patterns for CES, CRRA, and CD functions. Check parameter values to allow CES to collapse
        TODO: to CD, and CRRA to log-Utility?
        """
        system = self.system_equations
        simplified_system = system.copy()
        variables = [x for eq in system for x in eq.atoms() if isinstance(x, TimeAwareSymbol)]
        generated_multipliers = list(set([x for x in variables if 'lambda__' in x.base_name]))

        # Strictly heuristic simplification: look for an equation of the form x = y and use it to substitute away
        # the generated multipliers.

        for x in generated_multipliers:
            candidates = [eq for eq in simplified_system if x in eq.atoms()]
            for eq in candidates:
                # x = y will have 2 atoms, x = -y will have 3
                if len(eq.atoms()) <= 3:
                    sub_dict = sp.solve(eq, x, dict=True)[0]
                    simplified_system = [eq.subs(sub_dict) for eq in simplified_system]
                    break

        simplified_system = [eq for eq in simplified_system if eq != 0]

        self.system_equations = simplified_system

    def solve_optimization(self, try_simplify: bool = True) -> None:
        """
        :return: None

        Solve the optimization problem implied by the block structure:
           max  Sum_{t=0}^\infty [Objective] subject to [Constraints]
        [Controls]

        By setting up the following Lagrangian:
        $L = Sum_{t=0}^\infty Objective - lagrange_multiplier[1] * constraint[1] - ... - lagrange_multiplier[n] * constraint[n]
        And taking the derivative with respect to each control variable in turn.

        All first order conditions, along with the constraints and objective are stored in the .system_equations method.
        No attempt is made to simplify the resulting system.
        TODO: Add helper functions to simplify common setups, including CRRA/log-utility (extract Euler equation,
            labor supply curve, etc), and common production functions (CES, CD -- extract demand curves, prices, or
            marginal costs)

        TODO: Automatically solving for un-named lagrange multipliers is currently done by the Model class, is this
                correct?
        """
        if not self.initialized:
            raise ValueError(f'Block {self.name} is not initialized, cannot call Block.solve_optimization() '
                             f'before initialization')

        sub_dict = dict()

        if self.definitions is not None:
            _, definitions = unpack_keys_and_values(self.definitions)
            sub_dict = {eq.lhs: eq.rhs for eq in definitions}

        if self.identities is not None:
            _, identities = unpack_keys_and_values(self.identities)
            for eq in identities:
                self.system_equations.append(set_equality_equals_zero(eq.subs(sub_dict)))

        if self.constraints is not None:
            _, constraints = unpack_keys_and_values(self.constraints)
            for eq in constraints:
                self.system_equations.append(set_equality_equals_zero(eq.subs(sub_dict)))

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
            self.system_equations.append(multipliers[obj_idx] - diff_through_time(lagrange,
                                                                                  objective.lhs,
                                                                                  discount_factor))

        for control in controls:
            foc = diff_through_time(lagrange, control, discount_factor)
            self.system_equations.append(foc.powsimp())

        if try_simplify:
            self.simplify_system_equations()
