import sympy as sp

from gEconpy.classes.containers import SymbolDictionary
from gEconpy.classes.time_aware_symbol import DEFAULT_ASSUMPTIONS, TimeAwareSymbol
from gEconpy.exceptions import (
    ControlVariableNotFoundException,
    DuplicateParameterError,
    DynamicCalibratingEquationException,
    MultipleObjectiveFunctionsException,
    OptimizationProblemNotDefinedException,
)
from gEconpy.parser.errors import ParseLocation
from gEconpy.utilities import (
    diff_through_time,
    expand_subs_for_all_times,
    flatten_substitution_dict,
    set_equality_equals_zero,
    step_equation_backward,
    step_equation_forward,
    unpack_keys_and_values,
)

_TARGET_TIME_INDICES = (-1, 0, 1)

_FLAG_TO_ALLOWED_COMPONENTS = {
    "is_calibrating": ["calibration"],
    "exclude": ["constraints"],
    "minimize": ["objective"],
    "maximize": ["objective"],
}

_N_ATOMS_IN_DIRECT_DEFINITION = 3


class Block:
    r"""
    One block of a DSGE model: its equations, parameters, and the optimization problem they imply.

    A block with both ``controls`` and an ``objective`` is an optimization problem. :meth:`solve_optimization`
    builds its Lagrangian and differentiates it with respect to each control. A block with neither is a set of
    identities.

    Parameters
    ----------
    name : str
        The name of the block.
    definitions : dict mapping int to sympy.Eq, optional
        Definition equations, keyed by equation number.
    controls : list of TimeAwareSymbol, optional
        Control variables of the optimization problem.
    objective : dict mapping int to sympy.Eq, optional
        The objective equation, keyed by equation number. Exactly one entry is allowed.
    constraints : dict mapping int to sympy.Eq, optional
        Constraint equations, keyed by equation number.
    identities : dict mapping int to sympy.Eq, optional
        Identity equations, keyed by equation number.
    calibration : dict mapping int to sympy.Eq, optional
        Calibration equations, keyed by equation number.
    shocks : list of TimeAwareSymbol, optional
        Shock variables.
    multipliers : dict mapping int to TimeAwareSymbol or None, optional
        The Lagrange multiplier on each constraint, keyed by constraint index. Entries of None get a generated
        multiplier named ``lambda__<short_name>_<i>``.
    equation_flags : dict mapping int to dict, optional
        The flag dictionary of each equation, keyed by equation number.
    source : str, optional
        The source code of the GCN file, for error reporting.
    symbol_locations : dict mapping str to ParseLocation, optional
        The location of each symbol in ``source``, for error reporting.
    ss_solution_dict : SymbolDictionary, optional
        Analytically known steady-state solutions, used to resolve calibration expressions that reference
        steady-state variables such as ``phi_B = f(Y[ss])``.

    Examples
    --------
    The parser builds one of these for every block in a GCN file, so the constructor is only needed when
    assembling a model programmatically:

    .. code-block:: python

        import sympy as sp

        from gEconpy.classes.time_aware_symbol import TimeAwareSymbol
        from gEconpy.model.block import Block

        U, U_next = TimeAwareSymbol("U", 0), TimeAwareSymbol("U", 1)
        C, L, w = (TimeAwareSymbol(name, 0) for name in ("C", "L", "w"))
        beta = sp.Symbol("beta")

        block = Block(
            name="HOUSEHOLD",
            objective={0: sp.Eq(U, sp.log(C) - L + beta * U_next)},
            constraints={1: sp.Eq(C, w * L)},
            controls=[C, L],
            multipliers={0: None, 1: None},
            calibration={2: sp.Eq(beta, sp.Float(0.99))},
            equation_flags={0: {}, 1: {}, 2: {"is_calibrating": False}},
        )
        print(block.variables, block.param_dict)
    """

    def __init__(
        self,
        name: str,
        definitions: dict[int, sp.Eq] | None = None,
        controls: list[TimeAwareSymbol] | None = None,
        objective: dict[int, sp.Eq] | None = None,
        constraints: dict[int, sp.Eq] | None = None,
        identities: dict[int, sp.Eq] | None = None,
        calibration: dict[int, sp.Eq] | None = None,
        shocks: list[TimeAwareSymbol] | None = None,
        multipliers: dict[int, TimeAwareSymbol | None] | None = None,
        equation_flags: dict[int, dict[str, bool]] | None = None,
        source: str | None = None,
        symbol_locations: dict[str, ParseLocation] | None = None,
        ss_solution_dict: SymbolDictionary | None = None,
    ) -> None:
        self.name = name
        self.short_name = "".join(word[0] for word in name.split("_"))

        self.definitions = definitions
        self.controls = controls
        self.objective = objective
        self.constraints = constraints
        self.identities = identities
        self.shocks = shocks
        self.calibration = calibration

        self.variables: list[TimeAwareSymbol] = []
        self.param_dict: SymbolDictionary[str, float] = SymbolDictionary()
        self.calib_dict: SymbolDictionary[str, float] = SymbolDictionary()
        self.deterministic_dict: SymbolDictionary[str, float] = SymbolDictionary()

        self.system_equations: list[sp.Expr] = []
        self.multipliers = multipliers or {}
        self.eliminated_variables: list[sp.Symbol] = []
        self.equation_flags = equation_flags or {}

        self._source = source
        self._symbol_locations = symbol_locations or {}
        self._ss_solution_dict = ss_solution_dict

        self.n_equations = sum(
            len(eq_dict) if eq_dict else 0 for eq_dict in [definitions, objective, constraints, identities, calibration]
        )

        self.initialized = self._validate_initialization()
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
        """Parameters defined by a deterministic relationship to other parameters."""
        return list(self.deterministic_dict.to_sympy().keys())

    @property
    def deterministic_relationships(self) -> list[sp.Expr]:
        """Expressions defining each deterministic parameter, in the order of ``deterministic_params``."""
        return list(self.deterministic_dict.values())

    @property
    def params_to_calibrate(self) -> list[sp.Symbol]:
        """Parameters whose values are pinned by a steady-state calibration target."""
        return list(self.calib_dict.to_sympy().keys())

    @property
    def calibrating_equations(self) -> list[sp.Expr]:
        """Calibration equations, in the order of ``params_to_calibrate``."""
        return list(self.calib_dict.values())

    def solve_optimization(self, try_simplify: bool = True) -> None:
        r"""
        Derive the block's system equations, including the first-order conditions of its optimization problem.

        The block structure implies the program

        .. math::

            \max_{\text{controls}} \sum_{t=0}^{\infty} \text{objective} \quad
            \text{subject to} \quad \text{constraints}

        with Lagrangian

        .. math::

            \mathcal{L} = \sum_{t=0}^{\infty} \text{objective}
                - \lambda_1 \, \text{constraint}_1 - \dots - \lambda_n \, \text{constraint}_n.

        Differentiating :math:`\mathcal{L}` with respect to each control gives one first-order condition per control.
        An objective tagged ``@minimize`` is negated before the Lagrangian is formed, so the first-order conditions
        are those of the minimization program.

        The identities, the constraints not tagged ``@exclude``, the objective, and the first-order conditions are
        stored in ``system_equations``, with definitions substituted in. A block with no optimization problem stores
        only its identities and constraints.

        Parameters
        ----------
        try_simplify : bool, optional
            Whether to run :meth:`simplify_system_equations` on the result. Defaults to True.

        Examples
        --------
        A household choosing consumption and labor subject to a budget constraint yields one first-order condition
        per control:

        .. code-block:: python

            import sympy as sp

            from gEconpy.classes.time_aware_symbol import TimeAwareSymbol
            from gEconpy.model.block import Block

            U, U_next = TimeAwareSymbol("U", 0), TimeAwareSymbol("U", 1)
            C, L, w = (TimeAwareSymbol(name, 0) for name in ("C", "L", "w"))
            beta = sp.Symbol("beta")

            block = Block(
                name="HOUSEHOLD",
                objective={0: sp.Eq(U, sp.log(C) - L + beta * U_next)},
                constraints={1: sp.Eq(C, w * L)},
                controls=[C, L],
                multipliers={0: None, 1: None},
                equation_flags={0: {}, 1: {}},
            )
            block.solve_optimization()
            print(block.system_equations)
        """
        sub_dict = {}

        self.system_equations = []

        if self.definitions is not None:
            _, definitions = unpack_keys_and_values(self.definitions)
            sub_dict = {eq.lhs: eq.rhs for eq in definitions}

        if self.identities is not None:
            _, identities = unpack_keys_and_values(self.identities)
            for eq in identities:
                self.system_equations.append(set_equality_equals_zero(eq.subs(sub_dict)))

        if self.constraints is not None:
            eq_idx, constraints = unpack_keys_and_values(self.constraints)
            for idx, eq in zip(eq_idx, constraints, strict=True):
                if not self.equation_flags[idx].get("exclude", False):
                    self.system_equations.append(set_equality_equals_zero(eq.subs(sub_dict)))

        if self.controls is None and self.objective is None:
            return

        obj_idx, objective = unpack_keys_and_values(self.objective)
        obj_idx, objective = obj_idx[0], objective[0]

        self.system_equations.append(set_equality_equals_zero(objective.subs(sub_dict)))

        _, multipliers = unpack_keys_and_values(self.multipliers)

        discount_factor = self._get_discount_factor()
        lagrange = self._build_lagrangian()

        if multipliers[obj_idx] is not None:
            raise NotImplementedError(
                "Lagrange multipliers on the objective equation are not supported. Rewrite the model to define the "
                "stochastic discount factor directly."
            )

        for control in self.controls:
            foc = self._compute_foc(control, lagrange, discount_factor)
            self.system_equations.append(foc.powsimp())

        if try_simplify:
            self.simplify_system_equations()

        self._get_variable_list()

    def simplify_system_equations(self) -> None:
        """
        Eliminate generated Lagrange multipliers from ``system_equations`` and canonicalize powers.

        A generated multiplier (named ``lambda__*``) that appears in a two-term identity ``x = y`` or ``x = -y`` is
        solved for and substituted out of every equation. User-named multipliers stay, matching gEcon. Every
        remaining equation is then passed through :func:`sympy.powsimp` to collapse the ``x**e / x`` patterns that
        chain-rule differentiation leaves behind.

        Examples
        --------
        With ``try_simplify=False`` the generated multiplier ``lambda__H_1`` stays in the system until this method
        removes it:

        .. code-block:: python

            import sympy as sp

            from gEconpy.classes.time_aware_symbol import TimeAwareSymbol
            from gEconpy.model.block import Block

            U, U_next = TimeAwareSymbol("U", 0), TimeAwareSymbol("U", 1)
            C, L, w = (TimeAwareSymbol(name, 0) for name in ("C", "L", "w"))
            beta = sp.Symbol("beta")

            block = Block(
                name="HOUSEHOLD",
                objective={0: sp.Eq(U, sp.log(C) - L + beta * U_next)},
                constraints={1: sp.Eq(C, w * L)},
                controls=[C, L],
                multipliers={0: None, 1: None},
                equation_flags={0: {}, 1: {}},
            )
            block.solve_optimization(try_simplify=False)
            print(block.system_equations)

            block.simplify_system_equations()
            print(block.system_equations, block.eliminated_variables)
        """
        system = self.system_equations
        simplified_system = system.copy()
        variables = [x for eq in system for x in eq.atoms() if isinstance(x, TimeAwareSymbol)]
        generated_multipliers = list({x for x in variables if "lambda__" in x.base_name})

        eliminated_variables = []
        for multiplier in generated_multipliers:
            candidates = [eq for eq in simplified_system if multiplier in eq.atoms()]
            for eq in candidates:
                if len(eq.atoms()) <= _N_ATOMS_IN_DIRECT_DEFINITION:
                    sub_dict = sp.solve(eq, multiplier, dict=True)[0]
                    sub_dict = expand_subs_for_all_times(sub_dict)
                    eliminated_variables.extend(list(sub_dict.keys()))
                    simplified_system = [eq.subs(sub_dict) for eq in simplified_system]
                    break

        simplified_system = [eq for eq in simplified_system if eq != 0]
        simplified_system = [sp.powsimp(eq) for eq in simplified_system]

        self.system_equations = simplified_system
        self.eliminated_variables = eliminated_variables

        for key, value in self.multipliers.items():
            if value in eliminated_variables:
                self.multipliers[key] = None

    def _validate_initialization(self) -> bool:
        """
        Check that the block is well formed.

        An optimization problem needs both ``controls`` and ``objective``, exactly one objective equation, and every
        control appearing somewhere in ``objective``, ``definitions``, or ``constraints``. Equation flags may only
        appear on the component they apply to.
        """
        if self.objective is not None and self.controls is None:
            raise OptimizationProblemNotDefinedException(block_name=self.name, missing="controls")

        if self.objective is None and self.controls is not None:
            raise OptimizationProblemNotDefinedException(block_name=self.name, missing="objective")

        if self.objective is not None and len(list(self.objective.values())) > 1:
            raise MultipleObjectiveFunctionsException(block_name=self.name, eqs=list(self.objective.values()))

        if self.controls is not None:
            equation_dicts = [d for d in (self.definitions, self.objective, self.constraints) if d is not None]
            for control in self.controls:
                control_found = any(control in eq.atoms() for eq_dict in equation_dicts for eq in eq_dict.values())
                if not control_found:
                    location = self._symbol_locations.get(str(control.base_name))
                    raise ControlVariableNotFoundException(
                        self.name,
                        control,
                        source=self._source,
                        location=location,
                    )

        components = {
            "definitions": self.definitions,
            "objective": self.objective,
            "constraints": self.constraints,
            "identities": self.identities,
        }
        for component_name, equations in components.items():
            if equations is None:
                continue
            for key, eq in equations.items():
                for flag, allowed_components in _FLAG_TO_ALLOWED_COMPONENTS.items():
                    if self.equation_flags[key].get(flag, False) and component_name not in allowed_components:
                        raise ValueError(
                            f"Equation {eq} in {component_name} block of {self.name} has an invalid decorator: "
                            f"{flag}. This flag should only appear in the {allowed_components[0]} block."
                        )

        if self.objective is not None:
            for key in self.objective:
                flags = self.equation_flags[key]
                if flags.get("minimize", False) and flags.get("maximize", False):
                    raise ValueError(
                        f"Objective equation in block {self.name} has both @minimize and @maximize tags. Use only one."
                    )

        return True

    def _consolidate_definitions(self) -> None:
        """Substitute definitions that refer to other definitions until each depends on no other."""
        if self.definitions is None:
            return

        sub_dict = flatten_substitution_dict({eq.lhs: eq.rhs for eq in self.definitions.values()})

        self.definitions = {k: sp.Eq(v.lhs, v.rhs.subs(sub_dict)) for k, v in self.definitions.items()}

    def _get_variable_list(self) -> None:
        """Collect every steady-state variable of the block into ``variables``, sorted by name."""
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

        if self.multipliers is not None:
            _, multipliers = unpack_keys_and_values(self.multipliers)
            multipliers = [x for x in multipliers if x is not None]

        all_equations = [eq for eqs_list in [objective, constraints, identities] for eq in eqs_list]
        flat_sub_dict = flatten_substitution_dict(sub_dict) if sub_dict else {}
        for eq in all_equations:
            atoms = eq.subs(flat_sub_dict).atoms() if flat_sub_dict else eq.atoms()
            variables = [x for x in atoms if isinstance(x, TimeAwareSymbol)]
            for variable in variables:
                if variable.to_ss() not in self.variables:
                    self.variables.append(variable.to_ss())

        shocks = self.shocks or []
        self.variables = [*self.variables, *multipliers]
        self.variables = sorted(
            {x for x in self.variables if x.set_t(0) not in shocks},
            key=lambda x: x.name,
        )

    def _get_param_dict_and_calibrating_equations(self) -> None:
        """
        Split the calibration block into parameters, calibrating equations, and deterministic relationships.

        gEcon's calibration block mixes three things. A parameter is an equation ``x = y`` with ``x`` a plain
        ``sympy.Symbol`` and ``y`` a number. A calibrating equation is written with ``->`` in the GCN file, is
        flagged as such by the parser, and adds a condition to the steady-state system. Its variables must all be in
        the steady state. A deterministic relationship is a parameter defined as a function of other parameters, the
        analog of a ``#`` line in Dynare.
        """
        if self.calibration is None:
            return

        eq_idxs, equations = unpack_keys_and_values(self.calibration)
        duplicates = []

        for idx, eq in zip(eq_idxs, equations, strict=True):
            atoms = eq.atoms()
            lhs, rhs = eq.lhs, eq.rhs
            if not lhs.is_symbol:
                raise ValueError(
                    "Left-hand side of calibrating expressions should be the single parameter to be "
                    f"computed. Found multiple arguments: {eq.lhs.args}"
                )

            param = eq.lhs

            if eq.rhs.is_number:
                value = eq.rhs.evalf()
                if param in self.param_dict:
                    duplicates.append(param)
                else:
                    self.param_dict[param] = value

            elif self.equation_flags[idx]["is_calibrating"]:
                if not all(x.time_index == "ss" for x in atoms if isinstance(x, TimeAwareSymbol)):
                    location = self._symbol_locations.get(str(param))
                    raise DynamicCalibratingEquationException(
                        eq=eq, block_name=self.name, source=self._source, location=location
                    )

                if param in self.calib_dict:
                    duplicates.append(param)
                else:
                    self.calib_dict[param] = rhs

            else:
                ss_vars = [x for x in atoms if isinstance(x, TimeAwareSymbol)]
                if ss_vars:
                    rhs = self._try_substitute_ss_values(eq, rhs, ss_vars)
                    if any(isinstance(x, TimeAwareSymbol) for x in rhs.atoms()):
                        raise ValueError(
                            "Parameters defined as functions in the calibration sub-block cannot be functions "
                            f"of variables. Found:\n\n {eq} in {self.name}"
                        )

                if eq.lhs in self.deterministic_dict:
                    duplicates.append(lhs)
                else:
                    self.deterministic_dict[lhs] = rhs.doit()

        if len(duplicates) > 0:
            location = self._symbol_locations.get(str(duplicates[0]))
            raise DuplicateParameterError(duplicates, self.name, source=self._source, location=location)

    def _try_substitute_ss_values(
        self,
        eq: sp.Eq,
        rhs: sp.Expr,
        ss_vars: list[TimeAwareSymbol],
    ) -> sp.Expr:
        """
        Replace the steady-state variables in a deterministic calibration expression with their analytic values.

        Parameters
        ----------
        eq : sympy.Eq
            The calibration equation, used in error messages.
        rhs : sympy.Expr
            The right-hand side to substitute into.
        ss_vars : list of TimeAwareSymbol
            The steady-state variables found in ``rhs``.

        Returns
        -------
        substituted : sympy.Expr
            ``rhs`` as a function of parameters only.
        """
        if not all(x.time_index == "ss" for x in ss_vars):
            raise ValueError(
                "Parameters defined as functions in the calibration sub-block cannot be functions "
                f"of variables. Found:\n\n {eq} in {self.name}"
            )

        if not self._ss_solution_dict:
            raise ValueError(
                f"Calibration expression {eq} in {self.name} references steady-state variables "
                f"but no STEADY_STATE block with analytic solutions was found."
            )

        ss_sympy = self._ss_solution_dict.to_sympy()
        sub_dict = {}
        missing = []
        for var in ss_vars:
            matched = False
            for key, value in ss_sympy.items():
                if hasattr(key, "name") and key.name == var.name:
                    sub_dict[var] = value
                    matched = True
                    break
            if not matched:
                missing.append(var)

        if missing:
            names = ", ".join(str(v) for v in missing)
            raise ValueError(
                f"Calibration expression {eq} in {self.name} references steady-state variables "
                f"without analytic solutions: {names}. Provide analytic values in the STEADY_STATE block."
            )

        return rhs.subs(sub_dict)

    def _build_lagrangian(self) -> sp.Add:
        """Build the Lagrangian of the block's optimization program, generating multipliers where none were named."""
        objective = next(iter(self.objective.values()))
        obj_key = next(iter(self.objective.keys()))
        is_minimization = self.equation_flags[obj_key].get("minimize", False)

        constraints = self.constraints
        multipliers = self.multipliers
        sub_dict = {}

        if self.definitions is not None:
            for eq in self.definitions.values():
                sub_dict.update(_expand_definition_for_all_times(eq.lhs, eq.rhs))

        obj_rhs = objective.rhs.subs(sub_dict)
        if is_minimization:
            obj_rhs = -obj_rhs

        lagrange = obj_rhs
        next_generated_index = 1
        for key, constraint in constraints.items():
            if multipliers[key] is not None:
                lm = multipliers[key]
            else:
                lm = TimeAwareSymbol(f"lambda__{self.short_name}_{next_generated_index}", 0, **DEFAULT_ASSUMPTIONS)
                self.multipliers[next_generated_index] = lm
                next_generated_index += 1

            lagrange = lagrange - lm * (constraint.lhs.subs(sub_dict) - constraint.rhs.subs(sub_dict))

        return lagrange

    def _get_discount_factor(self) -> sp.Expr:
        """
        Extract the discount factor from a Bellman-form objective.

        A Bellman equation has the form ``X[] = a[] + b * E[][X[1]]``, where ``a[]`` is the instantaneous value and
        ``b`` is the discount factor. The continuation term must contain ``X[1]`` and nothing else at ``t+1``. A
        static objective, with no ``t+1`` variables, has discount factor 1.

        Returns
        -------
        discount_factor : sympy.Expr
            The coefficient on the continuation value.
        """
        _, objective = unpack_keys_and_values(self.objective)
        objective = objective[0]

        variables = [x for x in objective.atoms() if isinstance(x, TimeAwareSymbol)]

        if all(x.time_index in [0, -1] for x in variables):
            return sp.Float(1.0)

        current_value = objective.lhs
        continuation_value = [x for x in objective.rhs.args if x.has(current_value.set_t(1))]

        if len(continuation_value) == 0:
            raise ValueError(
                f"Block {self.name} did not find the continuation value of the current state value in the following"
                f"objective function: {objective}. Objectives should be written in the form "
                f"``V[t] = f(x[t]) + b[t] * E[V[t+1]]``, where V[t] is the current state value, f(x[t]) is the "
                f"instantaneous value function, and b[t] is the discount factor."
            )

        continuation_value = continuation_value[0]
        return continuation_value.subs({current_value.set_t(1): 1})

    def _compute_foc(self, control: TimeAwareSymbol, lagrange: sp.Expr, discount_factor: sp.Expr | int) -> sp.Expr:
        """
        Compute the first-order condition for one control variable.

        The default differentiates the Lagrangian through time with :func:`~gEconpy.utilities.diff_through_time`.
        Subclasses such as :class:`~gEconpy.model.block.cobb_douglas.CobbDouglasBlock` override this to emit a
        closed form for the constraint derivative, which is far smaller than the chain-rule expansion
        :func:`sympy.diff` produces.
        """
        return diff_through_time(lagrange, control, discount_factor)

    def __html_repr__(self) -> str:
        """Render the block as a collapsible HTML section with one sub-section per component."""
        html_parts = []
        html_parts.append(f"<details class='block-info'><summary class='block-title'>Block: {self.name}</summary>")
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
            if value is None:
                continue
            if isinstance(value, list):
                properties[prop.title()] = [sp.Set([sp.cancel(x) for x in value])]
            elif isinstance(value, dict):
                properties[prop.title()] = [sp.cancel(x) for x in value.values()]
            else:
                raise TypeError(f"Unexpected type for property {prop}")

        for prop_label, prop in properties.items():
            html_parts.append(f"<details class='property-details'><summary>{prop_label}</summary>")
            for item in prop:
                latex_repr = f"\\[{sp.latex(item)}\\]"
                html_parts.append(f"<p>{latex_repr}</p>")
            html_parts.append("</details>")

        html_parts.append("</div>")
        html_parts.append("</details>")

        return "\n".join(html_parts)


def _expand_definition_for_all_times(
    lhs: TimeAwareSymbol,
    rhs: sp.Expr,
) -> dict[TimeAwareSymbol, sp.Expr]:
    """
    Shift a definition ``X[t0] = f(...)`` to time indices -1, 0, and 1.

    Parameters
    ----------
    lhs : TimeAwareSymbol
        Left-hand side of the definition.
    rhs : sympy.Expr
        Right-hand side of the definition.

    Returns
    -------
    sub_dict : dict mapping TimeAwareSymbol to sympy.Expr
        The shifted right-hand side, keyed by the shifted left-hand side, for each target time index.
    """
    base_t = lhs.time_index
    sub_dict = {}

    for target_t in _TARGET_TIME_INDICES:
        offset = target_t - base_t
        shifted_lhs = lhs
        shifted_rhs = rhs

        if offset > 0:
            for _ in range(offset):
                shifted_lhs = step_equation_forward(shifted_lhs)
                shifted_rhs = step_equation_forward(shifted_rhs)
        elif offset < 0:
            for _ in range(-offset):
                shifted_lhs = step_equation_backward(shifted_lhs)
                shifted_rhs = step_equation_backward(shifted_rhs)

        sub_dict[shifted_lhs] = shifted_rhs

    return sub_dict
