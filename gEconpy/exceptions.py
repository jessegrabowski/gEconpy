import sympy as sp

from gEconpy.classes.time_aware_symbol import TimeAwareSymbol
from gEconpy.parser.constants import BLOCK_COMPONENTS
from gEconpy.solvers.gensys import interpret_gensys_output


class GCNSyntaxError(ValueError):
    def __init__(self, block_name: str, key: list[str]):
        self.block_name = block_name
        self.key = key

        message = (
            f"While parsing block {block_name}, found {' '.join(key)} outside of a block component. All "
            f"equations must be inside components with one of valid component names:"
            f" {', '.join(BLOCK_COMPONENTS)}"
        )

        super().__init__(message)


class DistributionParsingError(ValueError):
    def __init__(self, line):
        message = (
            f"While parsing parameter prior distributions, encountered an error in the following line:\n"
            f"{line}\n"
            f'Check that distribution is defined using the "~" operator according to the following syntax: '
            f'"parameter ~ d(a=, b=, ...) = initial_value;"'
        )

        super().__init__(message)


class MissingParameterValueException(ValueError):
    def __init__(self, parameter_name):
        message = (
            f"The prior distribution associated with parameter {parameter_name} does not appear to have an "
            f"initial value. Please declare an initial value for all parameters following the syntax: "
            f"parameter ~ d(a=, b=, ...) = initial_value;. If this is an exogenous shock, make sure it has "
            f"square brackets [] to indicate it is a variable and not a parameter."
        )

        super().__init__(message)


class InvalidComponentNameException(ValueError):
    def __init__(self, component_name: str, block_name: str, message=str) -> None:
        self.component_name = component_name
        self.block_name = block_name
        self.message = message

        super().__init__(message)


class BlockNotInitializedException(ValueError):
    def __init__(self, block_name: str) -> None:
        self.block_name = block_name
        self.message = (
            f"Block {self.name} called method _get_param_dict_and_calibrating_equations before "
            f"initialization. Call initialize_from_dictionary first."
        )

        super().__init__(self.message)


class DynamicCalibratingEquationException(ValueError):
    def __init__(self, eq: sp.Add, block_name: str):
        self.eq = eq
        self.block_name = block_name

        self.message = (
            f"In block {block_name}, calibrating equation {eq} has variables with non-steady-state "
            f"time values. All calibrating equations must be written in terms of steady-state, "
            f"in the form X[ss]."
        )

        super().__init__(self.message)


class OptimizationProblemNotDefinedException(ValueError):
    def __init__(self, block_name: str, missing: str) -> None:
        self.block_name = block_name
        self.missing = missing
        not_missing = "objective" if missing == "controls" else "controls"

        message = (
            f"Block {block_name} has a {missing} component but no {not_missing} component, verify whether"
            f"or not this block has an optimization problem."
        )

        super().__init__(message)


class MultipleObjectiveFunctionsException(ValueError):
    def __init__(self, block_name: str, eqs: list[sp.Expr]) -> None:
        self.block_name = block_name

        n_eqs = len(eqs)

        message = f"Block {block_name} appears to have multiple objectives, excepted just one but found {n_eqs}:\n"
        for eq in eqs:
            message += str(eq) + "\n"
        message += (
            " Only one objective function is supported. Please manually simplify the objective"
            " to a single function."
        )

        super().__init__(message)


class ControlVariableNotFoundException(ValueError):
    def __init__(self, block_name: str, control: TimeAwareSymbol):
        self.block_name = block_name
        self.control = control

        message = (
            f"Block {block_name} has {control} declared as a control variable, "
            f"but this variable was not found among model equations in components definitions,"
            f" objective, or constraints."
        )

        super().__init__(message)


class ModelUnknownParameterError(ValueError):
    def __init__(self, unknown_updates: list[str]):
        self.unknown_updates = unknown_updates

        message = (
            f"The following parameters were given new values, but do not exist in the model: "
            f"{', '.join(unknown_updates)}."
        )

        super().__init__(message)


class PerturbationSolutionNotFoundException(ValueError):
    def __init__(self):
        message = (
            "This operation cannot be completed until the model has a solved perturbation solution. Please "
            "call the .solve() method to solve for the policy function."
        )

        super().__init__(message)


class SteadyStateNotFoundError(ValueError):
    def __init__(self, equations):
        message = (
            "The provided steady-state values did not result in zero residuals for the following equations:\n"
            f"{', '.join(equations)}\n\nIf you used custom parameter values to compute the provided steady state, "
            f"you must also provide these parameter values to ``solve_model``."
        )

        super().__init__(message)


class MultipleSteadyStateBlocksException(ValueError):
    def __init__(self, ss_block_names: list[str]):
        message = (
            f"Found multiple blocks with reserved steady states names: {', '.join(ss_block_names)}. Please pass"
            f"only up to one steady state block, and do not name any blocks with the reserved steady states "
            f"names."
        )

        super().__init__(message)


class GensysFailedException(ValueError):
    def __init__(self, eu):
        message = interpret_gensys_output(eu)
        super().__init__(message)


class VariableNotFoundException(ValueError):
    def __init__(self, variable):
        var_name = variable.base_name
        message = f"Variable {var_name} was not found among model variables."

        super().__init__(message)


class InvalidDistributionException(ValueError):
    def __init__(self, variable, distribution_string):
        message = (
            f'The distribution associated with "{variable}", defined as "{distribution_string}", appears to have '
            f"a typo, please check the GCN file. Please also check that you have not supplied an initial "
            f"parameter value to an exogenous shock distribution, as in epsilon[] ~ N(mu=0, sd=1) = 0.5. Shock "
            f"distributions should NOT have an equals sign after the distribution definition."
        )

        super().__init__(message)


class RepeatedParameterException(ValueError):
    def __init__(self, d_str: str, parameter: str):
        message = (
            f'In {d_str}, the parameter "{parameter}" was '
            f"declared multiple times. Please check the GCN file for typos."
        )

        super().__init__(message)


class DistributionParameterNotFoundException(ValueError):
    def __init__(
        self,
        variable_name: str,
        d_name: str,
        param_name: str,
        valid_param_names: list[str],
        maybe_typo: str | None,
        best_guess: str | None,
    ):
        message = (
            f"No {param_name} parameter was found for the {d_name} distribution associated with model parameter "
            f'"{variable_name}". Valid aliases for {param_name} are: '
        )
        if len(valid_param_names) > 1:
            message += (
                ", ".join(valid_param_names[:-1]) + f", and {valid_param_names[-1]}."
            )
        else:
            message += f"{valid_param_names[0]}."

        if maybe_typo is not None and best_guess is not None:
            message += f"\n\nFound a similar alias to suspected typo: {maybe_typo}. Did you mean {best_guess}?"

        super().__init__(message)


class MultipleParameterDefinitionException(ValueError):
    def __init__(
        self, variable_name: str, d_name: str, param_name: str, result_list: list[str]
    ) -> None:
        message = (
            f'The {d_name} distribution associated with "{variable_name}" has multiple declarations for '
            f"{param_name}. Please pass only one of: "
        )
        message += ", ".join(result_list)

        super().__init__(message)


class UnusedParameterError(ValueError):
    def __init__(self, d_name: str, param_name: str) -> None:
        message = f"{d_name} distributions do not have a {param_name}; do not call this parse method."

        super().__init__(message)


class InvalidParameterException(ValueError):
    def __init__(self, dist_name, param_name, valid_params):
        message = (
            f"Unknown parameter {param_name} passed to distribution {dist_name}. Valid "
            f"parameters for this distribution are: {', '.join(valid_params)}"
        )

        super().__init__(message)


class InvalidMeanException(ValueError):
    def __init__(self, variable_name, d_name, mean, a, b):
        message = (
            f'The mean value of the {d_name} distribution associated with "{variable_name}" is invalid. Given '
            f"bounds [{a}, {b}], the requested mean {mean} is not possible. If you declare moments of a "
            f"distribution in the GCN file (such as mu, sigma, mean, or sd) along with upper and lower bounds, "
            f"gEcon.py will always produce a distribution that matches these moments given these bounds. As a "
            f"result, you cannot, for example, have a mean of 0 with a lower bound of zero.\n"
            f'To avoid this behavior, pass "loc" and "scale" parameters, rather than "mean" and "std"'
            f' parameters. This will produce a truncated distribution centered at "loc" with dispersion "scale",'
            f"but the moments will NOT be as expected."
        )

        super().__init__(message)


class DistributionOverDefinedException(ValueError):
    def __init__(
        self, variable_name, d_name, dist_n_params, n_params_passed, n_constraints
    ):
        message = (
            f"The {d_name} distribution associated wth {variable_name} is over-defined. The distribution has "
            f"{dist_n_params} free parameters, but you passed {n_params_passed} plus {n_constraints} moment "
            f"conditions."
        )
        super().__init__(message)


class OrphanParameterError(ValueError):
    def __init__(self, orphans):
        orphans = set(orphans)
        n = len(orphans)
        verb = "was" if n == 1 else "were"
        message = (
            f"The following parameter{'s' if n > 1 else ''} {verb} found among model equations but did not appear in "
            f"any calibration block: {', '.join([x.name for x in orphans])}"
        )

        super().__init__(message)


class ExtraParameterError(ValueError):
    def __init__(self, extras):
        n = len(extras)
        verb = "was" if n == 1 else "were"
        message = (
            f"The following parameter{'s' if n > 1 else ''} {verb} were given initial values in calibration blocks but "
            f"were not used in model equations: {', '.join([x.name for x in extras])} \n"
            f"Verify your model equations, or remove these parameters if they are not needed."
        )

        super().__init__(message)


class ExtraParameterWarning(UserWarning):
    def __init__(self, extras):
        n = len(extras)
        verb = "was" if n == 1 else "were"
        message = (
            f"The following parameter{'s' if n > 1 else ''} {verb} were given initial values in calibration blocks but "
            f"were not used in model equations: {', '.join([x.name for x in extras])} \n"
            f"Verify your model equations, or remove these parameters if they are not needed."
        )

        super().__init__(message)


class DuplicateParameterError(ValueError):
    def __init__(self, extras, block=None):
        n = len(extras)
        verb = "was" if n == 1 else "were"
        location = "calibration blocks"
        if block is not None:
            location = f"in {block} calibration block"
        message = (
            f"The following parameter{'s' if n > 1 else ''} {verb} were given initial values in {location} more "
            f"than once: {', '.join([x.name for x in extras])} \n"
            f"Model parameters should be declared only once. Check your GCN file and remove one of the declarations."
        )

        super().__init__(message)
