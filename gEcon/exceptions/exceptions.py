import sympy as sp
from typing import List, Optional
from gEcon.parser.constants import BLOCK_COMPONENTS
from gEcon.classes.TimeAwareSymbol import TimeAwareSymbol


class GCNSyntaxError(ValueError):

    def __init__(self, block_name: str, key: List[str]):
        self.block_name = block_name
        self.key = key

        valid_components = [x.name for x in list(BLOCK_COMPONENTS)]

        message = f'While parsing block {block_name}, found {" ".join(key)} outside of a block component. All ' \
                  f'equations must be inside components with one of valid component names:' \
                  f' {", ".join(valid_components)}'

        super().__init__(message)


class DistributionParsingError(ValueError):

    def __init__(self, line):
        message = f'While parsing parameter prior distributions, encountered an error in the following line:\n' \
                  f'{line}\n' \
                  f'Check that distribution is defined using the "~" operator according to the following syntax: ' \
                  f'"parameter ~ d(a=, b=, ...) = initial_value;"'

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
        self.message = f'Block {self.name} called method _get_param_dict_and_calibrating_equations before ' \
                       f'initialization. Call initialize_from_dictionary first.'

        super().__init__(self.message)


class DynamicCalibratingEquationException(ValueError):

    def __init__(self, eq: sp.Add, block_name: str):
        self.eq = eq
        self.block_name = block_name

        self.message = f'In block {block_name}, calibrating equation {eq} has variables with non-steady-state ' \
                       f'time values. All calibrating equations must be written in terms of steady-state, ' \
                       f'in the form X[ss].'

        super().__init__(self.message)


class OptimizationProblemNotDefinedException(ValueError):

    def __init__(self, block_name: str, missing: str) -> None:
        self.block_name = block_name
        self.missing = missing
        not_missing = 'objective' if missing == 'controls' else 'controls'

        message = f'Block {block_name} has a {missing} component but no {not_missing} component, verify whether' \
                  f'or not this block has an optimization problem.'

        super().__init__(message)


class MultipleObjectiveFunctionsException(ValueError):

    def __init__(self, block_name: str, eqs: List[sp.Eq]) -> None:
        self.block_name = block_name

        n_eqs = len(eqs)

        message = f'Block {block_name} appears to have multiple objectives, excepted just one but found {n_eqs}:\n'
        for eq in eqs:
            message += (str(eq) + '\n')
        message += f' Only one objective function is supported. Please manually simplify the objective' \
                   f' to a single function.'

        super().__init__(message)


class ControlVariableNotFoundException(ValueError):

    def __init__(self, block_name: str, control: TimeAwareSymbol):
        self.block_name = block_name
        self.control = control

        message = f'Block {block_name} has {control} declared as a control variable, ' \
                  f'but this variable was not found among model equations in components definitions,' \
                  f' objective, or constraints.'

        super().__init__(message)


class SteadyStateNotSolvedError(ValueError):

    def __init__(self):
        message = f'The system cannot be solved before the steady-state has been found! Call the .steady_state() method' \
                  f'to solve for the steady state.'

        super().__init__(message)


class GensysFailedException(ValueError):

    def __init__(self, eu):
        message = ''
        if eu[0] == -2 and eu[1] == -2:
            message = "Coincident zeros.  Indeterminacy and/or nonexistence."
        elif eu[0] == -1:
            message = f"System is indeterminate. There are {eu[2]} loose endogenous variables."
        elif eu[1] == -1:
            message = f'Solution exists, but it is not unique -- sunspots.'

        super().__init__(message)


class VariableNotFoundException(ValueError):

    def __init__(self, variable):
        var_name = variable.base_name
        message = f'Variable {var_name} was not found among model variables.'

        super().__init__(message)


class InvalidDistributionException(ValueError):

    def __init__(self, variable, distribution_string):
        message = f'The distribution associated with "{variable}", defined as "{distribution_string}", appears to have ' \
                  f'a typo, please check the GCN file. Please also check that you have not supplied an initial ' \
                  f'parameter value to an exogenous shock distribution, as in epsilon[] ~ N(mu=0, sd=1) = 0.5. Shock ' \
                  f'distributions should NOT have an equals sign after the distribution definition.'

        super().__init__(message)


class DistributionNotFoundException(ValueError):

    def __init__(self, d_str: str, best_guess: str, best_guess_canon: str):
        message = f'Distribution "{d_str}" not recoginized, did you mean {best_guess}, implying the ' \
                  f' {best_guess_canon} distribution?'

        super().__init__(message)


class RepeatedParameterException(ValueError):

    def __init__(self, variable_name: str, d_str: str, parameter: str):
        message = f'In the {d_str} distribution associated with "{variable_name}", the parameter "{parameter}"was ' \
                  f'declared multiple times. Please check the GCN file for typos.'

        super().__init__(message)


class ParameterNotFoundException(ValueError):
    def __init__(self, variable_name: str, d_name: str, param_name: str, valid_param_names: List[str],
                 maybe_typo: Optional[str], best_guess: Optional[str]):
        message = f'No {param_name} parameter was found for the {d_name} distribution associated with model parameter ' \
                  f'"{variable_name}". Valid aliases for {param_name} are: '
        if len(valid_param_names) > 1:
            message += ', '.join(valid_param_names[:-1]) + f', and {valid_param_names[-1]}.'
        else:
            message += f'{valid_param_names[0]}.'

        if maybe_typo is not None and best_guess is not None:
            message += f'\n\nFound a similar alias to suspected typo: {maybe_typo}. Did you mean {best_guess}?'

        super().__init__(message)


class MultipleParameterDefinitionException(ValueError):
    def __init__(self, variable_name: str, d_name: str, param_name: str, result_list: List[str]) -> None:
        message = f'The {d_name} distribution associated with "{variable_name}" has multiple declarations for ' \
                  f'{param_name}. Please pass only one of: '
        message += ', '.join(result_list)

        super().__init__(message)


class UnusedParameterError(ValueError):
    def __init__(self, d_name: str, param_name: str) -> None:
        message = f'{d_name} distributions do not have a {param_name}; do not call this parse method.'

        super().__init__(message)


class IgnoredCloseMatchWarning(UserWarning):
    pass


class UnusedParameterWarning(UserWarning):
    pass
