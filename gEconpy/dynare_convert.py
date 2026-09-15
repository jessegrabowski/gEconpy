from pathlib import Path
from typing import TYPE_CHECKING

import sympy as sp

from sympy.core import Mul, Pow, Rational, S
from sympy.core.mul import _keep_coeff
from sympy.core.numbers import equal_valued
from sympy.printing.octave import OctaveCodePrinter, precedence

from gEconpy.classes.time_aware_symbol import TimeAwareSymbol

if TYPE_CHECKING:
    from gEconpy.model.model import Model


def make_mod_file(
    model: "Model", linewidth: int = 100, use_cse: bool = True, out_path: str | Path | None = None
) -> str | None:
    """
    Write a gEconpy model as a Dynare ``.mod`` file.

    The file declares the variables, shocks, and parameters, writes the model equations, and adds a steady state
    block, a ``check`` command, a standard deviation of 0.01 for every shock, and a first-order ``stoch_simul``
    command. When the model has a complete analytic steady state the steady state block is a ``steady_state_model``
    block. Otherwise the numeric steady state is solved and written as an ``initval`` block. See [1]_ for the Dynare
    syntax.

    Parameters
    ----------
    model : Model
        Model to export.
    linewidth : int, optional
        Maximum number of characters per line in the declaration blocks. Defaults to 100.
    use_cse : bool, optional
        If True, rewrite the analytic steady state in terms of common sub-expressions found by :func:`sympy.cse`.
        Defaults to True.
    out_path : str or Path, optional
        Path to write the generated file to. By default the file is returned as a string and nothing is written.

    Returns
    -------
    mod_file : str or None
        The contents of the Dynare model file, or None when ``out_path`` is given.

    References
    ----------
    .. [1] Adjemian, S., Bastani, H., Juillard, M., Mihoubi, F., Perendia, G., Ratto, M.,
       and Villemot, S. "Dynare: Reference Manual, Version 4." *CEPREMAP* (2011).

    Examples
    --------
    Export the RBC example model and inspect the generated Dynare code:

    .. code-block:: python

        from gEconpy import make_mod_file, model_from_gcn
        from gEconpy.data import get_example_gcn

        model = model_from_gcn(get_example_gcn("RBC"), verbose=False)
        mod_file = make_mod_file(model)
        print(mod_file)

    With ``out_path`` the file is written to disk and nothing is returned:

    .. code-block:: python

        from gEconpy import make_mod_file, model_from_gcn
        from gEconpy.data import get_example_gcn

        model = model_from_gcn(get_example_gcn("RBC"), verbose=False)
        make_mod_file(model, out_path="rbc.mod")
    """
    mod_blocks = [
        write_variable_declarations(model, linewidth=linewidth),
        write_shock_declarations(model, linewidth=linewidth),
        write_parameter_declarations(model, linewidth=linewidth),
        write_model_equations(model),
        write_steady_state(model, use_cse=use_cse),
        "check(qz_zero_threshold=1e-20);",
        write_shock_std(model),
        "stoch_simul(order=1, irf=100, qz_zero_threshold=1e-20);",
    ]

    mod_file = "\n\n".join(mod_blocks)

    if out_path is None:
        return mod_file

    Path(out_path).write_text(mod_file)
    return None


class DynareCodePrinter(OctaveCodePrinter):
    """Print sympy expressions as Dynare model code."""

    def _print_Mul(self, expr):
        if expr.is_number and expr.is_imaginary and (S.ImaginaryUnit * expr).is_Integer:
            return f"{self._print(-S.ImaginaryUnit * expr)}i"

        prec = precedence(expr)

        coeff, rest = expr.as_coeff_Mul()
        if coeff < 0:
            expr = _keep_coeff(-coeff, rest)
            sign = "-"
        else:
            sign = ""

        numerator = []
        denominator = []
        parenthesized_pows = []

        args = expr.as_ordered_factors() if self.order not in ("old", "none") else Mul.make_args(expr)

        for item in args:
            if item.is_commutative and item.is_Pow and item.exp.is_Rational and item.exp.is_negative:
                if item.exp != -1:
                    denominator.append(Pow(item.base, -item.exp, evaluate=False))
                else:
                    # A Mul base with exponent -1 needs its own parentheses in the denominator.
                    if len(item.args[0].args) != 1 and isinstance(item.base, Mul):
                        parenthesized_pows.append(item)
                    denominator.append(Pow(item.base, -item.exp))
            elif item.is_Rational and item is not S.Infinity:
                if item.p != 1:
                    numerator.append(Rational(item.p))
                if item.q != 1:
                    denominator.append(Rational(item.q))
            else:
                numerator.append(item)

        numerator = numerator or [S.One]

        numerator_str = [self.parenthesize(x, prec) for x in numerator]
        denominator_str = [self.parenthesize(x, prec) for x in denominator]

        for item in parenthesized_pows:
            if item.base in denominator:
                position = denominator.index(item.base)
                denominator_str[position] = f"({denominator_str[position]})"

        mul_symbol = " .* " if expr.is_Matrix else " * "
        div_symbol = " ./ " if expr.is_Matrix else " / "

        numerator_joined = mul_symbol.join(numerator_str)
        if not denominator:
            return sign + numerator_joined
        if len(denominator) == 1:
            return sign + numerator_joined + div_symbol + denominator_str[0]
        return sign + numerator_joined + div_symbol + f"({mul_symbol.join(denominator_str)})"

    def _print_Pow(self, expr):
        prec = precedence(expr)

        if equal_valued(expr.exp, 0.5):
            return f"sqrt({self._print(expr.base)})"

        if expr.is_commutative:
            div_symbol = " ./ " if expr.is_Matrix else " / "
            if equal_valued(expr.exp, -0.5):
                return "1" + div_symbol + f"sqrt({self._print(expr.base)})"
            if equal_valued(expr.exp, -1):
                return "1" + div_symbol + f"{self.parenthesize(expr.base, prec)}"

        return f"{self.parenthesize(expr.base, prec)} ^ {self.parenthesize(expr.exp, prec)}"

    def _print_TimeAwareSymbol(self, expr):
        name = expr.base_name
        time_index = expr.time_index

        if time_index == "ss":
            return f"{name}_ss"
        if time_index == 0:
            return name
        if time_index > 0:
            return f"{name}(+{time_index})"

        return f"{name}({time_index})"


def write_lines_from_list(items_to_write: list[str], linewidth: int = 100, line_start: str = "") -> str:
    """
    Join items into comma-separated declaration lines, wrapping when a line grows too long.

    Parameters
    ----------
    items_to_write : list of str
        Names to write.
    linewidth : int, optional
        Maximum number of characters per line, including the terminating semicolon. Defaults to 100.
    line_start : str, optional
        Keyword to place at the start of every line, for example ``var``. Defaults to an empty string.

    Returns
    -------
    lines : str
        The declaration lines, each terminated with a semicolon.
    """
    lines = []
    line = line_start

    for item in items_to_write:
        addition = f", {item}" if line != line_start else f" {item}"

        if line != line_start and (len(line) + len(addition) + len(";")) > linewidth:
            lines.append(line + ";")
            line = f"{line_start} {item}"
        else:
            line += addition

    lines.append(line + ";")
    return "\n".join(lines)


def write_variable_declarations(mod: "Model", linewidth: int = 100) -> str:
    """Write the Dynare ``var`` block declaring every model variable."""
    var_names = [var.base_name for var in mod.variables]
    return write_lines_from_list(var_names, linewidth=linewidth, line_start="var")


def write_shock_declarations(mod: "Model", linewidth: int = 100) -> str:
    """Write the Dynare ``varexo`` block declaring every model shock."""
    shock_names = [shock.base_name for shock in mod.shocks]
    return write_lines_from_list(shock_names, linewidth=linewidth, line_start="varexo")


def write_values_from_dict(d: dict[str, float], round: int = 3) -> str:
    """
    Write one ``name = value;`` assignment per dictionary entry.

    Parameters
    ----------
    d : dict mapping str to float
        Names and values to assign.
    round : int, optional
        Number of decimal places to print. Defaults to 3.

    Returns
    -------
    assignments : str
        One assignment per line.
    """
    return "".join(f"{name} = {value:0.{round}f};\n" for name, value in d.items())


def write_param_names(mod: "Model", linewidth: int = 100) -> str:
    """Write the Dynare ``parameters`` block declaring every model parameter name."""
    param_names = [param.name for param in mod.params]
    return write_lines_from_list(param_names, linewidth=linewidth, line_start="parameters")


def write_parameter_declarations(mod: "Model", linewidth: int = 100) -> str:
    """Write the Dynare ``parameters`` block followed by an assignment for each parameter value."""
    param_names = write_param_names(mod, linewidth=linewidth)
    param_values = write_values_from_dict(mod.parameters().to_string())

    return f"{param_names}\n\n{param_values}"


def find_ss_variables(mod: "Model") -> list[TimeAwareSymbol]:
    """Return the steady-state variables used by the model equations, sorted by base name."""
    variables = set().union(*(eq.free_symbols for eq in mod.equations))

    return sorted(
        (x for x in variables if isinstance(x, TimeAwareSymbol) and x.time_index == "ss"),
        key=lambda x: x.base_name,
    )


def write_model_equations(mod: "Model") -> str:
    """Write the Dynare ``model`` block, with a local definition for each steady-state value the equations use."""
    printer = DynareCodePrinter()

    required_ss_values = set(find_ss_variables(mod))
    defined_ss_values = {eq.lhs for eq in mod.steady_state_relationships}

    if required_ss_values <= defined_ss_values:
        ss_dict = {
            eq.lhs.name: printer.doprint(eq.rhs)
            for eq in mod.steady_state_relationships
            if eq.lhs in required_ss_values
        }
    else:
        ss_values = mod.steady_state(verbose=False, progressbar=False).to_sympy()
        ss_dict = {k.name: v for k, v in ss_values.items() if k in required_ss_values}

    local_definitions = "".join(f"#{k} = {v};\n" for k, v in ss_dict.items())
    equations = "\n".join([printer.doprint(eq) + ";" for eq in mod.equations])

    return f"model;\n\n{local_definitions}{equations}\n\nend;"


def write_steady_state(mod: "Model", use_cse: bool = True) -> str:
    """
    Write a ``steady_state_model`` block if an analytic steady state exists, otherwise an ``initval`` block.

    Parameters
    ----------
    mod : Model
        Model to export.
    use_cse : bool, optional
        If True, rewrite the analytic steady state in terms of common sub-expressions found by :func:`sympy.cse`.
        Defaults to True.

    Returns
    -------
    block : str
        The steady state block, followed by Dynare's ``steady`` and ``resid`` commands.
    """
    has_analytic_steady_state = len(mod.steady_state_relationships) == len(mod.variables)

    if has_analytic_steady_state:
        block = _write_steady_state_model_block(mod, use_cse=use_cse)
    else:
        block = _write_initval_block(mod)

    return block + "\n\nsteady;\nresid;"


def write_shock_std(mod: "Model") -> str:
    """Write the Dynare ``shocks`` block, giving every shock a standard deviation of 0.01."""
    shock_lines = "".join(f"var {shock.base_name};\nstderr 0.01;\n\n" for shock in mod.shocks)
    return f"shocks;\n{shock_lines}end;"


def _write_steady_state_model_block(mod: "Model", use_cse: bool) -> str:
    printer = DynareCodePrinter()
    out = "steady_state_model;\n"

    eqs = mod.steady_state_relationships
    if use_cse:
        replacements, eqs = sp.cse(eqs)
        for var, expr in replacements:
            out += f"{var} = {printer.doprint(expr)};\n"
        out += "\n\n"

    for eq in eqs:
        out += f"{eq.lhs.base_name} = {printer.doprint(eq.rhs)};\n"

    return out + "\n\nend;"


def _write_initval_block(mod: "Model") -> str:
    steady_state = mod.steady_state(verbose=False, progressbar=False)
    ss_dict = {k.base_name: v for k, v in steady_state.to_sympy().items()}

    return f"initval;\n{write_values_from_dict(ss_dict)}\nend;"
