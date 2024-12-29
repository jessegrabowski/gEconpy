from functools import reduce
from typing import TYPE_CHECKING

import sympy as sp

from sympy.core import Mul, Pow, Rational, S
from sympy.core.mul import _keep_coeff
from sympy.core.numbers import equal_valued
from sympy.printing.octave import OctaveCodePrinter, precedence

from gEconpy.classes.time_aware_symbol import TimeAwareSymbol

if TYPE_CHECKING:
    from gEconpy.model.model import Model

OPERATORS = list("+-/*^()=")


class DynareCodePrinter(OctaveCodePrinter):
    def __init__(self, settings=None):
        settings = {} if settings is None else settings
        super().__init__(settings)

    def _print_Mul(self, expr):
        # print complex numbers nicely in Octave
        if expr.is_number and expr.is_imaginary and (S.ImaginaryUnit * expr).is_Integer:
            return f"{self._print(-S.ImaginaryUnit * expr)}i"

        # cribbed from str.py
        prec = precedence(expr)

        c, e = expr.as_coeff_Mul()
        if c < 0:
            expr = _keep_coeff(-c, e)
            sign = "-"
        else:
            sign = ""

        a = []  # items in the numerator
        b = []  # items that are in the denominator (if any)

        pow_paren = []  # Will collect all pow with more than one base element and exp = -1

        if self.order not in ("old", "none"):
            args = expr.as_ordered_factors()
        else:
            # use make_args in case expr was something like -x -> x
            args = Mul.make_args(expr)

        # Gather args for numerator/denominator
        for item in args:
            if (
                item.is_commutative
                and item.is_Pow
                and item.exp.is_Rational
                and item.exp.is_negative
            ):
                if item.exp != -1:
                    b.append(Pow(item.base, -item.exp, evaluate=False))
                else:
                    if len(item.args[0].args) != 1 and isinstance(
                        item.base, Mul
                    ):  # To avoid situations like #14160
                        pow_paren.append(item)
                    b.append(Pow(item.base, -item.exp))
            elif item.is_Rational and item is not S.Infinity:
                if item.p != 1:
                    a.append(Rational(item.p))
                if item.q != 1:
                    b.append(Rational(item.q))
            else:
                a.append(item)

        a = a or [S.One]

        a_str = [self.parenthesize(x, prec) for x in a]
        b_str = [self.parenthesize(x, prec) for x in b]

        # To parenthesize Pow with exp = -1 and having more than one Symbol
        for item in pow_paren:
            if item.base in b:
                b_str[b.index(item.base)] = f"({b_str[b.index(item.base)]})"

        def multjoin(a, a_str):
            # here we probably are assuming the constants will come first
            r = a_str[0]
            for i in range(1, len(a)):
                mulsym = " * " if not expr.is_Matrix else " .* "
                r = r + mulsym + a_str[i]
            return r

        if not b:
            return sign + multjoin(a, a_str)
        elif len(b) == 1:
            divsym = " / " if not expr.is_Matrix else " ./ "
            return sign + multjoin(a, a_str) + divsym + b_str[0]
        else:
            divsym = " / " if not expr.is_Matrix else " ./ "
            return sign + multjoin(a, a_str) + divsym + f"({multjoin(b, b_str)})"

    def _print_Pow(self, expr):
        powsymbol = " ^ "

        PREC = precedence(expr)

        if equal_valued(expr.exp, 0.5):
            return f"sqrt({self._print(expr.base)})"

        if expr.is_commutative:
            if equal_valued(expr.exp, -0.5):
                sym = " / " if not expr.is_Matrix else " ./ "
                return "1" + sym + f"sqrt({self._print(expr.base)})"
            if equal_valued(expr.exp, -1):
                sym = " / " if not expr.is_Matrix else " ./ "
                return "1" + sym + f"{self.parenthesize(expr.base, PREC)}"

        return f"{self.parenthesize(expr.base, PREC)}{powsymbol}{self.parenthesize(expr.exp, PREC)}"

    def _print_TimeAwareSymbol(self, expr):
        name = expr.base_name
        t = expr.time_index

        if t == "ss":
            return f"{name}_{t}"
        elif t == 0:
            return f"{name}"
        elif t > 0:
            return f"{name}(+{t})"

        return f"{name}({t})"


def write_lines_from_list(items_to_write, linewidth=100, line_start=""):
    lines = []
    line = line_start

    for item in items_to_write:
        addition = f", {item}" if line != line_start else f" {item}"

        # Add 1 to account for the final semicolon
        if (len(line) + len(addition) + 1) > linewidth:
            lines.append(line + ";")  # Add semicolon to complete the line
            line = f"{line_start} {item}"
        else:
            line += addition

    lines.append(line + ";")  # Add the final line with a semicolon
    return "\n".join(lines)


def write_variable_declarations(mod: "Model", linewidth=100):
    var_names = [var.base_name for var in mod.variables]
    return write_lines_from_list(var_names, linewidth=linewidth, line_start="var")


def write_shock_declarations(mod: "Model", linewidth=100):
    shock_names = [shock.base_name for shock in mod.shocks]
    return write_lines_from_list(shock_names, linewidth=linewidth, line_start="varexo")


def write_values_from_dict(d, round: int = 3):
    out = ""
    for name, value in d.items():
        out += f"{name} = {value:0.{round}f};\n"
    return out


def write_param_names(mod: "Model", linewidth=100):
    param_names = [param.name for param in mod.params]
    param_string = write_lines_from_list(
        param_names, linewidth=linewidth, line_start="parameters"
    )

    return param_string


def write_parameter_declarations(mod: "Model", linewidth=100):
    param_string = write_param_names(mod, linewidth=linewidth)
    param_string += "\n\n"
    param_string += write_values_from_dict(mod.parameters().to_string())

    return param_string


def find_ss_variables(mod: "Model"):
    variables = reduce(
        lambda s, eq: s.union(set(eq.free_symbols)), mod.equations, set()
    )

    return sorted(
        [
            x
            for x in variables
            if isinstance(x, TimeAwareSymbol) and (x.time_index == "ss")
        ],
        key=lambda x: x.base_name,
    )


def write_model_equations(mod: "Model"):
    printer = DynareCodePrinter()

    required_ss_values = find_ss_variables(mod)
    defined_ss_values = [x.lhs for x in mod.steady_state_relationships]

    if not all(ss_var in defined_ss_values for ss_var in required_ss_values):
        ss_values = mod.steady_state(verbose=False, progressbar=False).to_sympy()
        ss_dict = {k.name: v for k, v in ss_values.items() if k in required_ss_values}
    else:
        ss_dict = {
            eq.lhs: eq.rhs
            for eq in mod.steady_state_relationships
            if eq.lhs in required_ss_values
        }
        ss_dict = {k.name: printer.doprint(v) for k, v in ss_dict.items()}

    model_block = "model;\n\n"
    for k, v in ss_dict.items():
        model_block += f"#{k} = {v};\n"

    model_block += "\n".join([printer.doprint(eq) + ";" for eq in mod.equations])
    model_block += "\n\nend;"

    return model_block


def write_steady_state(mod: "Model", use_cse=True):
    printer = DynareCodePrinter()

    # Check for a full analytic steady state. If available, we can write a
    # steady_state_model block
    if len(mod.steady_state_relationships) == len(mod.variables):
        out = "steady_state_model;\n"
        eqs = mod.steady_state_relationships
        if use_cse:
            cse, eqs = sp.cse(eqs)
            for var, expr in cse:
                out += f"{var} = {printer.doprint(expr)};\n"
            out += "\n\n"
        for eq in eqs:
            out += f"{eq.lhs.base_name} = {printer.doprint(eq.rhs)};\n"

        out += "\n\nend;"

    # Otherwise solve for a numeric steady state and use that as initial values to Dynare
    else:
        out = "initval;\n"
        steady_state = mod.steady_state(verbose=False, progressbar=False)
        ss_dict = {k.base_name: v for k, v in steady_state.to_sympy().items()}
        out += write_values_from_dict(ss_dict)
        out += "\nend;"

    out += "\n\nsteady;\nresid;"
    return out


def write_shock_std(mod: "Model"):
    out = "shocks;\n"
    shock_names = [shock.base_name for shock in mod.shocks]

    for shock in shock_names:
        out += f"var {shock};\nstderr 0.01;\n\n"

    out += "end;"
    return out


def make_mod_file(
    model: "Model", linewidth=100, use_cse: bool = True, out_path=None
) -> str | None:
    """
    Generate a string representation of a Dynare model file for a dynamic stochastic general equilibrium (DSGE) model.
    For more information, see [1].

    Parameters
    ----------
    model : Model
        A DSGE model object
    linewidth: int, default 100
        Maximum number of characters per line before a break is insterted
    use_cse: bool, default True
        If True, use ``sp.cse`` to identify common sub expressions in the analytic steady state and rewrite equations
        in terms of these sub expressions. This can make the steady state block more readable and provide modest
        performance increase for large models.
    out_path: str, optional
        If None, the generated mod file is printed to the terminal. Otherwise, it is written to ``out_path``.

    Returns
    -------
    str
        A string representation of a Dynare model file.

    References
    ----------
    ..[1] Adjemian, Stéphane, et al. "Dynare: Reference manual, version 4." (2011).
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

    with open(out_path, "w") as f:
        f.write(mod_file)
