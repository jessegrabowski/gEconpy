from hashlib import sha256

import numpy as np

from pytensor.link.numba.dispatch import basic as numba_basic
from pytensor.link.numba.dispatch.basic import (
    generate_fallback_impl,
    numba_funcify_and_cache_key,
    register_funcify_and_cache_key,
)
from pytensor.tensor.optimize import RootScalarOp

# Each solver is built per Op as a closure over that Op's compiled objective. numba caches a closure over a compiled
# function, but not a function that receives one as an argument, nor any function that calls it.


def make_value_at(objective):
    @numba_basic.numba_njit(fastmath=False)
    def value_at(x, args):
        return objective(np.asarray(x), *args)[0].item()

    return value_at


def make_newton(objective):
    @numba_basic.numba_njit(fastmath=False)
    def newton(x0, args, xtol, maxiter):
        x = x0
        for _ in range(maxiter):
            value, slope = objective(np.asarray(x), *args)
            value, slope = value.item(), slope.item()
            if value == 0.0:
                return x, True
            if slope == 0.0:
                return x, False
            step = value / slope
            x -= step
            if abs(step) <= xtol:
                return x, True
        return x, False

    return newton


def make_halley(objective):
    @numba_basic.numba_njit(fastmath=False)
    def halley(x0, args, xtol, maxiter):
        x = x0
        for _ in range(maxiter):
            value, slope, curvature = objective(np.asarray(x), *args)
            value, slope, curvature = value.item(), slope.item(), curvature.item()
            if value == 0.0:
                return x, True
            if slope == 0.0:
                return x, False
            step = value / slope
            correction = step * curvature / (2 * slope)
            if abs(correction) < 1:
                step /= 1 - correction
            x -= step
            if abs(step) <= xtol:
                return x, True
        return x, False

    return halley


def make_secant(objective):
    value_at = make_value_at(objective)

    @numba_basic.numba_njit(fastmath=False)
    def secant(x0, args, xtol, maxiter):
        previous = x0
        current = x0 * (1 + 1e-4) + (1e-4 if x0 >= 0 else -1e-4)
        previous_value = value_at(previous, args)
        current_value = value_at(current, args)
        for _ in range(maxiter):
            if current_value == previous_value:
                return current, current_value == 0.0
            following = current - current_value * (current - previous) / (current_value - previous_value)
            if abs(following - current) <= xtol:
                return following, True
            previous, previous_value = current, current_value
            current, current_value = following, value_at(following, args)
        return current, False

    return secant


def make_bisect(objective):
    value_at = make_value_at(objective)

    @numba_basic.numba_njit(fastmath=False)
    def bisect(lower, upper, args, xtol, rtol, maxiter):
        lower_value = value_at(lower, args)
        upper_value = value_at(upper, args)
        if not (np.isfinite(lower_value) and np.isfinite(upper_value)) or lower_value * upper_value > 0:
            return np.nan, False
        if lower_value == 0.0:
            return lower, True
        if upper_value == 0.0:
            return upper, True
        for _ in range(maxiter):
            middle = (lower + upper) / 2
            middle_value = value_at(middle, args)
            if middle_value == 0.0 or (upper - lower) / 2 < xtol + rtol * abs(middle):
                return middle, True
            if np.sign(middle_value) == np.sign(lower_value):
                lower, lower_value = middle, middle_value
            else:
                upper = middle
        return (lower + upper) / 2, False

    return bisect


@numba_basic.numba_njit(fastmath=False)
def brent_step(points, values, previous_step, current_step, tolerance, bisection):
    """Return the next pair of step sizes: an interpolation step when it is acceptable, a bisection step otherwise."""
    previous, current, block = points
    previous_value, current_value, block_value = values
    if abs(previous_step) <= tolerance or abs(current_value) >= abs(previous_value):
        return bisection, bisection

    if previous == block:
        trial = -current_value * (current - previous) / (current_value - previous_value)
    else:
        previous_slope = (previous_value - current_value) / (previous - current)
        block_slope = (block_value - current_value) / (block - current)
        trial = (
            -current_value
            * (block_value * block_slope - previous_value * previous_slope)
            / (block_slope * previous_slope * (block_value - previous_value))
        )
    if 2 * abs(trial) < min(abs(previous_step), 3 * abs(bisection) - tolerance):
        return current_step, trial
    return bisection, bisection


def make_brentq(objective):
    value_at = make_value_at(objective)

    @numba_basic.numba_njit(fastmath=False)
    def brentq(lower, upper, args, xtol, rtol, maxiter):
        # Brent's method as in scipy's brentq.c: inverse quadratic interpolation or a secant step when it stays
        # inside the bracket and shrinks fast enough, bisection otherwise.
        previous, current = lower, upper
        previous_value, current_value = value_at(previous, args), value_at(current, args)
        if not (np.isfinite(previous_value) and np.isfinite(current_value)) or previous_value * current_value > 0:
            return np.nan, False
        if previous_value == 0.0:
            return previous, True
        if current_value == 0.0:
            return current, True

        block, block_value = previous, previous_value
        previous_step = current_step = 0.0
        for _ in range(maxiter):
            if previous_value != 0.0 and current_value != 0.0 and np.sign(previous_value) != np.sign(current_value):
                block, block_value = previous, previous_value
                previous_step = current_step = current - previous
            if abs(block_value) < abs(current_value):
                previous, current, block = current, block, current
                previous_value, current_value, block_value = current_value, block_value, current_value

            tolerance = (xtol + rtol * abs(current)) / 2
            bisection = (block - current) / 2
            if current_value == 0.0 or abs(bisection) < tolerance:
                return current, True

            previous_step, current_step = brent_step(
                (previous, current, block),
                (previous_value, current_value, block_value),
                previous_step,
                current_step,
                tolerance,
                bisection,
            )

            previous, previous_value = current, current_value
            current += current_step if abs(current_step) > tolerance else (tolerance if bisection > 0 else -tolerance)
            current_value = value_at(current, args)
        return current, False

    return brentq


BRACKET_METHODS = {"brentq": make_brentq, "bisect": make_bisect}
OPEN_METHODS = {"newton": make_newton, "halley": make_halley, "secant": make_secant}


def solver_options(method: str, optimizer_kwargs: dict) -> dict | None:
    """Return the options ``method`` runs with, scipy's defaults filled in, or None when it is not implemented."""
    options = dict(optimizer_kwargs)
    if method in BRACKET_METHODS and "bracket" in options:
        lower, upper = (float(end) for end in options.pop("bracket"))
        settings = {
            "lower": lower,
            "upper": upper,
            "xtol": float(options.pop("xtol", 2e-12)),
            "rtol": float(options.pop("rtol", 4 * np.finfo(np.float64).eps)),
            "maxiter": int(options.pop("maxiter", 100)),
        }
    elif method in OPEN_METHODS:
        settings = {"xtol": float(options.pop("xtol", 1.48e-8)), "maxiter": int(options.pop("maxiter", 50))}
    else:
        return None
    return None if options else settings


@register_funcify_and_cache_key(RootScalarOp)
def numba_funcify_RootScalarOp(op, node, **kwargs):
    settings = solver_options(op.method, op.optimizer_kwargs)
    if settings is None:
        return generate_fallback_impl(op, node, **kwargs), None

    kwargs.pop("storage_map", None)
    objective, objective_key = numba_funcify_and_cache_key(
        op.fgraph, fgraph_name="numba_root_scalar_objective", **kwargs
    )

    as_output_dtype = np.dtype(node.outputs[0].type.dtype).type
    if op.method in BRACKET_METHODS:
        solve = BRACKET_METHODS[op.method](objective)
        lower, upper, xtol, rtol, maxiter = (settings[key] for key in ("lower", "upper", "xtol", "rtol", "maxiter"))

        @numba_basic.numba_njit(fastmath=False)
        def root_scalar(_x0, *args):
            root, converged = solve(lower, upper, args, xtol, rtol, maxiter)
            return np.asarray(as_output_dtype(root)), converged

    else:
        solve = OPEN_METHODS[op.method](objective)
        xtol, maxiter = settings["xtol"], settings["maxiter"]

        @numba_basic.numba_njit(fastmath=False)
        def root_scalar(x0, *args):
            root, converged = solve(x0.item(), args, xtol, maxiter)
            return np.asarray(as_output_dtype(root)), converged

    cache_key = None
    if objective_key is not None:
        cache_version = 2
        cache_key = sha256(
            str((type(op), cache_version, op.method, sorted(settings.items()), objective_key)).encode()
        ).hexdigest()
    return root_scalar, cache_key
