import os

from collections.abc import Callable, Iterator, Mapping, MutableMapping
from contextlib import contextmanager
from difflib import get_close_matches
from pathlib import Path
from typing import Any

RC_FILENAME = "geconpyrc"

_STATESPACE_SOLVER_OPTIONS = ("gensys", "cycle_reduction", "scan_cycle_reduction", "backward_direct")
_ADJOINT_METHOD_OPTIONS = ("doubling", "qz", "sylvester", "direct")


def _non_negative_float(value: Any) -> float:
    number = float(value)
    if not number >= 0:
        raise ValueError(f"expected a float >= 0, got {value!r}")
    return number


def _positive_float(value: Any) -> float:
    number = float(value)
    if not number > 0:
        raise ValueError(f"expected a float > 0, got {value!r}")
    return number


def _positive_int(value: Any) -> int:
    number = float(value)
    if isinstance(value, bool) or not number.is_integer() or number <= 0:
        raise ValueError(f"expected an integer > 0, got {value!r}")
    return int(number)


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in ("true", "false"):
        return value.lower() == "true"
    raise ValueError(f"expected True or False, got {value!r}")


def _optional_str(value: Any) -> str | None:
    if value is None or value == "None":
        return None
    if isinstance(value, str):
        return value
    raise ValueError(f"expected a string or None, got {value!r}")


def _one_of(options: tuple[str, ...]) -> Callable[[Any], str]:
    def validate(value: Any) -> str:
        if value not in options:
            raise ValueError(f"expected one of {', '.join(map(repr, options))}, got {value!r}")
        return value

    return validate


# Each key maps to its default and the validator every new value passes through.
_DEFAULTS: dict[str, tuple[Any, Callable[[Any], Any]]] = {
    "build.verbose": (True, _bool),
    "build.mode": (None, _optional_str),
    "build.on_unused_parameters": ("raise", _one_of(("raise", "warn", "ignore"))),
    "build.show_errors": (True, _bool),
    # An equation holds at the steady state when its residual is at most tol + rtol times the sum of its terms'
    # magnitudes. rtol holds an equation with large terms to the accuracy float64 can deliver at that size.
    "steady_state.tol": (1e-8, _non_negative_float),
    "steady_state.rtol": (1e-10, _non_negative_float),
    # The gradient is the Jacobian times the residual, so it inherits the Jacobian's scale. Holding it to the
    # residual tolerance rejects correct answers for a model whose equations are steep at the steady state.
    "steady_state.grad_tol": (1e-6, _positive_float),
    "sparse_root.tol": (1e-10, _positive_float),
    "sparse_root.maxiter": (1000, _positive_int),
    "perturbation.tol": (1e-8, _positive_float),
    "statespace.solver": ("gensys", _one_of(_STATESPACE_SOLVER_OPTIONS)),
    "statespace.max_iter": (50, _positive_int),
    "statespace.tol": (1e-6, _positive_float),
    "statespace.use_adjoint_gradients": (True, _bool),
    "statespace.adjoint_method": ("doubling", _one_of(_ADJOINT_METHOD_OPTIONS)),
    "statespace.use_direct_lyapunov": (False, _bool),
    "statespace.solver_tol": (1e-8, _positive_float),
}


class RcParams(MutableMapping[str, Any]):
    """
    Global defaults for gEconpy's settings, validated on assignment.

    Keys are fixed. Assigning to an unknown key, or a value its validator rejects, raises. A function argument
    given explicitly always takes precedence over the value here.
    """

    def __init__(self, values: Mapping[str, Any] | None = None):
        self._values = _default_values()
        self.update(values or {})

    def __getitem__(self, key: str) -> Any:
        _check_key(key)
        return self._values[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._values[key] = _validate(key, value)

    def __delitem__(self, key: str) -> None:
        raise TypeError("rcParams keys cannot be deleted. Assign the default instead.")

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def __repr__(self) -> str:
        lines = [f"    {key!r}: {value!r}," for key, value in self._values.items()]
        return "RcParams({\n" + "\n".join(lines) + "\n})"

    def reset(self) -> None:
        """Restore every key to its default."""
        self._values = _default_values()


def _default_values() -> dict[str, Any]:
    return {key: default for key, (default, _) in _DEFAULTS.items()}


def _validate(key: str, value: Any) -> Any:
    _check_key(key)
    _, validator = _DEFAULTS[key]
    try:
        return validator(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid value for rcParams[{key!r}]: {error}") from None


def _check_key(key: str) -> None:
    if key in _DEFAULTS:
        return
    suggestions = get_close_matches(key, _DEFAULTS, n=3)
    hint = f" Did you mean {', '.join(map(repr, suggestions))}?" if suggestions else ""
    raise KeyError(f"{key!r} is not an rcParams key.{hint} Valid keys: {', '.join(_DEFAULTS)}.")


def read_rc_file(fname: str | os.PathLike[str]) -> dict[str, Any]:
    """
    Read and validate the settings in a ``geconpyrc`` file.

    Each non-blank line holds ``key : value``. A ``#`` starts a comment that runs to the end of the line.

    Parameters
    ----------
    fname : str or path-like
        Path to the file.

    Returns
    -------
    settings : dict
        The validated value of every key the file sets.
    """
    settings = {}
    for line_number, raw_line in enumerate(Path(fname).read_text().splitlines(), start=1):
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue

        key, separator, value = (part.strip() for part in line.partition(":"))
        if not separator:
            raise ValueError(f"{fname}, line {line_number}: expected a line of the form 'key : value'")
        try:
            settings[key] = _validate(key, value)
        except (KeyError, ValueError) as error:
            raise ValueError(f"{fname}, line {line_number}: {error.args[0]}") from None
    return settings


def find_rc_file() -> Path | None:
    """
    Return the ``geconpyrc`` file gEconpy reads at import, or None if there is none.

    The working directory is searched first, then ``$XDG_CONFIG_HOME/geconpy``, where ``XDG_CONFIG_HOME`` defaults to
    ``~/.config``. The first file found is used.
    """
    config_home = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    for candidate in (Path.cwd() / RC_FILENAME, config_home / "geconpy" / RC_FILENAME):
        if candidate.is_file():
            return candidate
    return None


@contextmanager
def rc_context(rc: Mapping[str, Any] | None = None, fname: str | os.PathLike[str] | None = None) -> Iterator[RcParams]:
    """
    Change ``rcParams`` temporarily, restoring every key on exit.

    Parameters
    ----------
    rc : dict, optional
        Settings to apply, applied after those in ``fname``.
    fname : str or path-like, optional
        A ``geconpyrc`` file whose settings to apply.

    Yields
    ------
    rcParams : RcParams
        The global settings object.

    Examples
    --------
    .. code-block:: python

        import gEconpy as ge

        with ge.rc_context({"steady_state.rtol": 1e-8}):
            steady_state = model.steady_state()
    """
    saved = dict(rcParams)
    try:
        if fname is not None:
            rcParams.update(read_rc_file(fname))
        rcParams.update(rc or {})
        yield rcParams
    finally:
        rcParams._values = saved


def _load_rc_params() -> RcParams:
    rc_file = find_rc_file()
    return RcParams(read_rc_file(rc_file) if rc_file is not None else None)


rcParams = _load_rc_params()
