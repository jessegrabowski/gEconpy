from pathlib import Path
from typing import Any

from preliz.distributions.distributions import Distribution

from gEconpy.parser.ast import GCNBlock, GCNModel
from gEconpy.parser.ast.validation import full_validation
from gEconpy.parser.errors import ErrorCollector
from gEconpy.parser.grammar.gcn_file import parse_gcn
from gEconpy.parser.transform.to_distribution import distributions_from_model


class ParseResult:
    """
    A parsed GCN file: its AST, source text, and lazily computed validation results and SymPy conversions.

    Parameters
    ----------
    ast : GCNModel
        The parsed model.
    source : str
        The GCN source text the AST was parsed from.
    filename : str, optional
        The file the source came from, reported in error messages. Defaults to None.
    """

    def __init__(
        self,
        ast: GCNModel,
        source: str,
        filename: str | None = None,
    ):
        self.ast = ast
        self.source = source
        self.filename = filename
        self._validation_errors: ErrorCollector | None = None
        self._distributions: dict[str, tuple[Distribution, dict[str, Any]]] | None = None

    @property
    def validation_errors(self) -> ErrorCollector:
        """The validation errors and warnings, computed on first access."""
        if self._validation_errors is None:
            self._validation_errors = full_validation(self.ast)
        return self._validation_errors

    @property
    def has_errors(self) -> bool:
        """True if validation found at least one error-level issue. Warnings alone give False."""
        return self.validation_errors.has_errors

    @property
    def distributions(self) -> dict[str, tuple[Distribution, dict[str, Any]]]:
        """
        The prior and shock distributions declared in the model, computed on first access.

        Returns
        -------
        distributions : dict mapping str to tuple
            For each declared name, the pair returned by
            :func:`~gEconpy.parser.transform.to_distribution.ast_to_distribution_with_metadata`.
        """
        if self._distributions is None:
            self._distributions = distributions_from_model(self.ast)
        return self._distributions

    @property
    def blocks(self) -> list[GCNBlock]:
        """The parsed blocks, in source order."""
        return self.ast.blocks

    @property
    def options(self) -> dict[str, str | bool]:
        """The entries of the ``options`` block."""
        return self.ast.options

    @property
    def tryreduce(self) -> list[str]:
        """The variable names listed in the ``tryreduce`` block."""
        return self.ast.tryreduce

    @property
    def assumptions(self) -> dict[str, dict[str, bool]]:
        """The SymPy assumptions declared per symbol name."""
        return self.ast.assumptions

    def validate(self, raise_on_error: bool = True) -> ErrorCollector:
        """
        Run validation, raising the first error-level issue when asked to.

        Parameters
        ----------
        raise_on_error : bool, optional
            Raise the first error-level issue found. Defaults to True.

        Returns
        -------
        errors : ErrorCollector
            The validation errors and warnings.
        """
        errors = self.validation_errors
        if raise_on_error and errors.has_errors:
            errors.raise_first()
        return errors


def preprocess(
    source: str,
    filename: str | None = None,
    validate: bool = True,
) -> ParseResult:
    """
    Parse GCN source text into an AST and, when asked to, validate it.

    Grammar errors raise :class:`~gEconpy.parser.errors.GCNGrammarError`. Validation collects semantic errors on the
    result without raising, so that callers can report all of them at once.

    Parameters
    ----------
    source : str
        The GCN source text to parse.
    filename : str, optional
        Filename to report in error messages. Defaults to None.
    validate : bool, optional
        Run validation after parsing. Defaults to True.

    Returns
    -------
    result : ParseResult
        The parsed model with its source and validation results.
    """
    ast = parse_gcn(source, filename=filename or "<string>")
    result = ParseResult(ast=ast, source=source, filename=filename)

    if validate:
        result.validate(raise_on_error=False)

    return result


def preprocess_file(
    filepath: str | Path,
    validate: bool = True,
) -> ParseResult:
    """
    Read a GCN file and parse it with :func:`~gEconpy.parser.preprocessor.preprocess`.

    Parameters
    ----------
    filepath : str or Path
        Path to the GCN file.
    validate : bool, optional
        Run validation after parsing. Defaults to True.

    Returns
    -------
    result : ParseResult
        The parsed model with its source and validation results.
    """
    filepath = Path(filepath)
    content = filepath.read_text(encoding="utf-8")
    return preprocess(content, filename=str(filepath), validate=validate)


def quick_parse(source: str) -> GCNModel:
    """
    Parse GCN source text and return only the AST, skipping validation.

    Parameters
    ----------
    source : str
        The GCN source text to parse.

    Returns
    -------
    model : GCNModel
        The parsed AST.
    """
    return parse_gcn(source)
