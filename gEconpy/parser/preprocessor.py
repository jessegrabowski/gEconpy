from pathlib import Path
from typing import Any

from gEconpy.parser.ast import GCNModel
from gEconpy.parser.ast.validation import full_validation
from gEconpy.parser.errors import ErrorCollector
from gEconpy.parser.grammar.gcn_file import parse_gcn
from gEconpy.parser.transform.to_distribution import distributions_from_model
from gEconpy.parser.transform.to_sympy import model_to_sympy


class ParseResult:
    """
    Result of parsing a GCN file.

    Contains the AST, validation errors/warnings, and converted sympy equations.
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
        self._sympy_equations: dict | None = None
        self._distributions: dict | None = None

    @property
    def validation_errors(self) -> ErrorCollector:
        """Lazily compute validation errors."""
        if self._validation_errors is None:
            self._validation_errors = full_validation(self.ast)
        return self._validation_errors

    @property
    def has_errors(self) -> bool:
        """Check if there are any validation errors (not warnings)."""
        return self.validation_errors.has_errors

    @property
    def sympy_equations(self) -> dict[str, dict[str, list]]:
        """Lazily convert AST to sympy equations."""
        if self._sympy_equations is None:
            self._sympy_equations = model_to_sympy(self.ast)
        return self._sympy_equations

    @property
    def distributions(self) -> dict[str, tuple[Any, dict]]:
        """Lazily extract distributions from the model."""
        if self._distributions is None:
            self._distributions = distributions_from_model(self.ast)
        return self._distributions

    @property
    def blocks(self):
        """Convenience accessor for model blocks."""
        return self.ast.blocks

    @property
    def options(self):
        """Convenience accessor for model options."""
        return self.ast.options

    @property
    def tryreduce(self):
        """Convenience accessor for tryreduce variables."""
        return self.ast.tryreduce

    @property
    def assumptions(self):
        """Convenience accessor for variable assumptions."""
        return self.ast.assumptions

    def validate(self, raise_on_error: bool = True) -> ErrorCollector:
        """
        Run validation and optionally raise on errors.

        Parameters
        ----------
        raise_on_error : bool
            If True, raise an exception if there are errors.

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
    Parse and preprocess a GCN source string.

    This is the main entry point for parsing GCN files. It handles:
    - Parsing via pyparsing grammar
    - AST construction
    - Optional validation

    Parameters
    ----------
    source : str
        The GCN source text to parse.
    filename : str, optional
        The filename (for error messages).
    validate : bool
        If True, run validation after parsing.

    Returns
    -------
    result : ParseResult
        The parsing result containing AST and metadata.

    Raises
    ------
    GCNGrammarError
        If there are syntax errors in the source.
    GCNSemanticError
        If validate=True and there are semantic errors.
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
    Parse and preprocess a GCN file.

    Parameters
    ----------
    filepath : str | Path
        Path to the GCN file.
    validate : bool
        If True, run validation after parsing.

    Returns
    -------
    result : ParseResult
        The parsing result containing AST and metadata.
    """
    filepath = Path(filepath)
    content = filepath.read_text()
    return preprocess(content, filename=str(filepath), validate=validate)


def quick_parse(source: str) -> GCNModel:
    """
    Parse a GCN source string and return just the AST.

    This is a convenience function for when you just need the AST
    without validation or metadata.

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
