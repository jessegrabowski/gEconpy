from pathlib import Path

import pyparsing as pp

from gEconpy.parser.ast import GCNModel
from gEconpy.parser.constants import BLOCK_COMPONENTS
from gEconpy.parser.error_catalog import ErrorCode
from gEconpy.parser.errors import GCNGrammarError, GCNParseFailure, ParseLocation
from gEconpy.parser.grammar.blocks import MODEL_BLOCK
from gEconpy.parser.grammar.special_blocks import (
    ASSUMPTIONS_BLOCK,
    OPTIONS_BLOCK,
    TRYREDUCE_BLOCK,
)
from gEconpy.parser.grammar.tokens import COMMENT
from gEconpy.parser.suggestions import suggest_block_component

_COMPONENT_KEYWORDS = frozenset(c.lower() for c in BLOCK_COMPONENTS)


def _tag_options(t: list[str]) -> tuple[str, str]:
    return "options", t[0]


def _tag_tryreduce(t: list[str]) -> tuple[str, str]:
    return "tryreduce", t[0]


def _tag_assumptions(t: list[str]) -> tuple[str, str]:
    return "assumptions", t[0]


OPTIONS_TAGGED = OPTIONS_BLOCK.copy().add_parse_action(_tag_options)
TRYREDUCE_TAGGED = TRYREDUCE_BLOCK.copy().add_parse_action(_tag_tryreduce)
ASSUMPTIONS_TAGGED = ASSUMPTIONS_BLOCK.copy().add_parse_action(_tag_assumptions)

SPECIAL_BLOCK = OPTIONS_TAGGED | TRYREDUCE_TAGGED | ASSUMPTIONS_TAGGED

_COMPONENT_KEYWORD = pp.MatchFirst([pp.CaselessKeyword(kw) for kw in BLOCK_COMPONENTS])


def _component_outside_block_fail(s: str, loc: int, toks: list) -> None:
    component = toks[0]
    raise GCNParseFailure(
        s,
        loc,
        f"Component '{component}' found outside of block",
        code=ErrorCode.E016,
        found=component,
    )


ORPHAN_COMPONENT = (_COMPONENT_KEYWORD + pp.FollowedBy(pp.Literal("{"))).set_parse_action(_component_outside_block_fail)

GCN_FILE = pp.ZeroOrMore(SPECIAL_BLOCK)("special_blocks") + pp.OneOrMore(ORPHAN_COMPONENT | MODEL_BLOCK)("model_blocks")

GCN_FILE.ignore(COMMENT)


def _build_model(tokens, filename: str = "") -> GCNModel:
    model = GCNModel(filename=filename)

    for block_type, content in tokens.special_blocks:
        if block_type == "options":
            model.options = content
        elif block_type == "tryreduce":
            model.tryreduce = content
        elif block_type == "assumptions":
            model.assumptions = content

    model.blocks = list(tokens.model_blocks)

    return model


def _convert_parse_exception(exc: pp.ParseBaseException, text: str, filename: str = "") -> GCNGrammarError:
    lines = text.split("\n")

    message, code, found, suggestions = GCNParseFailure.decode(exc)

    if code == ErrorCode.E000:
        handler = StructuralErrorHandler.from_exception(exc)
        code = handler.get_error_code()
        message = handler.get_message()
        annotation = handler.get_annotation()
        notes = handler.get_notes()
        found = handler.found
    else:
        annotation = _get_semantic_annotation(code, found, message)
        notes = _get_semantic_notes(code, found)
        if suggestions and not any("Did you mean" in n for n in notes):
            notes.insert(0, f"Did you mean '{suggestions[0]}'?")

    found_clean = found.strip("'\"")

    line, col = exc.lineno, exc.col
    source_line = lines[line - 1] if 0 < line <= len(lines) else ""

    end_column = None
    if found_clean and col > 0:
        end_column = col + len(found_clean)

    location = ParseLocation(
        line=line,
        column=col,
        end_line=line if end_column else None,
        end_column=end_column,
        source_line=source_line,
        filename=filename,
    )

    return GCNGrammarError(
        message=message,
        expected=None,
        found=found if found else None,
        location=location,
        code=code,
        annotation=annotation,
        notes=notes,
    )


def _get_semantic_annotation(code: ErrorCode, found: str, message: str = "") -> str:
    if code == ErrorCode.E010:
        return f"invalid time index '{found}'"

    if code == ErrorCode.E005 and message:
        return message.lower()

    return code.title.lower()


def _get_semantic_notes(code: ErrorCode, found: str) -> list[str]:
    notes = []

    if code == ErrorCode.E013:
        suggestions = suggest_block_component(found)
        if suggestions:
            notes.append(f"Did you mean '{suggestions[0]}'?")

    notes.extend(code.info.fixes)

    return notes


class StructuralErrorHandler:
    """
    Centralized handler for structural parse errors from pyparsing.

    Structural errors (missing semicolons, unbalanced braces/parens, etc.) come from
    pyparsing's internal matching failures. This class maps pyparsing error patterns
    to our error codes and generates appropriate messages, annotations, and notes.
    """

    def __init__(self, msg: str, expected: str, found: str):
        self.msg = msg
        self.expected = expected
        self.found = found
        self.found_clean = found.strip("'\"")
        self._match: ErrorCode | None = None

    def _is_semicolon_error(self) -> bool:
        if "Expected ';'" in self.msg or self.expected in {";", "';'"}:
            return True
        if self.found_clean.lower() in _COMPONENT_KEYWORDS:
            return True
        return (
            "Expected '}'" in self.msg
            and self.found_clean
            and self.found_clean[0].isalpha()
            and "end of text" not in self.found_clean.lower()
        )

    def _is_paren_error(self) -> bool:
        if self.found_clean in {")", "("}:
            return True
        return bool("Expected ')'" in self.msg or "Expected '('" in self.msg)

    def _is_brace_error(self) -> bool:
        if self.found_clean in {"{", "}"}:
            return True
        return ("Expected '{'" in self.msg or "Expected '}'" in self.msg) and (
            not self.found_clean or "end of text" in self.found_clean.lower()
        )

    def _is_missing_rhs_error(self) -> bool:
        return "Expected '+'" in self.msg or "operations" in self.msg.lower()

    def get_error_code(self) -> ErrorCode:
        if self._match is not None:
            return self._match

        if self._is_paren_error():
            self._match = ErrorCode.E007
        elif self._is_semicolon_error():
            self._match = ErrorCode.E001
        elif self._is_brace_error():
            self._match = ErrorCode.E002
        elif self._is_missing_rhs_error():
            self._match = ErrorCode.E005
        else:
            self._match = ErrorCode.E000

        return self._match

    def get_message(self) -> str:
        code = self.get_error_code()

        if code != ErrorCode.E000:
            if code == ErrorCode.E001 and self.found_clean.lower() in _COMPONENT_KEYWORDS:
                return f"{code.title} after previous statement"
            return code.title

        return f"Syntax error: {self.msg}"

    def get_annotation(self) -> str:
        code = self.get_error_code()

        if code != ErrorCode.E000:
            return code.title.lower()

        if self.found:
            return f"unexpected '{self.found}'"

        return ""

    def get_notes(self) -> list[str]:
        code = self.get_error_code()

        if code != ErrorCode.E000:
            return list(code.info.fixes)

        return []

    @classmethod
    def from_exception(cls, exc: pp.ParseBaseException) -> "StructuralErrorHandler":
        msg = str(exc.msg) if hasattr(exc, "msg") else str(exc)
        expected = str(exc.expected) if hasattr(exc, "expected") and exc.expected else ""
        found = exc.found if hasattr(exc, "found") and exc.found else ""
        return cls(msg, expected, found)


def parse_gcn(text: str, filename: str = "") -> GCNModel:
    """Parse a complete GCN file into a GCNModel AST."""
    try:
        result = GCN_FILE.parse_string(text, parse_all=True)
        return _build_model(result, filename=filename)
    except pp.ParseBaseException as exc:
        raise _convert_parse_exception(exc, text, filename) from None


def parse_gcn_file(filepath: str) -> GCNModel:
    """Parse a GCN file from disk."""
    text = Path(filepath).read_text(encoding="utf-8")
    return parse_gcn(text, filename=filepath)


__all__ = [
    "GCN_FILE",
    "parse_gcn",
    "parse_gcn_file",
]
