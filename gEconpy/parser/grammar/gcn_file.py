from pathlib import Path

import pyparsing as pp

from gEconpy.parser.ast import GCNModel
from gEconpy.parser.constants import BLOCK_COMPONENTS, KNOWN_COMPONENTS
from gEconpy.parser.error_catalog import ErrorCode
from gEconpy.parser.errors import GCNGrammarError, GCNParseFailure, ParseLocation
from gEconpy.parser.grammar.blocks import MODEL_BLOCK
from gEconpy.parser.grammar.special_blocks import (
    ASSUMPTIONS_BLOCK,
    OPTIONS_BLOCK,
    TRYREDUCE_BLOCK,
)
from gEconpy.parser.grammar.tokens import COMMENT


def parse_gcn(text: str, filename: str = "") -> GCNModel:
    """
    Parse the text of a complete GCN file into an abstract syntax tree.

    Parameters
    ----------
    text : str
        Contents of the GCN file.
    filename : str, optional
        Name to report in error messages. Defaults to an empty string.

    Returns
    -------
    model : GCNModel
        The parsed model.
    """
    try:
        result = GCN_FILE.parse_string(text, parse_all=True)
    except pp.ParseBaseException as exc:
        raise _convert_parse_exception(exc, text, filename) from None
    return _build_model(result, filename=filename)


def parse_gcn_file(filepath: str) -> GCNModel:
    """
    Read a GCN file from disk and parse it into an abstract syntax tree.

    Parameters
    ----------
    filepath : str
        Path to the GCN file.

    Returns
    -------
    model : GCNModel
        The parsed model.
    """
    text = Path(filepath).read_text(encoding="utf-8")
    return parse_gcn(text, filename=filepath)


def _tagged(block: pp.ParserElement, kind: str) -> pp.ParserElement:
    return block.copy().add_parse_action(lambda t: (kind, t[0]))


SPECIAL_BLOCK = (
    _tagged(OPTIONS_BLOCK, "options")
    | _tagged(TRYREDUCE_BLOCK, "tryreduce")
    | _tagged(ASSUMPTIONS_BLOCK, "assumptions")
)

_COMPONENT_KEYWORD = pp.MatchFirst([pp.CaselessKeyword(kw) for kw in BLOCK_COMPONENTS])


def _component_outside_block_fail(s: str, loc: int, toks: pp.ParseResults) -> None:
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


def _build_model(tokens: pp.ParseResults, filename: str = "") -> GCNModel:
    model = GCNModel(filename=filename)

    for block_type, content in tokens.special_blocks:
        setattr(model, block_type, content)

    model.blocks = list(tokens.model_blocks)

    return model


def _convert_parse_exception(exc: pp.ParseBaseException, text: str, filename: str = "") -> GCNGrammarError:
    message, code, found, suggestions = GCNParseFailure.decode(exc)

    if code == ErrorCode.E000:
        pyparsing_message = str(exc.msg)
        code = _structural_error_code(pyparsing_message, found)
        message = _structural_message(code, pyparsing_message, found)
        annotation = _structural_annotation(code, found)
    else:
        annotation = _semantic_annotation(code, found, message)

    notes = _fix_notes(code)
    if suggestions:
        notes.insert(0, f"Did you mean '{suggestions[0]}'?")

    found_clean = found.strip("'\"")
    line, col = exc.lineno, exc.col
    lines = text.split("\n")
    source_line = lines[line - 1] if 0 < line <= len(lines) else ""

    end_column = col + len(found_clean) if found_clean and col > 0 else None
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
        found=found_clean,
        location=location,
        code=code,
        annotation=annotation,
        notes=notes,
    )


def _semantic_annotation(code: ErrorCode, found: str, message: str) -> str:
    if code == ErrorCode.E010:
        return f"invalid time index '{found}'"

    if code == ErrorCode.E005 and message:
        return message.lower()

    return code.title.lower()


def _fix_notes(code: ErrorCode) -> list[str]:
    if code == ErrorCode.E000:
        return []
    return list(code.info.fixes)


def _structural_error_code(message: str, found: str) -> ErrorCode:
    """
    Map a pyparsing failure with no encoded error data onto the catalog.

    Structural failures (a missing semicolon, an unbalanced brace or parenthesis) come from pyparsing's own
    matching, so the only evidence is its message and the token it stopped on.
    """
    found_clean = found.strip("'\"")
    at_end_of_text = "end of text" in found_clean.lower()

    if found_clean in {")", "("} or "Expected ')'" in message or "Expected '('" in message:
        return ErrorCode.E007

    expected_brace_but_found_word = (
        "Expected '}'" in message and bool(found_clean) and found_clean[0].isalpha() and not at_end_of_text
    )
    if "Expected ';'" in message or found_clean.lower() in KNOWN_COMPONENTS or expected_brace_but_found_word:
        return ErrorCode.E001

    expected_brace = "Expected '{'" in message or "Expected '}'" in message
    if found_clean in {"{", "}"} or (expected_brace and (not found_clean or at_end_of_text)):
        return ErrorCode.E002

    if "Expected '+'" in message or "operations" in message.lower():
        return ErrorCode.E005

    return ErrorCode.E000


def _structural_message(code: ErrorCode, message: str, found: str) -> str:
    if code == ErrorCode.E000:
        return f"Syntax error: {message}"

    if code == ErrorCode.E001 and found.strip("'\"").lower() in KNOWN_COMPONENTS:
        return f"{code.title} after previous statement"

    return code.title


def _structural_annotation(code: ErrorCode, found: str) -> str:
    if code != ErrorCode.E000:
        return code.title.lower()

    if found:
        return f"unexpected '{found}'"

    return ""


__all__ = [
    "GCN_FILE",
    "parse_gcn",
    "parse_gcn_file",
]
