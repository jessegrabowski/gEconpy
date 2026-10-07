import pytest

from pyparsing import ParseBaseException

from gEconpy.classes.time_aware_symbol import DEFAULT_ASSUMPTIONS
from gEconpy.exceptions import DeprecatedAssumptionsBlockWarning
from gEconpy.parser.ast import SymbolDeclaration
from gEconpy.parser.error_catalog import ErrorCode
from gEconpy.parser.errors import GCNParseFailure
from gEconpy.parser.grammar.special_blocks import (
    ASSUMPTIONS_BLOCK,
    OPTIONS_BLOCK,
    SYMBOLS_BLOCK,
    TRYREDUCE_BLOCK,
    extract_special_block_content,
    parse_assumptions,
    parse_options,
    parse_symbols,
    parse_tryreduce,
    remove_special_block,
)


def parse_symbols_block(text: str) -> dict[str, SymbolDeclaration]:
    return SYMBOLS_BLOCK.parse_string(text)[0]


def error_code_of(text: str) -> ErrorCode:
    with pytest.raises(ParseBaseException) as exc_info:
        SYMBOLS_BLOCK.parse_string(text)
    _message, code, _found, _suggestions = GCNParseFailure.decode(exc_info.value)
    return code


class TestOptionsBlock:
    def test_empty_options(self):
        text = "options { };"
        result = OPTIONS_BLOCK.parse_string(text)[0]
        assert result == {}

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("options { verbose = TRUE; };", {"verbose": True}),
            ("options { verbose = FALSE; };", {"verbose": False}),
            ("options { verbose = true; };", {"verbose": True}),
            ("options { solver = gensys; };", {"solver": "gensys"}),
            ("options { output logfile = TRUE; };", {"output logfile": True}),
            ("OPTIONS { verbose = TRUE; };", {"verbose": True}),
        ],
        ids=["true", "false", "lowercase_true", "identifier", "multi_word_key", "uppercase_keyword"],
    )
    def test_single_option(self, text, expected):
        assert OPTIONS_BLOCK.parse_string(text)[0] == expected

    def test_multiple_options(self):
        text = """options {
            verbose = TRUE;
            output = latex;
            debug = FALSE;
        };"""
        result = OPTIONS_BLOCK.parse_string(text)[0]
        assert result["verbose"] is True
        assert result["output"] == "latex"
        assert result["debug"] is False


class TestTryreduceBlock:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("tryreduce { };", []),
            ("tryreduce { U[]; };", ["U"]),
            ("tryreduce { U[], TC[], Div[]; };", ["U", "TC", "Div"]),
            ("TRYREDUCE { U[]; };", ["U"]),
            ("tryreduce\n{\n    U[], TC[];\n};", ["U", "TC"]),
        ],
        ids=["empty", "single", "multiple", "uppercase_keyword", "multiline"],
    )
    def test_tryreduce(self, text, expected):
        assert TRYREDUCE_BLOCK.parse_string(text)[0] == expected


@pytest.mark.filterwarnings("ignore::gEconpy.exceptions.DeprecatedAssumptionsBlockWarning")
class TestAssumptionsBlock:
    def test_single_assumption_single_variable(self):
        text = "assumptions { positive { C[]; }; };"
        result = ASSUMPTIONS_BLOCK.parse_string(text)[0]
        assert "C_t" in result
        assert result["C_t"]["positive"] is True

    @pytest.mark.parametrize(
        "text,expected_names",
        [
            ("assumptions { positive { C[], K[], L[]; }; };", ["C_t", "K_t", "L_t"]),
            ("assumptions { positive { alpha, beta; }; };", ["alpha", "beta"]),
            ("assumptions { positive { C[], alpha, K[], beta; }; };", ["C_t", "alpha", "K_t", "beta"]),
            ("ASSUMPTIONS { positive { C[]; }; };", ["C_t"]),
        ],
        ids=["variables", "parameters", "mixed", "uppercase_keyword"],
    )
    def test_assumption_applies_to_every_listed_name(self, text, expected_names):
        result = ASSUMPTIONS_BLOCK.parse_string(text)[0]
        assert list(result) == expected_names
        assert all(result[name]["positive"] is True for name in expected_names)

    def test_multiple_assumptions(self):
        text = """assumptions {
            positive { C[], K[]; };
            real { shock[]; };
        };"""
        result = ASSUMPTIONS_BLOCK.parse_string(text)[0]
        assert result["C_t"]["positive"] is True
        assert result["K_t"]["positive"] is True
        assert result["shock_t"]["real"] is True

    def test_name_in_several_subblocks_accumulates_assumptions(self):
        text = "assumptions { positive { C[]; }; negative { K[]; }; unit_interval { C[]; }; };"
        result = ASSUMPTIONS_BLOCK.parse_string(text)[0]
        assert result == {
            "C_t": {**DEFAULT_ASSUMPTIONS, "positive": True, "unit_interval": True},
            "K_t": {**DEFAULT_ASSUMPTIONS, "negative": True},
        }

    def test_listed_names_start_from_package_defaults(self):
        result = ASSUMPTIONS_BLOCK.parse_string("assumptions { positive { C[]; }; };")[0]
        assert result["C_t"] == {**DEFAULT_ASSUMPTIONS, "positive": True}

    @pytest.mark.parametrize(
        "declared, expected",
        [
            ("(0, None)", {"positive"}),
            ("[0, None)", {"nonnegative"}),
            ("(None, 0)", {"negative"}),
            ("(None, 0]", {"nonpositive"}),
            ("(0, 1)", {"positive", "unit_interval"}),
            ("[0, 1]", {"nonnegative"}),
        ],
        ids=["open_lower", "closed_lower", "open_upper", "closed_upper", "unit", "fully_closed"],
    )
    def test_brackets_choose_the_strict_or_closed_predicate(self, declared, expected):
        """A bracket says whether the endpoint belongs to the support, exactly as a written interval would."""
        result = parse_symbols_block(f"symbols {{ alpha {{ bounds = {declared}; }}; }};")
        derived = {key for key, value in result["alpha"].assumptions.items() if value} - set(DEFAULT_ASSUMPTIONS)

        assert derived == expected

    @pytest.mark.parametrize(
        "keyword, expected_closed",
        [("positive", (False, False)), ("nonnegative", (True, False)), ("nonpositive", (False, True))],
    )
    def test_sign_keywords_agree_with_the_bracket_they_stand_for(self, keyword, expected_closed):
        result = parse_symbols_block(f"symbols {{ alpha {{ {keyword} = True; }}; }};")

        assert result["alpha"].closed == expected_closed

    def test_unit_interval_implies_positive(self):
        text = "assumptions { unit_interval { alpha; }; };"
        result = ASSUMPTIONS_BLOCK.parse_string(text)[0]
        assert result["alpha"]["unit_interval"] is True
        assert result["alpha"]["positive"] is True

    def test_case_insensitive_assumption(self):
        text = "assumptions { POSITIVE { C[]; }; };"
        result = ASSUMPTIONS_BLOCK.parse_string(text)[0]
        assert result["C_t"]["positive"] is True

    def test_warns_and_names_the_symbols_replacement(self):
        with pytest.warns(DeprecatedAssumptionsBlockWarning) as record:
            ASSUMPTIONS_BLOCK.parse_string("assumptions { positive { C[]; }; unit_interval { alpha; }; };")

        message = str(record[0].message)
        assert "C[] { positive = True; };" in message
        assert "alpha { positive = True; unit_interval = True; };" in message

    def test_empty_assumptions(self):
        text = "assumptions { };"
        result = ASSUMPTIONS_BLOCK.parse_string(text)[0]
        assert result == {}


class TestSymbolsBlock:
    def test_empty_block(self):
        assert parse_symbols_block("symbols { };") == {}

    def test_entry_with_no_fields_declares_the_symbol(self):
        result = parse_symbols_block("symbols { alpha { }; };")
        assert result["alpha"].bounds == (None, None)
        assert result["alpha"].name is None

    @pytest.mark.parametrize(
        "text,attribute,expected",
        [
            ('symbols { alpha { name = "Capital share"; }; };', "name", "Capital share"),
            ('symbols { alpha { latex = "\\alpha"; }; };', "latex", "\\alpha"),
            ('symbols { alpha { typst = "alpha"; }; };', "typst", "alpha"),
            ('symbols { alpha { source = "Smets and Wouters (2007)"; }; };', "source", "Smets and Wouters (2007)"),
        ],
        ids=["name", "latex_is_raw", "typst", "source"],
    )
    def test_metadata_fields(self, text, attribute, expected):
        assert getattr(parse_symbols_block(text)["alpha"], attribute) == expected

    @pytest.mark.parametrize(
        "declared,expected",
        [
            ("(0, 1)", (0.0, 1.0)),
            ("(0, None)", (0.0, None)),
            ("(None, 0)", (None, 0.0)),
            ("(None, None)", (None, None)),
            ("(-1.5, 2.5)", (-1.5, 2.5)),
            ("(1e-3, None)", (1e-3, None)),
        ],
        ids=["unit", "half_open_above", "half_open_below", "unbounded", "negative_float", "scientific"],
    )
    def test_bounds_forms(self, declared, expected):
        result = parse_symbols_block(f"symbols {{ alpha {{ bounds = {declared}; }}; }};")
        assert result["alpha"].bounds == expected

    @pytest.mark.parametrize(
        "keyword,expected_bounds",
        [
            ("positive", (0.0, None)),
            ("nonnegative", (0.0, None)),
            ("negative", (None, 0.0)),
            ("nonpositive", (None, 0.0)),
            ("unit_interval", (0.0, 1.0)),
        ],
        ids=["positive", "nonnegative", "negative", "nonpositive", "unit_interval"],
    )
    def test_sign_keywords_are_sugar_for_bounds(self, keyword, expected_bounds):
        result = parse_symbols_block(f"symbols {{ alpha {{ {keyword} = True; }}; }};")
        assert result["alpha"].bounds == expected_bounds
        assert result["alpha"].assumptions[keyword] is True

    @pytest.mark.parametrize(
        "declared,expected",
        [
            ("(0, None)", {"positive"}),
            ("(1, None)", {"positive"}),
            ("(None, 0)", {"negative"}),
            ("(None, -2)", {"negative"}),
            ("(0, 1)", {"positive", "unit_interval"}),
            ("(None, None)", set()),
            ("(-1, 1)", set()),
        ],
        ids=["at_zero", "above_zero", "to_zero", "below_zero", "unit", "unbounded", "straddling"],
    )
    def test_bounds_derive_sympy_predicates(self, declared, expected):
        result = parse_symbols_block(f"symbols {{ alpha {{ bounds = {declared}; }}; }};")
        derived = {key for key, value in result["alpha"].assumptions.items() if value} - set(DEFAULT_ASSUMPTIONS)
        assert derived == expected

    def test_unit_interval_implies_positive(self):
        result = parse_symbols_block("symbols { alpha { unit_interval = True; }; };")
        assert result["alpha"].assumptions["unit_interval"] is True
        assert result["alpha"].assumptions["positive"] is True

    def test_non_sign_assumptions_pass_through(self):
        result = parse_symbols_block("symbols { N { integer = True; nonzero = True; real = False; }; };")
        assert result["N"].assumptions["integer"] is True
        assert result["N"].assumptions["nonzero"] is True
        assert result["N"].assumptions["real"] is False
        assert result["N"].bounds == (None, None)

    def test_entries_start_from_package_defaults(self):
        result = parse_symbols_block("symbols { alpha { positive = True; }; };")
        assert result["alpha"].assumptions == {**DEFAULT_ASSUMPTIONS, "positive": True}

    def test_a_variable_and_a_parameter_of_one_name_are_separate_entries(self):
        result = parse_symbols_block("symbols { beta[] { bounds = (0, 1); }; beta { bounds = (2, None); }; };")

        assert set(result) == {"beta_t", "beta"}
        assert result["beta_t"].assumptions["unit_interval"] is True
        assert "unit_interval" not in result["beta"].assumptions

    def test_declared_bounds_win_over_keyword_bounds(self):
        result = parse_symbols_block("symbols { psi { positive = True; bounds = (1, None); }; };")
        assert result["psi"].bounds == (1.0, None)

    def test_keywords_and_case_are_insensitive(self):
        result = parse_symbols_block("SYMBOLS { alpha { POSITIVE = true; }; };")
        assert result["alpha"].assumptions["positive"] is True

    def test_comments_are_ignored(self):
        text = """symbols {
            # the capital share
            alpha { bounds = (0, 1); };  # bounded to the unit interval
        };"""
        assert parse_symbols_block(text)["alpha"].bounds == (0.0, 1.0)

    def test_all_fields_together(self):
        text = """symbols {
            alpha {
                name = "Capital share of output";
                latex = "\\alpha";
                typst = "alpha";
                source = "Smets and Wouters (2007)";
                bounds = (0, 1);
            };
        };"""
        declaration = parse_symbols_block(text)["alpha"]
        assert declaration.symbol == "alpha"
        assert declaration.name == "Capital share of output"
        assert declaration.latex == "\\alpha"
        assert declaration.typst == "alpha"
        assert declaration.source == "Smets and Wouters (2007)"
        assert declaration.bounds == (0.0, 1.0)

    @pytest.mark.parametrize(
        "text,code",
        [
            ("symbols { alpha { negative = True; bounds = (0, 1); }; };", ErrorCode.E017),
            ("symbols { alpha { unit_interval = True; bounds = (2, 3); }; };", ErrorCode.E017),
            ("symbols { alpha { positive = True; negative = True; }; };", ErrorCode.E017),
            ("symbols { alpha { unit_interval = True; negative = True; }; };", ErrorCode.E017),
            ("symbols { alpha { bounds = (1, 0); }; };", ErrorCode.E017),
            ("symbols { alpha { bounds = (1, 1); }; };", ErrorCode.E017),
            ("symbols { alpha { bounds = [None, 0); }; };", ErrorCode.E021),
            ("symbols { alpha { positive = False; }; };", ErrorCode.E018),
            ("symbols { alpha { unit_interval = False; }; };", ErrorCode.E018),
            ("symbols { alpha { }; alpha { }; };", ErrorCode.E019),
            ("symbols { alpha { bound = (0, 1); }; };", ErrorCode.E020),
            ('symbols { alpha { long_name = "x"; }; };', ErrorCode.E020),
        ],
        ids=[
            "bound_contradicts_sign",
            "bound_disjoint_from_unit_interval",
            "keywords_exclude_each_other",
            "unit_interval_excluded_by_sign",
            "bounds_inverted",
            "bounds_degenerate",
            "closed_unbounded_side",
            "positive_false",
            "unit_interval_false",
            "duplicate_symbol",
            "misspelled_field",
            "dynare_field_name",
        ],
    )
    def test_rejected_declarations(self, text, code):
        assert error_code_of(text) is code

    @pytest.mark.parametrize(
        "text,symbol,expected_bounds",
        [
            ("symbols { alpha { positive = True; bounds = (0, 1); }; };", "alpha", (0.0, 1.0)),
            ("symbols { alpha { positive = True; bounds = (0, None); }; };", "alpha", (0.0, None)),
            ("symbols { z { real = False; }; };", "z", (None, None)),
            ("symbols { z { integer = False; }; };", "z", (None, None)),
            ("symbols { alpha { nonnegative = True; unit_interval = True; }; };", "alpha", (0.0, 1.0)),
            ("symbols { psi { positive = True; nonzero = True; }; };", "psi", (0.0, None)),
        ],
        ids=[
            "consistent_overlap",
            "identical_support",
            "real_false_allowed",
            "integer_false_allowed",
            "keywords_intersect",
            "sign_keyword_with_non_sign_keyword",
        ],
    )
    def test_accepted_declarations(self, text, symbol, expected_bounds):
        assert parse_symbols_block(text)[symbol].bounds == expected_bounds


class TestParseSymbolsFn:
    def test_finds_symbols_in_larger_text(self):
        text = """
        symbols { alpha { name = "Capital share"; }; };

        block Household { };
        """
        assert parse_symbols(text)["alpha"].name == "Capital share"

    def test_returns_empty_when_no_symbols(self):
        assert parse_symbols("block Household { };") == {}


class TestParseOptionsFn:
    def test_finds_options_in_larger_text(self):
        text = """
        options { verbose = TRUE; };

        block Household { };
        """
        result = parse_options(text)
        assert result["verbose"] is True

    def test_returns_empty_when_no_options(self):
        text = "block Household { };"
        result = parse_options(text)
        assert result == {}


class TestParseTryreduceFn:
    def test_finds_tryreduce_in_larger_text(self):
        text = """
        tryreduce { U[], TC[]; };

        block Household { };
        """
        result = parse_tryreduce(text)
        assert result == ["U", "TC"]

    def test_returns_empty_when_no_tryreduce(self):
        text = "block Household { };"
        result = parse_tryreduce(text)
        assert result == []


@pytest.mark.filterwarnings("ignore::gEconpy.exceptions.DeprecatedAssumptionsBlockWarning")
class TestParseAssumptionsFn:
    def test_finds_assumptions_in_larger_text(self):
        text = """
        assumptions { positive { C[]; }; };

        block Household { };
        """
        result = parse_assumptions(text)
        assert "C_t" in result

    def test_returns_default_when_no_assumptions(self):
        text = "block Household { };"
        result = parse_assumptions(text)
        assert result["anything"] == DEFAULT_ASSUMPTIONS


class TestSpecialBlockText:
    def test_extract_block_text(self):
        text = "options { verbose = TRUE; };\nblock Household { };"
        assert extract_special_block_content(text, "options") == "options { verbose = TRUE; };"
        assert extract_special_block_content(text, "OPTIONS") == "options { verbose = TRUE; };"
        assert extract_special_block_content(text, "tryreduce") is None

    def test_remove_block_text(self):
        text = "options { verbose = TRUE; };\nblock Household { };"
        assert remove_special_block(text, "options") == "\nblock Household { };"
        assert remove_special_block(text, "tryreduce") == text


class TestSpecialBlockErrors:
    def test_options_missing_semicolon(self):
        with pytest.raises(ParseBaseException):
            OPTIONS_BLOCK.parse_string("options { verbose = TRUE }")

    def test_tryreduce_missing_semicolon(self):
        with pytest.raises(ParseBaseException):
            TRYREDUCE_BLOCK.parse_string("tryreduce { U[] }")

    def test_assumptions_invalid_assumption(self):
        with pytest.raises(ParseBaseException, match="Unknown assumption 'invalid'"):
            ASSUMPTIONS_BLOCK.parse_string("assumptions { invalid { C[]; }; };")
