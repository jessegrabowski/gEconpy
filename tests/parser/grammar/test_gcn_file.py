import pytest

from gEconpy.data.examples import get_example_gcn
from gEconpy.parser.ast import GCNModel
from gEconpy.parser.errors import GCNGrammarError
from gEconpy.parser.grammar.gcn_file import parse_gcn, parse_gcn_file
from tests.conftest import TEST_GCNS


class TestGCNFileBasic:
    def test_single_empty_block(self):
        text = "block TEST { };"
        result = parse_gcn(text)
        assert isinstance(result, GCNModel)
        assert len(result.blocks) == 1
        assert result.blocks[0].name == "TEST"

    def test_multiple_empty_blocks(self):
        text = """
        block HOUSEHOLD { };
        block FIRM { };
        """
        result = parse_gcn(text)
        assert len(result.blocks) == 2
        assert result.blocks[0].name == "HOUSEHOLD"
        assert result.blocks[1].name == "FIRM"

    def test_filename_preserved(self):
        text = "block TEST { };"
        result = parse_gcn(text, filename="test.gcn")
        assert result.filename == "test.gcn"


class TestGCNFileWithSpecialBlocks:
    def test_options_block(self):
        text = """
        options {
            verbose = TRUE;
            output = latex;
        };

        block TEST { };
        """
        result = parse_gcn(text)
        assert result.options["verbose"] is True
        assert result.options["output"] == "latex"

    def test_tryreduce_block(self):
        text = """
        tryreduce {
            U[], TC[];
        };

        block TEST { };
        """
        result = parse_gcn(text)
        assert result.tryreduce == ["U", "TC"]

    def test_assumptions_block(self):
        text = """
        assumptions {
            positive { C[], K[], alpha; };
        };

        block TEST { };
        """
        result = parse_gcn(text)
        assert "C" in result.assumptions
        assert "K" in result.assumptions
        assert "alpha" in result.assumptions

    def test_all_special_blocks(self):
        text = """
        options { verbose = TRUE; };

        tryreduce { U[]; };

        assumptions {
            positive { C[]; };
        };

        block TEST { };
        """
        result = parse_gcn(text)
        assert result.options["verbose"] is True
        assert result.tryreduce == ["U"]
        assert "C" in result.assumptions


class TestGCNFileWithComponents:
    def test_block_with_identities(self):
        text = """
        block EQUILIBRIUM {
            identities {
                Y[] = C[] + I[];
            };
        };
        """
        result = parse_gcn(text)
        assert len(result.blocks[0].identities) == 1

    def test_block_with_calibration(self):
        text = """
        block TEST {
            calibration {
                beta = 0.99;
                alpha ~ Beta(a=2, b=5) = 0.35;
            };
        };
        """
        result = parse_gcn(text)
        assert len(result.blocks[0].calibration) == 2


class TestSimpleRBCModel:
    def test_simple_rbc(self):
        text = """
        tryreduce {
            U[], TC[];
        };

        block STEADY_STATE {
            identities {
                A[ss] = 1;
                r[ss] = 1 / beta - (1 - delta);
            };
        };

        block HOUSEHOLD {
            definitions {
                u[] = C[] ^ (1 - sigma_C) / (1 - sigma_C);
            };

            controls {
                C[], L[], I[], K[];
            };

            objective {
                U[] = u[] + beta * E[][U[1]];
            };

            constraints {
                C[] + I[] = r[] * K[-1] + w[] * L[] : lambda[];
                K[] = (1 - delta) * K[-1] + I[];
            };

            calibration {
                beta = 0.99;
                delta = 0.02;
            };
        };

        block FIRM {
            controls {
                K[-1], L[];
            };

            objective {
                TC[] = -(r[] * K[-1] + w[] * L[]);
            };

            constraints {
                Y[] = A[] * K[-1] ^ alpha * L[] ^ (1 - alpha) : mc[];
            };

            calibration {
                alpha = 0.35;
            };
        };

        block SHOCKS {
            identities {
                log(A[]) = rho_A * log(A[-1]) + epsilon_A[];
            };

            shocks {
                epsilon_A[];
            };

            calibration {
                rho_A = 0.95;
            };
        };
        """
        result = parse_gcn(text)

        assert result.block_names() == ["STEADY_STATE", "HOUSEHOLD", "FIRM", "SHOCKS"]
        assert result.tryreduce == ["U", "TC"]

        household = result.get_block("HOUSEHOLD")
        assert len(household.definitions) == 1
        assert len(household.controls) == 4
        assert len(household.objective) == 1
        assert len(household.constraints) == 2
        assert len(household.calibration) == 2

        firm = result.get_block("FIRM")
        assert len(firm.controls) == 2
        assert firm.constraints[0].lagrange_multiplier == "mc"


class TestGCNFileWithComments:
    def test_comments_ignored(self):
        text = """
        # This is a file-level comment

        block TEST {
            # Block comment
            identities {
                # Equation comment
                Y[] = C[];  # Inline comment
            };
        };
        """
        result = parse_gcn(text)
        assert len(result.blocks) == 1
        assert len(result.blocks[0].identities) == 1

    def test_comment_before_special_block(self):
        text = """
        # Options for model
        options { verbose = TRUE; };

        block TEST { };
        """
        result = parse_gcn(text)
        assert result.options["verbose"] is True


class TestGCNModelMethods:
    def test_get_block(self):
        text = """
        block HOUSEHOLD { };
        block FIRM { };
        """
        result = parse_gcn(text)
        assert result.get_block("HOUSEHOLD") is not None
        assert result.get_block("FIRM") is not None
        assert result.get_block("NONEXISTENT") is None

    def test_block_names(self):
        text = """
        block HOUSEHOLD { };
        block FIRM { };
        block SHOCKS { };
        """
        result = parse_gcn(text)
        assert result.block_names() == ["HOUSEHOLD", "FIRM", "SHOCKS"]

    def test_all_equations(self):
        text = """
        block TEST {
            identities {
                Y[] = C[];
                K[] = I[];
            };
            constraints {
                C[] = Y[] : lambda[];
            };
        };
        """
        result = parse_gcn(text)
        equations = result.all_equations()
        assert len(equations) == 3

    def test_all_variables(self):
        text = """
        block TEST {
            identities {
                Y[] = C[] + I[];
            };
        };
        """
        result = parse_gcn(text)
        variables = result.all_variables()
        var_names = {v.name for v in variables}
        assert "Y" in var_names
        assert "C" in var_names
        assert "I" in var_names

    def test_all_parameters(self):
        text = """
        block TEST {
            identities {
                Y[] = alpha * K[] + beta * L[];
            };
        };
        """
        result = parse_gcn(text)
        parameters = result.all_parameters()
        param_names = {p.name for p in parameters}
        assert "alpha" in param_names
        assert "beta" in param_names


class TestGCNFileErrors:
    @pytest.mark.parametrize(
        "text,match",
        [
            ("options { verbose = TRUE; };", "Syntax error"),
            ("block TEST { identities { Y[] = C[]; }", "Missing semicolon"),
            ("block TEST { invalid_component { }; };", "Unknown component"),
            ("block TEST { }", "semicolon"),
            ("identities { Y[] = C[]; };", "outside of block"),
        ],
        ids=["no_blocks", "unclosed_block", "unknown_component", "missing_semicolon", "orphan_component"],
    )
    def test_invalid_file_raises(self, text, match):
        with pytest.raises(GCNGrammarError, match=match):
            parse_gcn(text)


class TestGCNFileEdgeCases:
    def test_empty_special_blocks(self):
        text = """
        options { };
        tryreduce { };
        assumptions { };
        block TEST { };
        """
        result = parse_gcn(text)
        assert result.options == {}
        assert result.tryreduce == []
        assert result.assumptions == {}

    def test_last_special_block_of_a_type_wins(self):
        text = """
        options { verbose = TRUE; };
        options { verbose = FALSE; };
        block TEST { };
        """
        result = parse_gcn(text)
        assert result.options["verbose"] is False


class TestParseGCNFiles:
    @pytest.mark.parametrize(
        "gcn_path,expected_blocks",
        [
            (TEST_GCNS / "one_block_1.gcn", ["HOUSEHOLD"]),
            (TEST_GCNS / "basic_rbc.gcn", ["HOUSEHOLD", "FIRM", "TECHNOLOGY_SHOCKS"]),
            (get_example_gcn("RBC"), ["STEADY_STATE", "HOUSEHOLD", "FIRM", "TECHNOLOGY_SHOCKS"]),
        ],
        ids=["one_block_1", "basic_rbc", "RBC"],
    )
    def test_parse_file_from_disk(self, gcn_path, expected_blocks):
        model = parse_gcn_file(str(gcn_path))
        assert model.block_names() == expected_blocks
        assert model.filename == str(gcn_path)
