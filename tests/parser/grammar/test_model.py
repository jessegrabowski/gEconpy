from pathlib import Path

import pytest

from gEconpy.parser.ast import GCNBlock, GCNModel
from gEconpy.parser.grammar.model import parse_gcn


class TestParseGCNBasics:
    def test_empty_model(self):
        model = parse_gcn("")
        assert isinstance(model, GCNModel)
        assert model.blocks == []
        assert model.options == {}
        assert model.tryreduce == []

    def test_single_block(self):
        text = """
        block HOUSEHOLD
        {
            identities
            {
                Y[] = C[];
            };
        };
        """
        model = parse_gcn(text)
        assert len(model.blocks) == 1
        assert model.blocks[0].name == "HOUSEHOLD"

    def test_multiple_blocks(self):
        text = """
        block HOUSEHOLD
        {
            identities { Y[] = C[]; };
        };

        block FIRM
        {
            identities { P[] = MC[]; };
        };
        """
        model = parse_gcn(text)
        assert len(model.blocks) == 2
        assert model.block_names() == ["HOUSEHOLD", "FIRM"]

    def test_with_options(self):
        text = """
        options
        {
            output logfile = TRUE;
            output LaTeX = FALSE;
        };

        block TEST
        {
            identities { Y[] = C[]; };
        };
        """
        model = parse_gcn(text)
        assert model.options == {"output logfile": True, "output LaTeX": False}
        assert len(model.blocks) == 1

    def test_with_tryreduce(self):
        text = """
        tryreduce
        {
            U[], TC[];
        };

        block TEST
        {
            identities { Y[] = C[]; };
        };
        """
        model = parse_gcn(text)
        assert set(model.tryreduce) == {"U[]", "TC[]"}

    def test_with_assumptions(self):
        text = """
        assumptions
        {
            positive
            {
                C[], K[];
            };
        };

        block TEST
        {
            identities { Y[] = C[]; };
        };
        """
        model = parse_gcn(text)
        assert model.assumptions["C"]["positive"] is True
        assert model.assumptions["K"]["positive"] is True

    def test_comments_removed(self):
        text = """
        # This is a comment
        block TEST
        {
            # Another comment
            identities { Y[] = C[]; };
        };
        """
        model = parse_gcn(text)
        assert len(model.blocks) == 1


class TestParseGCNFullModels:
    def test_one_block_household(self):
        text = """
        block HOUSEHOLD
        {
            definitions
            {
                u[] = (C[] ^ (1 - gamma) - 1) / (1 - gamma);
            };

            controls
            {
                C[], K[];
            };

            objective
            {
                U[] = u[] + beta * E[][U[1]];
            };

            constraints
            {
                C[] + K[] - (1 - delta) * K[-1] = A[] * K[-1] ^ alpha : lambda[];
            };

            identities
            {
                log(A[]) = rho * log(A[-1]) + epsilon[];
            };

            shocks
            {
                epsilon[];
            };

            calibration
            {
                alpha = 0.4;
                beta = 0.99;
                delta = 0.02;
                rho = 0.95;
                gamma = 1.5;
            };
        };
        """
        model = parse_gcn(text)

        assert len(model.blocks) == 1
        block = model.blocks[0]
        assert block.name == "HOUSEHOLD"
        assert len(block.definitions) == 1
        assert len(block.controls) == 2
        assert block.objective is not None
        assert len(block.constraints) == 1
        assert len(block.identities) == 1
        assert len(block.shocks) == 1
        assert len(block.calibration) == 5

    def test_two_block_model(self):
        text = """
        block HOUSEHOLD
        {
            controls { C[], L[]; };
            objective { U[] = u[] + beta * E[][U[1]]; };
            constraints { C[] = w[] * L[] : lambda[]; };
        };

        block FIRM
        {
            controls { L[], K[-1]; };
            objective { Pi[] = Y[] - w[] * L[] - r[] * K[-1]; };
            constraints { Y[] = A[] * K[-1] ^ alpha * L[] ^ (1 - alpha) : mc[]; };
        };
        """
        model = parse_gcn(text)

        assert len(model.blocks) == 2
        assert model.get_block("HOUSEHOLD") is not None
        assert model.get_block("FIRM") is not None

    def test_full_model_with_all_special_blocks(self):
        text = """
        options
        {
            output logfile = TRUE;
            output LaTeX = TRUE;
        };

        assumptions
        {
            positive
            {
                Y[], C[], K[], L[];
            };
            negative
            {
                TC[];
            };
        };

        tryreduce
        {
            U[];
        };

        block HOUSEHOLD
        {
            controls { C[]; };
            objective { U[] = log(C[]) + beta * E[][U[1]]; };
            constraints { C[] = Y[] : lambda[]; };
        };

        block EQUILIBRIUM
        {
            identities { Y[] = C[]; };
        };
        """
        model = parse_gcn(text)

        assert model.options["output logfile"] is True
        assert model.options["output LaTeX"] is True
        assert model.tryreduce == ["U[]"]
        assert model.assumptions["Y"]["positive"] is True
        assert model.assumptions["TC"]["negative"] is True
        assert len(model.blocks) == 2


class TestParseGCNFiles:
    """Test parsing actual GCN files from test resources."""

    @pytest.fixture
    def gcn_dir(self):
        return Path(__file__).parent.parent.parent / "_resources" / "test_gcns"

    @pytest.fixture
    def gcn_files_dir(self):
        return Path(__file__).parent.parent.parent.parent / "GCN Files"

    def test_parse_one_block_1(self, gcn_dir):
        gcn_path = gcn_dir / "one_block_1.gcn"
        if not gcn_path.exists():
            pytest.skip(f"Test file not found: {gcn_path}")

        text = gcn_path.read_text()
        model = parse_gcn(text)

        assert len(model.blocks) == 1
        assert model.blocks[0].name == "HOUSEHOLD"

    def test_parse_rbc(self, gcn_files_dir):
        gcn_path = gcn_files_dir / "RBC.gcn"
        if not gcn_path.exists():
            pytest.skip(f"Test file not found: {gcn_path}")

        text = gcn_path.read_text()
        model = parse_gcn(text)

        assert len(model.blocks) >= 2
        block_names = model.block_names()
        assert "HOUSEHOLD" in block_names or "STEADY_STATE" in block_names

    def test_parse_basic_rbc(self, gcn_dir):
        gcn_path = gcn_dir / "basic_rbc.gcn"
        if not gcn_path.exists():
            pytest.skip(f"Test file not found: {gcn_path}")

        text = gcn_path.read_text()
        model = parse_gcn(text)

        assert isinstance(model, GCNModel)
        assert len(model.blocks) > 0


class TestModelMethods:
    def test_get_block(self):
        text = """
        block A { identities { X[] = 1; }; };
        block B { identities { Y[] = 2; }; };
        """
        model = parse_gcn(text)

        assert model.get_block("A") is not None
        assert model.get_block("B") is not None
        assert model.get_block("C") is None

    def test_block_names(self):
        text = """
        block FIRST { identities { X[] = 1; }; };
        block SECOND { identities { Y[] = 2; }; };
        block THIRD { identities { Z[] = 3; }; };
        """
        model = parse_gcn(text)
        assert model.block_names() == ["FIRST", "SECOND", "THIRD"]

    def test_all_equations(self):
        text = """
        block A
        {
            identities { X[] = 1; Y[] = 2; };
        };
        block B
        {
            identities { Z[] = 3; };
        };
        """
        model = parse_gcn(text)
        equations = model.all_equations()
        assert len(equations) == 3

    def test_all_variables(self):
        text = """
        block TEST
        {
            identities
            {
                Y[] = C[] + I[];
                K[] = (1 - delta) * K[-1] + I[];
            };
        };
        """
        model = parse_gcn(text)
        variables = model.all_variables()
        var_names = {v.name for v in variables}
        assert var_names == {"Y", "C", "I", "K"}

    def test_all_parameters(self):
        text = """
        block TEST
        {
            identities
            {
                Y[] = alpha * K[-1] ^ beta;
            };
            calibration
            {
                alpha = 0.3;
                beta = 0.7;
            };
        };
        """
        model = parse_gcn(text)
        params = model.all_parameters()
        param_names = {p.name for p in params}
        assert "alpha" in param_names
        assert "beta" in param_names
