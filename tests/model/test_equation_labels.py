import pytest

from gEconpy import model_from_gcn
from gEconpy.data import get_example_gcn
from gEconpy.parser.loader import load_gcn_string
from tests.conftest import TEST_GCNS


@pytest.fixture
def labelled_model_path(tmp_path):
    """``one_block_2.gcn`` with one constraint and one control labeled, as a file the full pipeline can load."""
    source = (TEST_GCNS / "one_block_2.gcn").read_text()
    labelled = source.replace(
        "        I[] = Y[] - C[] : lambda[];",
        '        @name = "Resource constraint" I[] = Y[] - C[] : lambda[];',
        1,
    ).replace(
        "        C[], L[], I[], K[], Y[];",
        '        @foc_name = "Consumption Euler equation"\n        C[];\n        L[], I[], K[], Y[];',
        1,
    )

    # Editing the shared fixture would otherwise leave this a model with no labels at all, which several of the
    # assertions below are satisfied by.
    assert "@name" in labelled and "@foc_name" in labelled, "one_block_2.gcn changed; update the replacements"

    path = tmp_path / "labelled.gcn"
    path.write_text(labelled)
    return path


def test_a_label_survives_the_whole_pipeline(labelled_model_path):
    """
    The label rides on the equation id, so multiplier elimination and tryreduce carry it without knowing about it.

    ``one_block_2.gcn`` eliminates a multiplier and reduces ``C[]``, which is what makes this more than an AST
    assertion.
    """
    model = model_from_gcn(labelled_model_path, verbose=False)

    assert model._equation_labels["HOUSEHOLD.constraints.1"] == "Resource constraint"
    assert model._equation_labels["HOUSEHOLD.foc.C"] == "Consumption Euler equation"


def test_every_labelled_id_still_names_a_surviving_equation(labelled_model_path):
    """A label for an equation the simplifier dropped would point at nothing."""
    model = model_from_gcn(labelled_model_path, verbose=False)

    assert model._equation_labels
    assert set(model._equation_labels) <= set(model._equation_ids)


def test_an_unlabelled_model_carries_no_labels():
    assert model_from_gcn(TEST_GCNS / "open_rbc.gcn", verbose=False)._equation_labels == {}


def test_a_control_caption_lands_on_the_derived_condition_not_the_control():
    source = """
    block HOUSEHOLD
    {
        controls { @foc_name = "Consumption Euler equation" C[]; L[]; };
        objective { U[] = log(C[]) - L[] + 0.99 * E[][U[1]]; };
        constraints { @name = "Budget constraint" C[] = w[] * L[] : lambda[]; };
    };
    """

    labels = load_gcn_string(source).equation_labels

    assert labels == {
        "HOUSEHOLD.constraints.0": "Budget constraint",
        "HOUSEHOLD.foc.C": "Consumption Euler equation",
    }


class TestGeneratedCaptions:
    def test_an_unannotated_first_order_condition_gets_a_caption_from_its_block_and_symbol(self):
        """The default is what most models print, since nobody annotates every control."""
        model = model_from_gcn(get_example_gcn("RBC"), verbose=False)

        assert model.equation_label("HOUSEHOLD.foc.C") == "Household first-order condition for Consumption"
        assert model.equation_label("FIRM.foc.L") == "Firm first-order condition for Hours worked"

    def test_a_multi_word_block_name_reads_as_a_sentence(self):
        """An underscore is a word break, and only the first word is capitalized."""
        model = model_from_gcn(get_example_gcn("RBC_two_household"), verbose=False)

        label = model.equation_label("RICARDIAN_HOUSEHOLD.foc.C_R")

        assert label.startswith("Ricardian household first-order condition for ")

    def test_a_symbol_without_a_declared_name_falls_back_to_its_identifier(self):
        model = model_from_gcn(TEST_GCNS / "open_rbc.gcn", verbose=False)

        assert model.equation_label("HOUSEHOLD.foc.C") == "Household first-order condition for C"

    @pytest.mark.parametrize(
        "equation_id",
        ["HOUSEHOLD.identities.0", "HOUSEHOLD.constraints.0", "HOUSEHOLD.objective"],
        ids=["identity", "constraint", "objective"],
    )
    def test_nothing_is_derived_for_an_equation_the_author_wrote(self, equation_id):
        """An authored equation has source text to label, so inventing a caption for it would be guessing."""
        model = model_from_gcn(TEST_GCNS / "open_rbc.gcn", verbose=False)

        assert model.equation_label(equation_id) is None

    def test_an_authored_caption_beats_the_generated_one(self, labelled_model_path):
        model = model_from_gcn(labelled_model_path, verbose=False)

        assert model.equation_label("HOUSEHOLD.foc.C") == "Consumption Euler equation"
        assert model.equation_label("HOUSEHOLD.foc.L") == "Household first-order condition for L"
