import pytest

from gEconpy import model_from_gcn
from gEconpy.parser.loader import load_gcn_string
from tests._resources.cache_compiled_models import load_and_cache_example
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

    assert model._equation_labels["Household.constraints.1"] == "Resource constraint"
    assert model._equation_labels["Household.foc.C"] == "Consumption Euler equation"


def test_every_labelled_id_still_names_a_surviving_equation(labelled_model_path):
    """A label for an equation the simplifier dropped would point at nothing."""
    model = model_from_gcn(labelled_model_path, verbose=False)

    assert model._equation_labels
    assert set(model._equation_labels) <= set(model._equation_ids)


def test_an_unlabelled_model_carries_no_labels():
    assert model_from_gcn(TEST_GCNS / "open_rbc.gcn", verbose=False)._equation_labels == {}


def test_a_control_caption_lands_on_the_derived_condition_not_the_control():
    source = """
    block Household
    {
        controls { @foc_name = "Consumption Euler equation" C[]; L[]; };
        objective { U[] = log(C[]) - L[] + 0.99 * E[][U[1]]; };
        constraints { @name = "Budget constraint" C[] = w[] * L[] : lambda[]; };
    };
    """

    labels = load_gcn_string(source).equation_labels

    assert labels == {
        "Household.constraints.0": "Budget constraint",
        "Household.foc.C": "Consumption Euler equation",
    }


class TestDerivativeConditions:
    def test_an_unannotated_first_order_condition_is_not_captioned(self):
        """Only an author captions an equation, exactly as for every other kind of row."""
        model = load_and_cache_example("RBC")

        assert model.equation_label("Household.foc.C") is None
        assert model.equation_label("Firm.foc.L") is None

    def test_an_uncaptioned_condition_prints_the_derivative_it_came_from(self):
        """The prefix is what tells a reader which control the condition belongs to, now that no caption does."""
        model = load_and_cache_example("RBC")

        latex = model.table("equations").to_latex()

        assert r"\frac{\partial \mathcal{L}}{\partial C_{t}} = 0 &\implies" in latex

    def test_the_control_carries_the_time_index_it_was_declared_with(self):
        """``K[-1]`` is a control of the firm, and a derivative with respect to ``K_t`` would be the wrong one."""
        model = load_and_cache_example("RBC_two_household")

        latex = model.table("equations").to_latex()

        assert r"\partial K_{t-1}} = 0" in latex

    def test_a_declared_latex_name_renders_in_the_derivative(self, tmp_path):
        """The prefix is built from the same symbol the equations are, so an override has to reach both."""
        source = (TEST_GCNS / "open_rbc.gcn").read_text()
        declared = source.replace(
            "    C[] { positive = True; };", r'    C[] { positive = True; latex = "\mathcal{C}"; };', 1
        )
        assert declared != source, "open_rbc.gcn changed; update the replacement"
        path = tmp_path / "declared.gcn"
        path.write_text(declared)

        latex = model_from_gcn(path, verbose=False).table("equations").to_latex()

        assert r"\partial {\mathcal{C}}_{t}} = 0" in latex

    def test_an_authored_caption_keeps_its_tag_instead_of_the_derivative(self, labelled_model_path):
        """A caption and a derivative in the same row would say the same thing twice."""
        model = model_from_gcn(labelled_model_path, verbose=False)

        latex = model.table("equations").to_latex()

        assert r"\tag{\text{Consumption Euler equation}}" in latex
        assert r"\partial C_{t}} = 0" not in latex

    def test_an_equation_the_author_wrote_gets_neither(self):
        """An authored equation has source text to label, so inventing a caption or a derivative would be guessing."""
        model = model_from_gcn(TEST_GCNS / "open_rbc.gcn", verbose=False)

        row = next(r for r in model.table("equations").rows if r.equation_id == "Household.constraints.0")

        assert row.label is None
        assert row.foc_control is None
