import numpy as np
import pytest

from gEconpy import model_from_gcn
from gEconpy.parser.loader import load_gcn_file, load_gcn_string
from tests._resources.cache_compiled_models import load_and_cache_model
from tests.conftest import TEST_GCNS

SOURCE = """
symbols
{
    C[] { bounds = (0, None); };
    L[] { bounds = (0, None); };
    A[] { bounds = (0, None); };
    beta { bounds = (0, 1); };
};

block HOUSEHOLD
{
    controls { C[], L[]; };
    objective { U[] = log(C[]) - L[] + beta * E[][U[1]]; };
    constraints { C[] = A[] * L[] : lambda[]; };
    identities { log(A[]) = 0.9 * log(A[-1]); };
};
"""


def test_ids_name_the_block_and_component_they_came_from():
    block = load_gcn_string(SOURCE).block_dict["HOUSEHOLD"]

    assert block.system_equation_ids == [
        "HOUSEHOLD.identities.0",
        "HOUSEHOLD.constraints.0",
        "HOUSEHOLD.objective",
        "HOUSEHOLD.foc.C",
        "HOUSEHOLD.foc.L",
    ]


@pytest.mark.parametrize("gcn_file", ["open_rbc.gcn", "one_block_2.gcn", "full_nk.gcn", "rbc_empty_lead_column.gcn"])
def test_every_equation_keeps_a_unique_id_through_the_pipeline(gcn_file):
    """A filter that drops an equation has to drop its id, or every id after the hole names the wrong equation."""
    primitives = load_gcn_file(TEST_GCNS / gcn_file)

    assert len(primitives.equation_ids) == len(primitives.equations)
    assert len(set(primitives.equation_ids)) == len(primitives.equation_ids)


def test_multiplier_elimination_drops_the_id_of_the_equation_it_removes():
    """Eliminating a generated multiplier reduces the investment FOC to zero, and its id must go with it."""
    unsimplified = load_gcn_file(TEST_GCNS / "one_block_2.gcn", simplify_blocks=False).block_dict["HOUSEHOLD"]
    simplified = load_gcn_file(TEST_GCNS / "one_block_2.gcn", simplify_blocks=True).block_dict["HOUSEHOLD"]

    dropped = set(unsimplified.system_equation_ids) - set(simplified.system_equation_ids)

    assert dropped
    assert len(simplified.system_equation_ids) == len(simplified.system_equations)
    assert simplified.system_equation_ids == [
        eq_id for eq_id in unsimplified.system_equation_ids if eq_id not in dropped
    ]


def test_ids_index_the_steady_state_system_too():
    model = load_and_cache_model("open_rbc.gcn")

    assert len(model._equation_ids) == len(model._equations)
    assert len(model._equation_ids) == len(model._steady_state_equations)


def test_a_failed_steady_state_names_the_worst_equation():
    model = load_and_cache_model("open_rbc.gcn")
    residual = np.zeros(len(model._equation_ids))
    residual[2] = -4.2e-3
    residual[3] = 1.1e-3

    note = model._worst_residual_note(residual)

    assert model._equation_ids[2] in note
    assert "4.200e-03" in note


def test_the_note_is_empty_when_the_residual_does_not_match_the_ids():
    model = load_and_cache_model("open_rbc.gcn")

    assert model._worst_residual_note(np.zeros(len(model._equation_ids) + 1)) == ""


def test_the_parsed_file_reaches_the_model():
    model = model_from_gcn(TEST_GCNS / "open_rbc.gcn", verbose=False)

    assert model._source_ast.block_names() == [
        "STEADY_STATE",
        "HOUSEHOLD",
        "TECHNOLOGY_SHOCKS",
        "EQULIBRIUM",
    ]
