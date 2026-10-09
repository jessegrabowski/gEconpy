from tests._resources.cache_compiled_models import load_and_cache_model


def test_spelling_out_a_default_still_hits_the_cached_model():
    """A caller who passes a default explicitly must not get a rebuild of a model already in the cache."""
    bare = load_and_cache_model("one_block_1.gcn")

    assert load_and_cache_model("one_block_1.gcn", on_unused_parameters="raise") is bare
    assert load_and_cache_model("one_block_1.gcn", infer_steady_state=True) is bare


def test_an_argument_that_changes_the_model_gets_its_own_entry():
    """Normalizing the key must not go so far as to collapse builds that differ."""
    inferred = load_and_cache_model("one_block_1.gcn")
    not_inferred = load_and_cache_model("one_block_1.gcn", infer_steady_state=False)

    assert inferred is not not_inferred
