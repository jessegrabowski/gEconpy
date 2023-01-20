def load_gcn(gcn_path: str) -> str:
    """
    Loads a model file as raw text.

    Parameters
    ----------
    gcn_path : str
        File path to model file (GCN file).

    Returns
    -------
    str
        Raw-text of the model file.
    """

    with open(gcn_path, encoding="utf-8") as file:
        gcn_raw = file.read()
    return gcn_raw
