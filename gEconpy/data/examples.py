from pathlib import Path


def get_example_gcn(name: str) -> Path:
    """
    Return the path of one of the example GCN files bundled with gEconpy.

    Parameters
    ----------
    name : str
        File name of the example model without the ``.gcn`` extension, for example ``"RBC"`` or
        ``"New_Keynesian"``.

    Returns
    -------
    path : Path
        Path to the ``.gcn`` file.

    Examples
    --------
    Load the RBC example model:

    .. code-block:: python

        from gEconpy import model_from_gcn
        from gEconpy.data import get_example_gcn

        model = model_from_gcn(get_example_gcn("RBC"), verbose=False)
    """
    gcn_dir = Path(__file__).parent / "GCN Files"
    return gcn_dir / f"{name}.gcn"
