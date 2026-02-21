from pathlib import Path


def get_example_gcn(name: str) -> Path:
    """
    Get the file path for an example model given its name.

    The name should correspond to a .gcn file in the "GCN Files" directory.

    Parameters
    ----------
    name : str
        The name of the example model to load. Must correspond to a .gcn file in the "GCN Files" directory.

    Returns
    -------
    model : Model
        The loaded model.
    """
    gcn_dir = Path(__file__).parent / "GCN Files"
    return gcn_dir / f"{name}.gcn"
