import pytest
from pathlib import Path
from gEconpy.parser.file_loaders import load_gcn


ROOT = Path(__file__).parent.parent.absolute()


@pytest.fixture
def model():
    return load_gcn(ROOT / "_resources" / "test_gcns" / "one_block_1_dist.gcn")
