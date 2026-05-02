# Import the form modules so their @register_block decorators run on package load. Loading any submodule via
# ``from gEconpy.model.block.X import ...`` runs this __init__ first, so the registry is guaranteed populated before
# dispatch_block is ever called.
from gEconpy.model.block import cobb_douglas  # noqa: F401 — side-effect import for registration
from gEconpy.model.block.basic import Block
from gEconpy.model.block.registry import dispatch_block, register_block

__all__ = ["Block", "dispatch_block", "register_block"]
