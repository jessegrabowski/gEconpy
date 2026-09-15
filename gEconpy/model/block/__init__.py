# The form modules register their Block subclasses on import, and importing any submodule runs this __init__ first,
# so the registry is populated before dispatch_block can be called.
from gEconpy.model.block import (
    ces,  # noqa: F401
    cobb_douglas,  # noqa: F401
)
from gEconpy.model.block.basic import Block
from gEconpy.model.block.registry import dispatch_block, register_block

__all__ = ["Block", "dispatch_block", "register_block"]
