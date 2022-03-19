from gEcon.parser.constants import BLOCK_COMPONENTS
from typing import Optional
from gEcon.exceptions.exceptions import InvalidComponentNameException


def block_is_empty(block: str) -> bool:
    """
    :param block: str, raw text of a model block
    :return: bool

    Check whether a model block is empty, i.e. contains no flags, variables, or equations.
    """
    return block.strip() == '{ };'


def validate_key(key: str, block_name: str) -> None:
    """
    :param block_name: str, optional, the name of the block
    :param key: str, a block sub-component name
    :return: None

    The R implementation of gEcon only allows the names in BLOCK_COMPONENTS to be used inside model blocks.
    This function checks that a component name matches something in that list, and raises an error if not.

    TODO: Allow arbitrary component names? Is there any need to?
    """
    try:
        BLOCK_COMPONENTS(key.upper())
    except ValueError as e:
        valid_names = ', '.join([s.name.lower() for s in list(BLOCK_COMPONENTS)])
        error = f'Valid sub-block names are: {valid_names}\n'
        error += f'Found: {key} in block {block_name}'

        raise InvalidComponentNameException(component_name=key, block_name=block_name, message=error)
