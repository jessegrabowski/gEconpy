import re
from typing import Dict, Tuple

from gEconpy.exceptions.exceptions import (
    DistributionParsingError,
    MissingParameterValueException,
)
from gEconpy.parser.constants import (
    BLOCK_END_TOKEN,
    CALIBRATING_EQ_TOKEN,
    EXPECTATION_TOKEN,
    LAG_TOKEN,
    LEAD_TOKEN,
    OPERATORS,
    SS_TOKEN,
)


def remove_extra_spaces(text: str) -> str:
    """
    Remove multiple spaces and return the resulting string.

    Parameters
    ----------
    text : str
        A string to remove extra spaces from.

    Returns
    -------
    str
        A string with multiple spaces replaced by single spaces.

    Notes
    -----
    This function is used to replace multiple spaces ('   ') with single spaces (' ') that result from removing special
    characters during preprocessing (tabs, newlines, etc).
    """

    out_text = re.sub(" +", " ", text)
    return out_text.strip()


def remove_newlines_and_tabs(text: str) -> str:
    """
    Remove newline and tab characters from a string.

    Parameters
    ----------
    text : str
        A string to remove newline and tab characters from.

    Returns
    -------
    str
        A string with newline and tab characters removed.
    """

    out_text = text.replace("\n", " ")
    out_text = out_text.replace("\t", " ")
    out_text = remove_extra_spaces(out_text)

    return out_text


def remove_comments(text: str) -> str:
    """
    Remove comments from a string.

    Parameters
    ----------
    text : str
        A string representing a GCN model file with comments to remove.

    Returns
    -------
    str
        The input string with comments removed.

    Notes
    -----
    The GCN model language allows for comments using the # prefix, either on their own line or following an equation
    in-line. This function strips these comments out.
    """

    lines = text.split("\n")
    lines = [line.strip() for line in lines if len(line.strip()) > 0]
    output = []

    for line in lines:
        if line[0] != "#":
            if "#" in line:
                hash_idx = line.find("#")
                output.append(line[:hash_idx])
            else:
                output.append(line)

    return "\n".join(output)


def extract_distributions(text: str) -> Tuple[str, Dict[str, str]]:
    """
    Extract prior distributions from a GCN model file and return the "clean" model file and a dictionary of the form
    parameter:distribution.

    Parameters
    ----------
    text : str
        Raw model file return by function `load_gcn`.

    Returns
    -------
    str
        Model file with prior distribution information removed.
    Dict[str, str]
        Dictionary of the form parameter:distribution.

    Examples
    --------
    .. code-block:: py
         extract_distributions('alpha ~ Beta(mean=0.5, sd=0.1) = 0.55;')
        # ('alpha = 0.55;', {"alpha": "Beta(mean=0.5, sd=0.1)"})
    """

    lines = text.split("\n")
    output = []
    prior_dict = {}

    for line in lines:
        if "~" in line:
            param_name, other = line.split("~")

            # This is a shock definition, there won't be an "=" after the distribution
            if "[]" in param_name:

                dist_info = other.strip().replace(";", "")
                new_line = param_name.strip() + ";"

            # This is a parameter definition, but it might be missing a default value
            else:
                # Extract the distribution declaration
                *dist_info, param_value = other.split("=")
                dist_info = "=".join(dist_info)

                # This should only happen in the user didn't give a default value
                if ")" in param_value:
                    raise MissingParameterValueException(param_name)

                new_line = f"{param_name.strip()} = {param_value.strip()}"
            output.append(new_line)
            prior_dict[param_name.strip()] = dist_info.strip()
        else:
            if line.count("=") > 1:
                raise DistributionParsingError(line)

            output.append(line)

    output = "\n".join(output).strip()

    return output, prior_dict


def add_spaces_around_expectations(text: str) -> str:
    """
    Insert spaces around expectation tokens and the square brackets that define what is in the expectation.

    Parameters
    ----------
    text : str
        A raw model file as plaintext.

    Returns
    -------
    str
        A raw model as plaintext.

    Examples
    --------
    .. code-block:: python
        add_spaces_around_expectations("E[][u[] + beta * U[1]];")
        # Output: "E[] [ u[] + beta * U[1] ];"

    """

    # Only add white space to the left of the expectation token so we can look for [[ when splitting the square brackets
    out_text = re.sub(f"(\\b{re.escape(EXPECTATION_TOKEN)})", r" \g<0>", text)
    out_text = re.sub(r"(?<=\])[\[\]]|(?<!(\[|\]))\]", r" \g<0> ", out_text)

    return out_text


def repair_special_tokens(text: str) -> str:
    """
    Repair the lag, lead, ss, and calibrating_eq tokens needed to mark variables in a later processing step.

    Parameters
    ----------
    text : str
        A raw model file as plaintext.

    Returns
    -------
    str
        A raw model file as plaintext.

    Examples
    --------
    .. code-block:: python
        repair_special_tokens(" u[ -1 ]")
        # Output: "u[-1]"
    """

    out_text = re.sub(r"\[ *\- *1 *\]", LAG_TOKEN, text)
    out_text = re.sub(r"\[ *1 *\]", LEAD_TOKEN, out_text)
    out_text = re.sub(r"\[ *ss * \]", SS_TOKEN, out_text)
    out_text = re.sub(r" * - * > *", f" {CALIBRATING_EQ_TOKEN} ", out_text)
    out_text = re.sub("} ;", BLOCK_END_TOKEN, out_text)

    return out_text


def add_spaces_around_operators(text: str) -> str:
    """
    Insert spaces around math operators.

    Parameters
    ----------
    text : str
        Raw text model file, including special model syntax and mathematical equations.

    Returns
    -------
    text : str
        Same text, with spaces added around math operators.

    Notes
    -----
    To convert the model into a series of tokens that can be processed, space is added between math operators,
    defined in the OPERATORS global as '+-*/^=();:'. Mathematical "sentences" should then be of the form
    {Y[] = a + X[] ; };, which can be parsed in a later step.

    Several errors are introduced by simply adding spaces around operators: lagged variables tokens, written as X[-1],
    are mutilated to X[ - 1], lead tokens are mutilated to X[  1  ], steady_state tokens become [ ss],
    the calibrating equation assignment operator "->" becomes " - >", and the "end of block" token, "};"
    is mutilated to "} ;". These errors are corrected by the repair_special_tokens function.
    """

    out_text = re.sub(f"[{OPERATORS}]", r" \g<0> ", text)
    out_text = add_spaces_around_expectations(out_text)
    out_text = remove_extra_spaces(out_text)
    out_text = repair_special_tokens(out_text)

    return out_text


def delete_block(text: str, block_name: str) -> str:
    """
    Delete a block from a model text file.

    Parameters
    ----------
    text: str
        Raw model file as text.
    block_name: str
        Block name to delete.

    Returns
    -------
    str
        Model file without the selected block.

    Special blocks "options" and "tryreduce" follow a special format. These blocks are pre-processed separately from
    the rest of the model blocks. This is a helper function to delete these blocks from the raw text after they have
    been processed, making assumptions about structure of the remaining blocks uniform.

    Examples
    --------
    >>> delete_block("options { }; tryreduce { };", "tryreduce")
    'options { };'
    """

    if block_name not in text:
        return text
    elif block_name == "assumptions":
        return re.sub(block_name + " {.*?}; };", "", text).strip()
    else:
        return re.sub(block_name + " {.*?};", "", text).strip()
