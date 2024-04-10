"""Writes etot.input file according to dictionary of etot.input file"""
from __future__ import annotations
from typing import Dict, Any, Union
from collections.abc import Iterable
from pathlib import Path


__author__ = "Hanyu Liu"
__email__ = "domainofbuaa@gmail.com"
__date__ = "2024-4-7"


def write_etot_input(directory: str, parameters: Dict[str, Any], header=None):
    etot_input_string: str = generate_etot_input_lines(parameters)
    with open(Path(directory) / "etot.input", "w") as etot_input:
        if header is not None:
            etot_input.write(header + "\n")
        etot_input.write(etot_input_string)


def generate_etot_input_lines(parameters: Union[str, Dict[str, Any]]):
    if isinstance(parameters, str):
        return parameters
    elif parameters is None:
        return ""
    else:
        etot_input_lines = []
        for item in parameters.items():
            etot_input_lines += list(generate_line(*item))
        # Adding a newline at the end of the file
        return "\n".join(etot_input_lines) + "\n"


def generate_line(key, value, num_spaces=0):
    indent = " " * num_spaces
    if key.upper() == "PARALLEL":
        yield indent + f"{value}"
    else:
        if isinstance(value, str):
            if value.find("\n") != -1:
                value = '"' + value + '"'
            yield indent + f"{key.upper()} = {value}"
        elif isinstance(value, dict):
            yield indent + f"{key.upper()} "
            for item in value.items():
                yield from generate_line(*item, num_spaces + 4)
            yield indent + "}"
        elif isinstance(value, Iterable):
            yield indent + f"{key.upper()} = {' '.join(str(x) for x in value)}"
        else:
            yield indent + f"{key.upper()} = {value}"
