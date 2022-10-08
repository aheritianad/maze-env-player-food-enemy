import numpy as np
import json
from typing import *


def save_to_json(data: dict[str:np.array], path: str) -> None:
    """Function which saves a python dictionary object into a json file.

    Args:
        data (dict[str:np.array]): python dictionary.
        path (str): path for the json file.
    """
    with open(path, "w") as json_file:
        dumped = {state: qvalue.tolist() for state, qvalue in data.items()}
        json.dump(dumped, json_file, indent=3)


def read_from_json(path: str) -> dict[str:np.array]:
    """Function which reads a json file and return a python dictionary

    Args:
        path (str): path where the json file is located.

    Returns:
        dict[str:np.array]: python dictionary version of the json file.
    """
    with open(path, "r") as json_file:
        content = json.load(json_file)
    data = {state: np.array(qvalue) for state, qvalue in content.items()}
    return data
