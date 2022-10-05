import numpy as np
import json


def save_to_json(data, path):
    with open(path, "w") as json_file:
        dumped = {state: qvalue.tolist() for state, qvalue in data.items()}
        json.dump(dumped, json_file, indent=3)


def read_from_json(path):
    with open(path, "r") as json_file:
        content = json.load(json_file)
    data = {state: np.array(qvalue) for state, qvalue in content.items()}
    return data
