import json


def load_json(path):
    with open(path, 'r') as fp:
        obj = json.load(fp)
    return obj
