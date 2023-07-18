import os
import json
import glob
import random
import numpy as np


def generate_fingerprint(fingerprint_size=128):
    return np.random.randint(0, high=255, size=fingerprint_size, dtype=int).tolist()


def generate_data_dict(file_path):
    return dict(
        file_path=file_path,
        image=file_path,
        fingerprint=generate_fingerprint(128)
    )


def generate_dataset(file_list):
    return [generate_data_dict(f) for f in file_list]


def save_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as fp:
        json.dump(obj, fp, indent=4)
    return


def split_dataset(json_data, split_ratio=0.75):
    n = len(json_data)
    split_num = int(n * split_ratio)
    json_data = json_data.copy()
    random.shuffle(json_data)
    return json_data[:split_num], json_data[split_num:]


if __name__ == "__main__":
    file_list = glob.glob("dataset/celeba/img_align_celeba/*.jpg")
    json_data = generate_dataset(file_list)
    train_set, test_set = split_dataset(json_data, split_ratio=0.9)
    train_set, valid_set = split_dataset(train_set, split_ratio=0.8)
    json_dataset = dict(train=train_set, valid=valid_set, test=test_set)
    save_json(os.path.join("dataset", "celeba", "data.json"), json_dataset)
