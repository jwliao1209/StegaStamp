import os
import numpy as np


def generate_fingerprints(fingerprint_size=128):
    return np.random.randint(0, high=255, size=fingerprint_size, dtype=int)


def generate_dataset():
    file_path = os.path.join("dataset")
    return dict(file_path=file_path, image=file_path, fingerprints=generate_fingerprints(128))


if __name__ == "__main__":
    file_list = os.listdir("dataset/celeba/img_align_celeba")
    # generate_dataset()
    print(file_list)
