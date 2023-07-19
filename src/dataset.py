import os
import copy
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader

from src.transforms import get_transform
from src.utils import load_json


class FingerprintsDataset(Dataset):
    def __init__(self, data_list, transform=None):
        super(FingerprintsDataset, self).__init__()
        self.data_list = data_list
        self.transform = transform

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, i):
        data = copy.deepcopy(self.data_list[i])
        return self.transform(data) if self.transform is not None else data


class LitDataLoader(pl.LightningDataModule):
    def __init__(self, batch_size, num_workers):
        super(LitDataLoader, self).__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
    
    def setup(self, stage=None):
        self.data_list = load_json(
            os.path.join("dataset", "celeba", "data.json")
        )
        return
    
    def train_dataloader(self):
        train_set = FingerprintsDataset(
            data_list=self.data_list["train"],
            transform=get_transform(),
        )
        train_loader = DataLoader(
            train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
        return train_loader

    def val_dataloader(self):
        valid_set = FingerprintsDataset(
            data_list=self.data_list["valid"],
            transform=get_transform(),
        )
        valid_loader = DataLoader(
            valid_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
        return valid_loader

    def test_dataloader(self):
        test_set = FingerprintsDataset(
            data_list=self.data_list["test"],
            transform=get_transform(),
        )
        test_loader = DataLoader(
            test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
        return test_loader
