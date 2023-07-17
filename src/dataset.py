import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule


class FingerprintsDataset(Dataet):
    def __init__(self):
        super(FingerprintsDataset, self).__init__()

    def __len__(self):
        return len()
    
    def __getitem__(self, i):
        return



class LitDataLoader(LightningDataModule):
    def __init__(self, batch_size, num_workers):
        super(LitDataLoader, self).__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
    
    def setup(self, stage=None):
        return
    
    def train_dataloader(self):
        train_trans = transforms.Compose(
            [
                transforms.CenterCrop(148),
                transforms.Resize(128),
                transforms.ToTensor(),
            ]
        )
        train_set = torchvision.datasets.CelebA(
            "./dataset",
            split="train",
            download=True,
            transform=train_trans,
        )
        train_loader = DataLoader(
            train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
        return train_loader

    def val_dataloader(self):
        valid_trans = transforms.Compose(
            [
                transforms.CenterCrop(148),
                transforms.Resize(128),
                transforms.ToTensor(),
            ]
        )
        valid_set = torchvision.datasets.CelebA(
            "./dataset",
            split="valid",
            download=True,
            transform=valid_trans,        
        )
        valid_loader = DataLoader(
            valid_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
        return valid_loader

    def test_dataloader(self):
        test_trans = transforms.Compose(
            [
                transforms.CenterCrop(148),
                transforms.Resize(128),
                transforms.ToTensor(),
            ]
        )
        test_set = torchvision.datasets.CelebA(
            "./dataset",
            split="test",
            download=True,
            transform=test_trans,
        )
        test_loader = DataLoader(
            test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
        return test_loader
