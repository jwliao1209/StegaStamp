import torch
from torchvision.io import read_image
from torchvision.transforms import Compose, Resize, Normalize

from monai.transforms import ToTensord


class BaseTransform(object):
    def __init__(self, keys, **kwargs):
        self.keys = keys
        self._parse_var(**kwargs)

    def __call__(self, data, **kwargs):
        for key in self.keys:
            if key in data:
                data[key] = self._process(data[key], **kwargs)
            else:
                raise KeyError(f"{key} is not a key in data.")
        return data

    def _parse_var(self, **kwargs):
        pass


class ReadImaged(BaseTransform):
    def __init__(self, keys, **kwargs):
        super(ReadImaged, self).__init__(keys, **kwargs)

    def _process(self, single_data, **kwargs):
        single_data = read_image(single_data)
        return single_data


class ResizeImaged(BaseTransform):
    def __init__(self, keys, **kwargs):
        super(ResizeImaged, self).__init__(keys, **kwargs)

    def _parse_var(self, **kwargs):
        self.resize = Resize(kwargs.get('size'))

    def _process(self, single_data, **kwargs):
        return self.resize(single_data)


class ConvertToFloat(BaseTransform):
    def __init__(self, keys, **kwargs):
        super(ConvertToFloat, self).__init__(keys, **kwargs)
    
    def _process(self, single_data, **kwargs):
        return single_data.float()


class MinMaxNormalizeImaged(BaseTransform):
    def __init__(self, keys, **kwargs):
        super(MinMaxNormalizeImaged, self).__init__(keys, **kwargs)

    def _process(self, single_data, **kwargs):
        maximum = torch.max(single_data)
        mininum = torch.min(single_data)
        return (single_data - mininum) / (maximum - mininum)


class StandardNormalizeImaged(BaseTransform):
    def __init__(self, keys, **kwargs):
        super(StandardNormalizeImaged, self).__init__(keys, **kwargs)

    def _parse_var(self, **kwargs):
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        self.normal = Normalize(mean, std)

    def _process(self, single_data, **kwargs):
        return self.normal(single_data)


def get_transform():
    return Compose(
        [
            ReadImaged(keys=["image"]),
            ResizeImaged(keys=["image"], size=(128, 128)),
            ToTensord(keys=["image", "fingerprint"]),
            ConvertToFloat(keys=["image", "fingerprint"]),
            MinMaxNormalizeImaged(keys=["image"]),
            # StandardNormalizeImaged(keys=["image"]),
        ]
    )
