import numpy as np
import torchvision.transforms as transforms

from torchvision.datasets import CIFAR10

from .base import BaseDataset, BaseHorizontalDataset, BaseVerticalDataset
from configs.base import Config


class Cifar10Dataset(BaseDataset):
    def __init__(
        self,
        cfg: Config,
    ):
        """Classification dataset
        args:
            root_path (str): path to the extracted folder of cifar 10.
        """
        super(Cifar10Dataset, self).__init__(cfg)
        self.classes = list(range(10))

        trainset = CIFAR10(
            root=cfg.data_root, train=True, download=True, transform=None
        )
        testset = CIFAR10(
            root=cfg.data_root, train=False, download=True, transform=None
        )

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]
                ),
            ]
        )

        self._init_torchvision_dataset(trainset, testset)


class Cifar10HorizontalDataset(Cifar10Dataset, BaseHorizontalDataset):
    pass


class Cifar10VerticalDataset(Cifar10HorizontalDataset, BaseVerticalDataset):
    pass
