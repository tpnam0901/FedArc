import torchvision.transforms as transforms
from torchvision.datasets import CIFAR100
from .base import BaseDataset, BaseHorizontalDataset, BaseVerticalDataset
from configs.base import Config


class Cifar100Dataset(BaseDataset):
    def __init__(
        self,
        cfg: Config,
    ):
        """Classification dataset
        args:
            root_path (str): path to the extracted folder of cifar 10.
        """
        super(Cifar100Dataset, self).__init__(cfg)
        self.classes = list(range(100))

        trainset = CIFAR100(
            root=cfg.data_root, train=True, download=True, transform=None
        )
        testset = CIFAR100(
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


class Cifar100HorizontalDataset(Cifar100Dataset, BaseHorizontalDataset):
    pass


class Cifar100VerticalDataset(Cifar100HorizontalDataset, BaseVerticalDataset):
    pass
