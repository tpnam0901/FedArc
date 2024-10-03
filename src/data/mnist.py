from torchvision.datasets import MNIST
import torchvision.transforms as transforms

from .base import BaseDataset, BaseHorizontalDataset, BaseVerticalDataset
from configs.base import Config


class MNISTDataset(BaseDataset):
    def __init__(
        self,
        cfg: Config,
    ):
        """Classification dataset
        args:
            root_path (str): path to the extracted folder of cifar 10.
        """
        super(MNISTDataset, self).__init__(cfg)
        self.classes = list(range(10))

        trainset = MNIST(root=cfg.data_root, train=True, download=True, transform=None)
        testset = MNIST(root=cfg.data_root, train=False, download=True, transform=None)

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

        self._init_torchvision_dataset(trainset, testset)


class MNISTHorizontalDataset(MNISTDataset, BaseHorizontalDataset):
    pass


class MNISTVerticalDataset(MNISTHorizontalDataset, BaseVerticalDataset):
    pass
