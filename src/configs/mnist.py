import os
from configs.base import Config as BaseConfig


class Config(BaseConfig):
    # Base
    def __init__(self, **kwargs):
        super(Config, self).__init__(**kwargs)
        self.name = "BaseFederatedLearning"
        self.add_args()
        for key, value in kwargs.items():
            setattr(self, key, value)

    def add_args(self, **kwargs):

        self.client_do_evaluation: bool = False
        self.model_type: str = "SimpleCNN_MNIST"

        self.embedding_channel_in: int = 64
        self.embedding_channel_out: int = 32
        self.embedding_linear_in: int = 512
        self.embedding_linear_out: int = 512

        self.data_root: str = "working/dataset/MNIST"
        os.makedirs(self.data_root, exist_ok=True)

        self.data_type: str = "MNISTVerticalDataset"

        self.name = self.data_type + self.name
        for key, value in kwargs.items():
            setattr(self, key, value)
