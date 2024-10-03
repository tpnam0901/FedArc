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

        self.batch_size = 32
        self.num_clients = 50
        self.shard_per_client = 4
        self.client_do_evaluation: bool = False

        self.model_type: str = "ResNet18"
        self.embedding_channel_in: int = 512
        self.embedding_channel_out: int = 512
        self.embedding_linear_in: int = 512
        self.embedding_linear_out: int = 512  # embedding_size

        self.data_type: str = "Cifar100VerticalDataset"
        self.data_root: str = "working/dataset/CIFAR/cifar-100-python"

        self.name = self.data_type + self.name
        for key, value in kwargs.items():
            setattr(self, key, value)
