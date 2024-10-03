from configs.cifar100 import Config as BaseConfig


class Config(BaseConfig):
    def __init__(self, **kwargs):
        super(Config, self).__init__(**kwargs)
        super().add_args()
        self.name = "CentralizedLearning"
        self.add_args()
        for key, value in kwargs.items():
            setattr(self, key, value)

    def add_args(self, **kwargs):
        self.batch_size = 100
        self.num_clients = 1
        self.num_epochs = self.num_epochs * self.num_rounds
        self.num_rounds = 1
        self.client_do_evaluation = True
        self.client_best_weight = True

        self.name = self.data_type + self.name
        for key, value in kwargs.items():
            setattr(self, key, value)
