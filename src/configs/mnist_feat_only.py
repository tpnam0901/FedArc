from configs.mnist import Config as BaseConfig


class Config(BaseConfig):
    # Base
    def __init__(self, **kwargs):
        super(Config, self).__init__(**kwargs)
        super().add_args()
        self.name = "FederatedLearningFeatureOnly"
        self.add_args()
        for key, value in kwargs.items():
            setattr(self, key, value)

    def add_args(self, **kwargs):
        self.loss_type: str = "CrossEntropyLoss_CombinedMarginLoss"

        self.name = self.data_type + self.name
        for key, value in kwargs.items():
            setattr(self, key, value)
