import logging
import os
from abc import ABC, abstractmethod
from typing import List, Union
import importlib
import sys


class Base(ABC):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    @abstractmethod
    def show(self):
        pass

    @abstractmethod
    def save(self, save_folder: str):
        pass

    @abstractmethod
    def load(self, cfg_path: str):
        pass


class BaseConfig(Base):
    def __init__(self, **kwargs):
        super(BaseConfig, self).__init__(**kwargs)

    def show(self):
        for key, value in self.__dict__.items():
            logging.info(f"{key}: {value}")

    def save(self, save_folder: str):
        message = "\n"
        for k, v in sorted(vars(self).items()):
            message += f"{str(k):>30}: {str(v):<40}\n"

        os.makedirs(os.path.join(save_folder), exist_ok=True)
        out_cfg = os.path.join(save_folder, "cfg.log")
        with open(out_cfg, "w") as cfg_file:
            cfg_file.write(message)
            cfg_file.write("\n")

        logging.info(message)

    def load(self, cfg_path: str):
        def decode_value(value: str):
            value = value.strip()
            value_converted = None
            if "." in value and value.replace(".", "").isdigit():
                value_converted = float(value)
            elif value.isdigit():
                value_converted = int(value)
            elif "e" not in value and "-" in value:
                value_converted = str(value)
            elif value.replace("e", "").replace("-", "").isdigit():
                value_converted = float(value)
            elif value == "True":
                value_converted = True
            elif value == "False":
                value_converted = False
            elif (
                value.startswith("'")
                and value.endswith("'")
                or value.startswith('"')
                and value.endswith('"')
            ):
                value_converted = value[1:-1]
            elif value.startswith("(") or value.startswith("["):
                value_converted = []
                for temp in value.strip("(").strip(")").split(","):
                    value_converted.append(decode_value(temp))
                if value.startswith("("):
                    value_converted = tuple(value_converted)
            else:
                value_converted = value
            return value_converted

        with open(cfg_path, "r") as f:
            data = f.read().split("\n")
            # remove all empty strings
            data = list(filter(None, data))
            # convert to dict
            data_dict = {}
            for i in range(len(data)):
                key, value = (
                    data[i].split(":")[0].strip(),
                    data[i].split(":")[1].strip(),
                )
                if value.startswith("[") and value.endswith("]"):
                    value = value[1:-1].split(",")
                    value = [decode_value(x) for x in value]
                else:
                    value = decode_value(value)
                data_dict[key] = value
        for key, value in data_dict.items():
            setattr(self, key, value)


class Config(BaseConfig):
    # Base
    def __init__(self, **kwargs):
        super(Config, self).__init__(**kwargs)
        self.name = "BaseFederatedLearning"
        self.set_args()
        for key, value in kwargs.items():
            setattr(self, key, value)

    def set_args(self, **kwargs):
        # -------------------------- Training settings -------------------------- #
        self.num_epochs: int = 5
        self.batch_size: int = 50
        self.checkpoint_dir: str = "working/checkpoints"
        self.ckpt_save_fred: int = 5000
        self.datetime: str = ""

        # -------------------------- Federated settings -------------------------- #
        # Whether to allow same feature dataset or not
        self.num_rounds: int = 10000
        self.num_clients: int = 100
        self.num_client_prob: float = 0.1
        self.client_same_label_length: bool = True
        self.shard_per_client: int = 2
        self.client_do_evaluation: bool = True
        # Required client_do_evaluation set to True
        self.client_best_weight: bool = False
        # fed_avg, fed_moving_avg
        self.aggregate_type: str = "fed_avg"

        # CrossEntropyLoss, CrossEntropyLoss_CombinedMarginLoss, CrossEntropyLoss_CosineSimilarityLoss, CrossEntropyLoss_CombinedMarginLoss_CosineSimilarityLoss
        self.loss_type: str = "CrossEntropyLoss"
        self.loss_reduction: str = "mean"
        self.loss_lambda: float = 0.5
        # Feature loss
        self.margin_loss_scale: float = 64
        self.margin_loss_m1: float = 1.0  # SphereFace
        self.margin_loss_m2: float = 0.0  # CosFace
        self.margin_loss_m3: float = 0.4  # ArcFace
        # -------------------------- Optim settings -------------------------- #
        self.learning_rate: float = 0.001
        self.weight_decay: float = 0.00001
        self.momentum: float = 0.9
        self.lr_step_size: int = 50
        self.gamma: float = 0.1

        # -------------------------- Model settings -------------------------- #
        self.model_type: str = "EfficientNetB0"
        self.num_classes: int = -1
        self.embedding_channel_in: int = 320
        self.embedding_channel_out: int = 512
        self.embedding_linear_in: int = 1024
        self.embedding_linear_out: int = 512  # embedding_size

        # -------------------------- Dataset settings -------------------------- #
        self.data_root: str = "working/dataset/CIFAR/cifar-10-batches-py"
        # Cifar10Dataset, Cifar10HorizontalDataset, Cifar10VerticalDataset,
        # MNISTDataset, MNISTHorizontalDataset, MNISTVerticalDataset
        self.data_type: str = "Cifar10Dataset"
        self.num_workers: int = 4  # map to "workers"
        self.val_size: float = 0.2
        self.val_from_train: bool = False
        self.test_as_val: bool = True
        self.pin_memory: bool = True

        for key, value in kwargs.items():
            setattr(self, key, value)


def import_config(
    path: str,
):
    """Get arguments for training and evaluate
    Returns:
        cfg: ArgumentParser
    """
    # Import config from path
    spec = importlib.util.spec_from_file_location("config", path)
    config = importlib.util.module_from_spec(spec)
    sys.modules["config"] = config
    spec.loader.exec_module(config)
    cfg = config.Config()
    return cfg
