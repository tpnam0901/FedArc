import os
import math
import torch
import numpy as np
import random
import time
import csv

random.seed(1996)

from scipy.special import softmax
import torchvision.transforms as transforms

from typing import Tuple
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from configs.base import Config


class BaseDataset(Dataset):
    def __init__(
        self,
        cfg: Config,
    ):
        """Classification dataset
        args:
            root_path (str): path to the extracted folder of cifar 10.
        """
        super(BaseDataset, self).__init__()
        self.classes = []

        self.train_X = []
        self.train_y = []

        self.test_X = []
        self.test_y = []

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

        self.val_X = []
        self.val_y = []
        self.raw_train_X = self.train_X
        self.raw_train_y = self.train_y

        self.X = np.asarray(self.train_X)
        self.y = np.asarray(self.train_y)

        self.cfg = cfg

    def _init_torchvision_dataset(self, trainset, testset):
        self.train_X = []
        self.train_y = []
        for x, y in trainset:
            self.train_X.append(np.array(x))
            self.train_y.append(y)

        self.test_X = []
        self.test_y = []
        for x, y in testset:
            self.test_X.append(np.array(x))
            self.test_y.append(y)

        self.val_X = []
        self.val_y = []
        self.raw_train_X = self.train_X
        self.raw_train_y = self.train_y

        self.X = np.asarray(self.train_X)
        self.y = np.asarray(self.train_y)

    def idx2cls(self, cls_id: int):
        return self.classes[cls_id]

    def train_val_split(self, val_size: float, seed: int = 1996):
        self.train_X, self.val_X, self.train_y, self.val_y = train_test_split(
            self.raw_train_X,
            self.raw_train_y,
            test_size=val_size,
            random_state=seed,
            shuffle=True,
            stratify=self.raw_train_y,
        )

    def train(self):
        self.X = self.train_X
        self.y = self.train_y

    def val(self):
        self.X = self.val_X
        self.y = self.val_y

    def test(self):
        self.X = self.test_X
        self.y = self.test_y

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.transform(self.X[index].astype(np.uint8))
        label = torch.tensor(self.y[index])
        return sample, label

    def __len__(self):
        return len(self.X)


class BaseHorizontalDataset(BaseDataset):
    def _sort_data(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        idx = np.argsort(y)
        return X[idx], y[idx]

    def fl_split(self):
        print("Running FL split for Horizontal Dataset")
        time.sleep(2)
        self.client_dataset = {}
        raw_train_X = np.asarray(self.raw_train_X)
        raw_train_y = np.asarray(self.raw_train_y)

        if self.cfg.client_same_label_length:
            skf = StratifiedKFold(n_splits=self.cfg.num_clients)
            for client_id, (_, test_index) in enumerate(
                skf.split(raw_train_X, raw_train_y)
            ):
                self.client_dataset.update(
                    {
                        client_id: {
                            "train_X": raw_train_X[list(test_index)],
                            "train_y": raw_train_y[list(test_index)],
                        }
                    }
                )
        else:
            # Get the client weight base on the number of samples
            client_weights = list(softmax(np.random.randn(self.cfg.num_clients)))
            client_id = 0
            current_train_X = raw_train_X
            current_train_y = raw_train_y

            # Log dataset len per client
            with open(
                os.path.join(self.cfg.checkpoint_dir, "clients_datainfo.csv"),
                "w",
                newline="",
            ) as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["client_id"] + self.classes)
                while len(client_weights) > 0:
                    client_random_weight = client_weights.pop()
                    num_folds = int(1 / client_random_weight)
                    if num_folds < 2:
                        num_folds = 2
                    if len(client_weights) > 0:
                        self.client_dataset.update(
                            {
                                client_id: {
                                    "train_X": current_train_X,
                                    "train_y": current_train_y,
                                }
                            }
                        )
                        classes_samples = [0] * len(self.classes)
                        for y in current_train_y:
                            classes_samples[y] += 1
                        writer.writerow([client_id] + classes_samples)
                        break

                    skf = StratifiedKFold(n_splits=num_folds)
                    isAssign = False
                    for train_index, test_index in skf.split(
                        current_train_X, current_train_y
                    ):
                        if not isAssign:
                            isAssign = True
                            # Update the client dataset
                            self.client_dataset.update(
                                {
                                    client_id: {
                                        "train_X": current_train_X[test_index],
                                        "train_y": current_train_y[test_index],
                                    }
                                }
                            )
                            classes_samples = [0] * len(self.classes)
                            for y in current_train_y:
                                classes_samples[y] += 1
                            writer.writerow([client_id] + classes_samples)
                            # Update the current train dataset
                            current_train_X = current_train_X[train_index]
                            current_train_y = current_train_y[train_index]
                    client_id += 1
                    print(client_id)
                    # Calculate the new weights for the remaining dataset
                    client_weights = list(softmax(client_weights))
                    print(len(client_weights))

    def get_dataset(self, client_id: int):
        client_dataset = getattr(
            self,
            "client_dataset",
            {0: {"train_X": self.train_X, "train_y": self.train_y}},
        )
        return client_dataset[client_id]

    def set_dataset(self, client_id: int):
        client_dataset = getattr(
            self,
            "client_dataset",
            {0: {"train_X": self.train_X, "train_y": self.train_y}},
        )
        self.X = self.train_X = client_dataset[client_id]["train_X"]
        self.y = self.train_y = client_dataset[client_id]["train_y"]


class BaseVerticalDataset(BaseHorizontalDataset):
    def fl_split(self):
        print("Running FL split for Vertical Dataset")
        time.sleep(2)
        self.client_dataset = {}

        self.train_X, self.train_y = self._sort_data(self.raw_train_X, self.raw_train_y)

        # Get the number of shards base on the number of shards per client
        num_shards = self.cfg.num_clients * self.cfg.shard_per_client
        shards_weights = softmax(
            [1] * num_shards
            if self.cfg.client_same_label_length
            else np.random.randn(num_shards)
        )
        previous_index = 0

        shard_data = []
        for shard_id in range(num_shards):
            shard_num_samples = math.floor(shards_weights[shard_id] * len(self.train_X))
            shard_data.append(
                (
                    self.train_X[previous_index : previous_index + shard_num_samples],
                    self.train_y[previous_index : previous_index + shard_num_samples],
                )
            )
            previous_index += shard_num_samples

        # shuffle shard data
        random.shuffle(shard_data)

        share_index = 0
        for client_id in range(self.cfg.num_clients):
            train_X = []
            train_y = []
            for _ in range(self.cfg.shard_per_client):
                train_X.extend(shard_data[share_index][0])
                train_y.extend(shard_data[share_index][1])
                share_index += 1
            self.client_dataset.update(
                {
                    client_id: {
                        "train_X": train_X,
                        "train_y": train_y,
                    }
                }
            )
