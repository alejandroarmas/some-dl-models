"""
Concrete IO class for a specific dataset
"""

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

import pickle
import random
from code.base_class.dataset import dataset, datasetConfig
from code.lib.notifier import DatasetNotifier
from typing import Optional

import numpy as np
import torch
from cerberus import Validator  # type: ignore
from torch.utils.data import Dataset


class LoadedDataset(Dataset):
    def __init__(self, X, y, transform=None, train=True, toByte=False):

        self.X = X
        self.y = y
        self.transform = transform
        self.train = train
        self.toByte = toByte

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):

        image = self.X[idx]
        label = self.y[idx]

        sample = {"image": image, "label": label}

        # convert tensor to bytes (if needed)
        if self.transform is not None:
            sample["image"] = (
                self.transform(sample["image"].byte())
                if self.transform and self.toByte
                else self.transform(sample["image"])
            )

        return sample


class PickleLoader(dataset):
    def __init__(self, config: datasetConfig, manager: Optional[DatasetNotifier] = None):
        super().__init__(config, manager)

    def load(self) -> dict:

        data = {}

        with open(f"{self.dataset_source_folder_path}{self.dataset_source_file_name}", "rb") as f:
            data = pickle.load(f)

        return data


class ValidatedPickleLoader(dataset):
    """
    This follows a proxy pattern to extend our dataset and validate our
    loaded data.

    Args:
        dataset (_type_): _description_

    Returns:
        _type_: _description_
    """

    real_loader: PickleLoader
    samples_to_verify: int

    def __init__(
        self,
        config: datasetConfig,
        manager: Optional[DatasetNotifier] = None,
        _samples_to_verify=40,
    ):
        super().__init__(config, manager)
        self.real_loader = PickleLoader(config, manager)
        self.samples_to_verify = _samples_to_verify
        self.device = config["device"]

    def load(self, deactivateReorder=False) -> dict:

        data = self.real_loader.load()

        print("training set size:", len(data["train"]), "testing set size:", len(data["test"]))

        image_schema = {
            "type": "list",
            "schema": {
                "type": "dict",
                "schema": {"image": {"type": "list"}, "label": {"type": "integer"}},
            },
        }
        schema = {dataset_type: image_schema for dataset_type in ("train", "test")}
        # this is used to give us assurance that the data loaded, matches the above schema
        v = Validator(schema)

        batch = {
            "train": random.sample(data["train"], self.samples_to_verify),
            "test": random.sample(data["test"], self.samples_to_verify),
        }

        v.validate(batch)

        # disables reordering of axes in dataset
        if deactivateReorder:
            return {
                "X_train": torch.FloatTensor(
                    np.array([example["image"] for example in data["train"]])
                ).to(self.device),
                "y_train": torch.LongTensor(
                    np.array([example["label"] for example in data["train"]])
                ).to(self.device),
                "X_test": torch.FloatTensor(
                    np.array([example["image"] for example in data["test"]])
                ).to(self.device),
                "y_test": torch.LongTensor(
                    np.array([example["label"] for example in data["test"]])
                ).to(self.device),
            }

        return {
            "X_train": torch.FloatTensor(
                np.array([np.moveaxis(example["image"], 2, 0) for example in data["train"]])
            ).to(self.device),
            "y_train": torch.LongTensor(
                np.array([example["label"] for example in data["train"]])
            ).to(self.device),
            "X_test": torch.FloatTensor(
                np.array([np.moveaxis(example["image"], 2, 0) for example in data["test"]])
            ).to(self.device),
            "y_test": torch.LongTensor(np.array([example["label"] for example in data["test"]])).to(
                self.device
            ),
        }
