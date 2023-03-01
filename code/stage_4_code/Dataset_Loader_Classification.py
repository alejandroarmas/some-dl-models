import os
from code.base_class.dataset import dataset, datasetConfig
from code.lib.notifier import DatasetNotifier
from typing import Optional

import numpy as np
from torch.utils.data import Dataset


class Classififcation_Dataset(Dataset):
    def __init__(self, X, y, base_dir: str):
        self.base_dir = base_dir
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        path = self.X[idx]
        label = self.y[idx]
        contents = ""
        with open(f"{self.base_dir}{path}") as f:
            contents = f.read()

        return contents, label


class Classification_Loader(dataset):
    def __init__(self, config: datasetConfig, manager: Optional[DatasetNotifier] = None):
        super().__init__(config, manager)
        self.train_pos_dir_path = f"{self.dataset_source_folder_path}/train/pos/"
        self.train_neg_dir_path = f"{self.dataset_source_folder_path}/train/neg/"
        self.test_pos_dir_path = f"{self.dataset_source_folder_path}/test/pos/"
        self.test_neg_dir_path = f"{self.dataset_source_folder_path}/test/neg/"

    # Loads Train and Test sets from disk, with X being filepaths from the base directory and Y being a one hot encoded
    # label (neg pos)
    def load(self):
        train_pos_files = [
            f"/train/pos/{f}"
            for f in os.listdir(self.train_pos_dir_path)
            if os.path.isfile(os.path.join(self.train_pos_dir_path, f))
        ]
        train_neg_files = [
            f"/train/neg/{f}"
            for f in os.listdir(self.train_neg_dir_path)
            if os.path.isfile(os.path.join(self.train_neg_dir_path, f))
        ]
        test_pos_files = [
            f"/test/pos/{f}"
            for f in os.listdir(self.test_pos_dir_path)
            if os.path.isfile(os.path.join(self.test_pos_dir_path, f))
        ]
        test_neg_files = [
            f"/test/neg/{f}"
            for f in os.listdir(self.test_neg_dir_path)
            if os.path.isfile(os.path.join(self.test_neg_dir_path, f))
        ]
        neg = np.array([1, 0])
        pos = np.array([0, 1])

        return {
            "X_train": train_pos_files + train_neg_files,
            "y_train": np.concatenate(
                (np.tile(pos, (1, len(train_pos_files))), np.tile(neg, (1, len(train_neg_files)))),
                axis=0,
            ),
            "X_test": test_pos_files + test_neg_files,
            "y_test": np.concatenate(
                (np.tile(pos, (1, len(test_pos_files))), np.tile(neg, (1, len(test_neg_files)))),
                axis=0,
            ),
        }
