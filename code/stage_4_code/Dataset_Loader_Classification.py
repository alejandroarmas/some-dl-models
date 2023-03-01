import os
from code.base_class.dataset import dataset, datasetConfig
from code.lib.notifier import DatasetNotifier
from typing import Optional

from torch.utils.data import Dataset


class Classification_Dataset(dataset, Dataset):
    def __init__(self, config: datasetConfig, manager: Optional[DatasetNotifier] = None):
        super().__init__(config, manager)
        self.pos_dir_path = f"{self.dataset_source_folder_path}/pos/"
        self.neg_dir_path = f"{self.dataset_source_folder_path}/neg/"

        pos_files = [
            f
            for f in os.listdir(self.pos_dir_path)
            if os.path.isfile(os.path.join(self.pos_dir_path, f))
        ]
        neg_files = [
            f
            for f in os.listdir(self.neg_dir_path)
            if os.path.isfile(os.path.join(self.neg_dir_path, f))
        ]

        self.labels = []
        for p in pos_files:
            self.labels.append((p, "pos"))
        for n in neg_files:
            self.labels.append((n, "neg"))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx][1]
        contents = ""
        if label == "pos":
            with open(f"{self.pos_dir_path}{self.labels[idx][0]}") as f:
                contents = f.read()
        elif label == "neg":
            with open(f"{self.neg_dir_path}{self.labels[idx][0]}") as f:
                contents = f.read()

        return contents, label

    def load(self):
        pass
