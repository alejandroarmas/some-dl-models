"""
Concrete IO class for a specific dataset
"""

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.dataset import dataset, datasetConfig
from code.lib.notifier import DatasetNotification, DatasetNotifier, MLEventType
from typing import Optional


class Dataset_Loader(dataset):
    data = None

    def __init__(self, config: datasetConfig, manager: Optional[DatasetNotifier] = None):
        super().__init__(config, manager)

    def load(self):
        print("loading data...")
        X = []
        y = []

        filename = f"{self.dataset_source_folder_path}{self.dataset_source_file_name}"

        with open(filename, "r") as f:
            for line in f:
                line = line.strip("\n")
                elements = [int(i) for i in line.split(" ")]
                X.append(elements[:-1])
                y.append(elements[-1])
            if self._manager is not None:
                self._manager.notify(MLEventType("load"), DatasetNotification(len(X), filename))

        return {"X": X, "y": y}
