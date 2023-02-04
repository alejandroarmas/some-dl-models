"""
Concrete IO class for a specific dataset
"""

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.dataset import dataset, datasetConfig
from code.lib.notifier import DatasetNotifier
from typing import Optional

import pandas as pd


class Dataset_Loader(dataset):
    data = None

    def __init__(self, config: datasetConfig, manager: Optional[DatasetNotifier] = None):
        super().__init__(config, manager)

    def load(self) -> dict:
        print("\nloading data...\n")

        # read in dataset using pandas
        file_Path = self.dataset_source_folder_path + self.dataset_source_file_name
        # with open(file_Path) as file:
        data_Frame: pd.DataFrame = pd.read_csv(file_Path)
        print(data_Frame, "\n")

        # extract labels and features from dataset ([row_start:row_end , column_start, column_end])
        X: pd.DataFrame = data_Frame.iloc[:, 1:]
        y: pd.Series = data_Frame.iloc[:, 0]

        return {"X": X, "y": y}
