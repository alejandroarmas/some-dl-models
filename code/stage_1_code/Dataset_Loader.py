"""
Concrete IO class for a specific dataset
"""

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.dataset import dataset, datasetConfig
from code.lib.notifier import DatasetNotification, DatasetNotifier, MLEventType
from typing import Optional
import pandas as pd


class Dataset_Loader(dataset):
    data = None

    def __init__(self, config: datasetConfig, manager: Optional[DatasetNotifier] = None):
        super().__init__(config, manager)

    def load(self):
        print("\nloading data...\n")
        X = []
        y = []

        # read in dataset using pandas
        file_Path = self.dataset_source_folder_path + self.dataset_source_file_name
        # with open(file_Path) as file:
        data_Frame = pd.read_csv(file_Path)
        print(data_Frame, "\n")
        
        # extract labels and features from dataset ([row_start:row_end , column_start, column_end])
        y = data_Frame.iloc[:, 0]
        X = data_Frame.iloc[:, 1:]

        return(X, y)
