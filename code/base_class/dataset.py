"""
Base IO class for all datasets
"""

# Copyright (c) 2017 Jiawei Zhang <jwzhanggy@gmail.com>
# License: TBD


import abc
from code.lib.notifier import DatasetNotifier
from typing import Optional, TypedDict

import torch


class datasetConfig(TypedDict):
    name: str
    description: str
    source_folder_path: str
    source_file_name: str
    device: torch.device


class dataset:
    """
    dataset: Abstract Class
    Entries: dataset_name: the name of the dataset
             dataset_description: the textual description of the dataset
    """

    dataset_name: str
    dataset_description: str

    dataset_source_folder_path: str
    dataset_source_file_name: str

    _manager: Optional[DatasetNotifier]

    data = None

    # initialization function
    def __init__(self, config: datasetConfig, manager: Optional[DatasetNotifier] = None):
        """
        Parameters: dataset name: dName, dataset description: dDescription
        Assign the parameters to the entries of the base class
        """
        self.dataset_name = config["name"]
        self.dataset_description = config["description"]
        self.dataset_source_folder_path = config["source_folder_path"]
        self.dataset_source_file_name = config["source_file_name"]
        self._manager = manager

    def attach_data(self, _d) -> None:
        self.data = _d

    # information print function
    def print_dataset_information(self):
        """
        Print the basic information about the dataset class
        inclduing the dataset name, and dataset description
        """
        print("Dataset Name: " + self.dataset_name)
        print("Dataset Description: " + self.dataset_description)

    # dataset load abstract function
    @abc.abstractmethod
    def load(self):
        return
