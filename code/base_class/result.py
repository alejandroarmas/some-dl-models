"""
Base evaluate class for all evaluation metrics and methods
"""

# Copyright (c) 2017 Jiawei Zhang <jwzhanggy@gmail.com>
# License: TBD


import abc
from code.lib.notifier import ResultNotifier
from typing import Optional, TypedDict


class resultConfig(TypedDict):
    name: str
    description: str
    destination_folder_path: str
    destination_file_name: str


class result:
    """
    ResultModule: Abstract Class
    Entries:
    """

    data = None

    result_name: str
    result_description: str

    result_destination_folder_path: str
    result_destination_file_name: str

    _manager: Optional[ResultNotifier]
    fold_count: Optional[int]

    # initialization function
    def __init__(self, config: resultConfig, manager: Optional[ResultNotifier] = None):

        self.result_name = config["name"]
        self.result_description = config["description"]
        self.result_destination_folder_path = config["destination_folder_path"]
        self.result_destination_file_name = config["destination_file_name"]
        self._manager = manager

    def get_data(self, _d) -> None:
        self.data = _d

    @abc.abstractmethod
    def save(self):
        return

    @abc.abstractmethod
    def load(self):
        return
