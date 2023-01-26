"""
Base SettingModule class for all experiment settings
"""

# Copyright (c) 2017 Jiawei Zhang <jwzhanggy@gmail.com>
# License: TBD

import abc
from code.base_class.dataset import dataset
from code.base_class.evaluate import evaluate
from code.base_class.method import method
from code.base_class.result import result
from code.lib.notifier import (
    MLEventType,
    SettingMSG,
    SettingNotification,
    SettingNotifier,
)
from typing import Optional, TypedDict

# -----------------------------------------------------


class SettingConfig(TypedDict):
    name: str
    description: str


class setting:
    """
    SettingModule: Abstract Class
    Entries:
    """

    _setting_name: str
    _setting_description: str

    _dataset: dataset
    _method: method
    _result: result
    _evaluate: evaluate

    __manager: Optional[SettingNotifier]

    def __init__(self, config: SettingConfig, manager: Optional[SettingNotifier] = None):
        self._setting_name = config["name"]
        self._setting_description = config["description"]
        self.__manager = manager

    def prepare(
        self, sDataset: dataset, sMethod: method, sResult: result, sEvaluate: evaluate
    ) -> None:
        self._dataset = sDataset
        self._method = sMethod
        self._result = sResult
        self._evaluate = sEvaluate

    def print_setup_summary(self) -> None:

        if self.__manager is not None:

            metadata = SettingMSG(
                {
                    "dataset": self._dataset.dataset_name,
                    "method": self._method.method_name,
                    "setting": self._setting_name,
                    "result": self._result.result_name,
                    "evaluation": self._evaluate.evaluate_name,
                }
            )

            self.__manager.notify(MLEventType("setting"), SettingNotification(metadata))

        print(
            "dataset:",
            self._dataset.dataset_name,
            ", method:",
            self._method.method_name,
            ", setting:",
            self._setting_name,
            ", result:",
            self._result.result_name,
            ", evaluation:",
            self._evaluate.evaluate_name,
        )

    @abc.abstractmethod
    def load_run_save_evaluate(self):
        return
