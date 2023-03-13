"""
Base SettingModule class for all experiment settings
"""

# Copyright (c) 2017 Jiawei Zhang <jwzhanggy@gmail.com>
# License: TBD

import abc
from code.base_class.artifacts import artifacts
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

import torch
from typing_extensions import NotRequired


class SettingConfig(TypedDict):
    name: str
    description: str
    device: torch.device
    params: NotRequired[dict]


class setting:
    """
    SettingModule: Abstract Class
    Entries:
    """

    _setting_name: str
    _setting_description: str
    params: dict

    _dataset: dataset
    _method: method
    _result: result
    _evaluate: evaluate
    _artifacts: artifacts

    __manager: Optional[SettingNotifier]

    def __init__(self, config: SettingConfig, manager: Optional[SettingNotifier] = None):
        self._setting_name = config["name"]
        self._setting_description = config["description"]
        self.__manager = manager
        self.device = config["device"]
        if "params" in config.keys():
            self.params = config["params"]
        else:
            self.params = {}

    def prepare(
        self,
        sDataset: dataset,
        sMethod: method,
        sResult: result,
        sEvaluate: evaluate,
        sArtifacts: artifacts,
    ) -> None:
        self._dataset = sDataset
        self._method = sMethod
        self._result = sResult
        self._evaluate = sEvaluate
        self._artifacts = sArtifacts

    def print_setup_summary(self) -> None:

        if self.__manager is not None:

            metadata = SettingMSG(
                {
                    "dataset": self._dataset.dataset_name,
                    "method": self._method.method_name,
                    "setting": self._setting_name,
                    "result": self._result.result_name,
                    "evaluation": self._evaluate.evaluate_name,
                    "artifacts": self._artifacts.model_name,
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
            ", artifacts:",
            self._artifacts.model_name,
        )

    @abc.abstractmethod
    def load_run_save_evaluate(self):
        return
