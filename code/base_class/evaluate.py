"""
Base evaluate class for all evaluation metrics and methods
"""

# Copyright (c) 2017 Jiawei Zhang <jwzhanggy@gmail.com>
# License: TBD

import abc
from code.lib.notifier import EvaluateNotifier
from typing import Optional, TypedDict


class EvaluateConfig(TypedDict):
    name: str
    description: str


class evaluate:
    """
    evaluate: Abstract Class
    Entries:
    """

    evaluate_name: str
    evaluate_description: Optional[str]
    _manager: Optional[EvaluateNotifier]
    data = None

    # initialization function
    def __init__(self, config: EvaluateConfig, manager: Optional[EvaluateNotifier] = None):
        self.evaluate_name = config["name"]
        self.evaluate_description = config["description"]
        self._manager = manager

    @abc.abstractmethod
    def evaluate(self) -> float:
        ...
