"""
Base MethodModule class for all models and frameworks
"""

# Copyright (c) 2017 Jiawei Zhang <jwzhanggy@gmail.com>
# License: TBD

import abc
from code.lib.notifier import MethodNotifier
from typing import Optional, TypedDict

from torchmetrics import MetricCollection


class methodConfig(TypedDict):
    name: str
    description: str
    hyperparameters: dict


class method:
    """
    MethodModule: Abstract Class
    Entries: method_name: the name of the MethodModule
            batch_metrics (Optional[dict]): see method_notifier.py to see what are the allowable types of metrics
                to pass into this dict
            notification_manager (Optional[MethodNotifier]): this pings our experiment handler with information
                pertinent to each epoch's training. Through this we are able to log information.

             method_description: the textual description of the MethodModule

             method_start_time: start running time of MethodModule
             method_stop_time: stop running time of MethodModule
             method_running_time: total running time of the MethodModule
             method_training_time: time cost of the training phrase
             method_testing_time: time cost of the testing phrase
    """

    method_name: str
    method_description: str

    data = None
    notification_manager: Optional[MethodNotifier]

    batch_metrics: Optional[MetricCollection]
    method_start_time = None
    method_stop_time = None
    method_running_time = None
    method_training_time = None
    method_testing_time = None

    # initialization function
    def __init__(
        self,
        config: methodConfig,
        manager: Optional[MethodNotifier] = None,
        metrics: Optional[MetricCollection] = None,
    ):
        self.method_name = config["name"]
        self.method_description = config["description"]
        self.notification_manager = manager
        self.batch_metrics = metrics

    # running function
    @abc.abstractmethod
    def run(self, trainData, trainLabel, testData):
        return
