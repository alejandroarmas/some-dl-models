from code.base_class.notifier import MLEventNotifier, MLEventType
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional, TypedDict

import torch


class Data(TypedDict):
    y_true: torch.Tensor
    y_pred: torch.Tensor


@dataclass(frozen=True)
class ResultNotification:
    data: Data
    filename: str
    artifact_filename: str
    state_dict: Optional[OrderedDict]


class ResultNotifier(MLEventNotifier):

    _notification_types = [
        "save",
        "load",
    ]

    def notify(self, event_type: MLEventType, data: ResultNotification) -> None:

        assert event_type.event_str in self._notification_types

        if event_type.event_str in self.subscribers:
            for subscribers in self.subscribers[event_type.event_str]:
                subscribers.update(data)
