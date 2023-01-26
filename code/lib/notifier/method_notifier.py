from code.base_class.notifier import MLEventNotifier, MLEventType
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class MethodNotification:
    y_true: torch.Tensor
    y_pred: torch.Tensor
    epoch: int
    accuracy: float
    loss: float


class MethodNotifier(MLEventNotifier):

    _notification_types = ["method"]

    def notify(self, event_type: MLEventType, data: MethodNotification) -> None:

        assert event_type.event_str in self._notification_types

        if event_type.event_str in self.subscribers:
            for subscribers in self.subscribers[event_type.event_str]:
                subscribers.update(data)
