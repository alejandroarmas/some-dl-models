from code.base_class.notifier import MLEventNotifier, MLEventType

from dataclasses import dataclass


@dataclass(frozen=True)
class EvaluateNotification():
    evaluation: float
    measure: str

class EvaluateNotifier(MLEventNotifier):

    _notification_types = ["evaluate"]

    def notify(self, event_type: MLEventType, data: EvaluateNotification) -> None:
        assert event_type.event_str in self._notification_types

        if event_type.event_str in self.subscribers:
            for subscribers in self.subscribers[event_type.event_str]:
                subscribers.update(data)
                