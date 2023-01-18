from code.base_class.notifier import MLEventNotifier, MLEventType


from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetNotification():
    examples_size: int
    filename: str


class DatasetNotifier(MLEventNotifier):

    _notification_types = ["load"]

    def notify(self, event_type: MLEventType, data: DatasetNotification) -> None:
        
        assert event_type.event_str in self._notification_types

        if event_type.event_str in self.subscribers:
            for subscribers in self.subscribers[event_type.event_str]:
                subscribers.update(data)





