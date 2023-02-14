from code.base_class.notifier import MLEventNotifier, MLEventType
from dataclasses import dataclass
from typing import TypedDict


class SettingMSG(TypedDict):
    dataset: str
    method: str
    setting: str
    result: str
    evaluation: str
    artifacts: str


@dataclass(frozen=True)
class SettingNotification:
    config_settings: SettingMSG


class SettingNotifier(MLEventNotifier):

    _notification_types = ["setting"]

    def notify(self, event_type: MLEventType, data: SettingNotification) -> None:

        assert event_type.event_str in self._notification_types

        if event_type.event_str in self.subscribers:
            for subscribers in self.subscribers[event_type.event_str]:
                subscribers.update(data)
