from abc import ABC, abstractmethod
from typing import List, Dict


from dataclasses import dataclass


@dataclass(frozen=True)
class MLEventType():
    event_str: str


class MLEventListener(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def update(self, data) -> None:
        pass


class MLEventNotifier(ABC):

    subscribers: Dict[str, List[MLEventListener]]

    def __init__(self):
        self.subscribers = {}


    def subscribe(self, listener: MLEventListener, event_type: MLEventType) -> None:
        
        if event_type.event_str in self.subscribers:
            self.subscribers[event_type.event_str].append(listener)
        else:
            self.subscribers[event_type.event_str] = [listener]

    def unsubscribe(self, listener: MLEventListener, event_type: MLEventType) -> None:

        if event_type.event_str in self.subscribers:
            self.subscribers[event_type.event_str] = [sub for sub in self.subscribers[event_type.event_str] if sub != listener]


    def subscription_exists(self, listener: MLEventListener, event_type: MLEventType) -> bool:
        if event_type.event_str not in self.subscribers:
            return False
        for subscriber in self.subscribers[event_type.event_str]:
            if listener is subscriber:
                return True
        return False


    @abstractmethod
    def notify(self, event_type: MLEventType, data) -> None:
        pass