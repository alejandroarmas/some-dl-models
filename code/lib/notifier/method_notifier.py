from code.base_class.notifier import MLEventNotifier, MLEventType
from dataclasses import asdict, dataclass


@dataclass(frozen=True, kw_only=True)
class MethodNotification:
    """
        This subclass can be extended to support a flexable
        collection of metrics into our experiment handlers. These
        metrics measure statistics about our model's performance
        on a given batch.

    Args:
        epoch (int): the epoch a metric is measured at
        loss (int): the current loss of our epoch
    """

    epoch: int
    loss: float
    type: str

    def dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True, kw_only=True)
class ClassificationNotification(MethodNotification):
    """
        This is a specialization of our method notifier for
        classification.

        In classification tasks, we are interested in measuring
        the metrics associated with accuracy, f1, precision, as
        well as recall.

    Args:
        accuracy (float): _description_
        f1 (float): _description_
        precision (float): _description_
        recall (float): _description_

    """

    accuracy: float
    f1: float
    precision: float
    recall: float


class MethodNotifier(MLEventNotifier):

    _notification_types = ["method"]

    def notify(self, event_type: MLEventType, data: MethodNotification) -> None:

        assert event_type.event_str in self._notification_types

        if event_type.event_str in self.subscribers:
            for subscribers in self.subscribers[event_type.event_str]:
                subscribers.update(data)
