from code.base_class.notifier import MLEventNotifier, MLEventType
from dataclasses import asdict, dataclass

"""
This cluster of class inheritance is necessary to preserve existing functionality while also allowing our method
notifications to distinguish between training data and testing data reports. Essentially when default fields are defined
in a dataclass, they must be defined last. However parent class fields always come before child class fields. All of
this is to use private classes to enforce the proper field order. Since the public classes have the same names and
fields, this will be a transparent change when it comes to how our notifications are created.
"""


@dataclass(frozen=True, kw_only=True)
class _MethodNotification_base:
    epoch: int
    loss: float


@dataclass(frozen=True, kw_only=True)
class _MethodNotification_default:
    type: str = "train"  # default value is train to preserve existing functionality


@dataclass(frozen=True, kw_only=True)
class _ClassificationNotification_base(_MethodNotification_base):
    accuracy: float
    f1: float
    precision: float
    recall: float


@dataclass(frozen=True, kw_only=True)
class _ClassificationNotification_defaults(_MethodNotification_default):
    pass


@dataclass(frozen=True, kw_only=True)
class MethodNotification(_MethodNotification_default, _MethodNotification_base):
    """
        This subclass can be extended to support a flexable
        collection of metrics into our experiment handlers. These
        metrics measure statistics about our model's performance
        on a given batch.

    Args:
        epoch (int): the epoch a metric is measured at
        loss (int): the current loss of our epoch
    """

    def dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True, kw_only=True)
class ClassificationNotification(
    _ClassificationNotification_defaults, MethodNotification, _ClassificationNotification_base
):
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

    pass


class MethodNotifier(MLEventNotifier):

    _notification_types = ["method"]

    def notify(self, event_type: MLEventType, data: MethodNotification) -> None:

        assert event_type.event_str in self._notification_types

        if event_type.event_str in self.subscribers:
            for subscribers in self.subscribers[event_type.event_str]:
                subscribers.update(data)
