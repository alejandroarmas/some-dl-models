from code.base_class.notifier import MLEventListener
from code.lib.notifier import (
    DatasetNotification,
    EvaluateNotification,
    MethodNotification,
    ResultNotification,
    SettingNotification,
)
from typing import Optional, Protocol, runtime_checkable


@runtime_checkable
class ExperimentTracker(Protocol):
    method_listener: Optional[MLEventListener]
    evaluate_listener: Optional[MLEventListener]
    result_listener: Optional[MLEventListener]
    setting_listener: Optional[MLEventListener]
    dataset_listener: Optional[MLEventListener]

    def log_model(self) -> None:
        ...  # Empty method body (explicit '...')

    def log_method(self, data: MethodNotification) -> None:
        ...  # Empty method body (explicit '...')

    def log_evaluation(self, data: EvaluateNotification) -> None:
        ...  # Empty method body (explicit '...')

    def log_dataset(self, data: DatasetNotification) -> None:
        ...  # Empty method body (explicit '...')

    def log_setting(self, data: SettingNotification) -> None:
        ...  # Empty method body (explicit '...')

    def log_results(self, data: ResultNotification) -> None:
        ...  # Empty method body (explicit '...')
