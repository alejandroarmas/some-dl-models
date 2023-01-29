"""
Concrete Evaluate class for a specific evaluation metrics
"""

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.evaluate import EvaluateConfig, evaluate
from code.lib.notifier import (
    EvaluateNotification,
    EvaluateNotifier,
    MLEventType,
)
from typing import Optional

from sklearn.metrics import f1_score  # type: ignore


class Evaluate_F1(evaluate):

    _measure = "f1"

    def __init__(self, config: EvaluateConfig, manager: Optional[EvaluateNotifier] = None):
        super().__init__(config, manager)

    def evaluate(self) -> float:

        output: float = f1_score(self.data["true_y"], self.data["pred_y"])  # type: ignore

        if self._manager is not None:
            self._manager.notify(
                MLEventType("evaluate"), EvaluateNotification(output, self._measure)
            )

        return output
