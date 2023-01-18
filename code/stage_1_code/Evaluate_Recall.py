'''
Concrete Evaluate class for a specific evaluation metrics
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.evaluate import evaluate, EvaluateConfig
from sklearn.metrics import recall_score

from code.lib.notifier import EvaluateNotification, EvaluateNotifier, MLEventType


class Evaluate_Recall(evaluate):

    _measure = 'recall'

    def __init__(self, config: EvaluateConfig, manager: EvaluateNotifier = None):
        super().__init__(config, manager)

    def evaluate(self):

        output: float = recall_score(self.data['true_y'], self.data['pred_y'])

        if self._manager is not None:
            self._manager.notify(MLEventType('evaluate'), EvaluateNotification(output, self._measure))        

        return output
        