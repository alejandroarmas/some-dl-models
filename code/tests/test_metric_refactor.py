import os
import unittest
from code.base_class.method import methodConfig
from code.base_class.notifier import MLEventType
from code.lib.comet_listeners import (
    CometConfig,
    CometExperimentTracker,
    CometMethodHandler,
)
from code.lib.notifier import ClassificationNotification, MethodNotifier
from code.stage_1_code.Method_MLP import Method_MLP
from unittest.mock import MagicMock

import torch
from comet_ml import Experiment
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
)


class TestMetrics(unittest.TestCase):
    def test_metrics(self):
        device = torch.device("cpu")
        config = CometConfig(
            {
                "api_key": os.environ["COMET_API_KEY"],
                "project_name": "some-dl-models",
                "workspace": "ecs189g",
            }
        )
        experiment_tracker = CometExperimentTracker(config, dry_run=True)

        m_notifier = MethodNotifier()
        m_notifier.subscribe(experiment_tracker.method_listener, MLEventType("method"))
        batch_metrics = MetricCollection(
            [
                BinaryAccuracy().to(device),
                BinaryF1Score().to(device),
                BinaryPrecision().to(device),
                BinaryRecall().to(device),
            ]
        )

        m_config = methodConfig(
            {
                "name": "dummy-method",
                "description": "This is a dummy model",
                "hyperparameters": {},
            }
        )

        method_obj = Method_MLP(m_config, m_notifier, batch_metrics)

        # ensures that past references to batch_metrics will not break
        assert method_obj.batch_metrics == method_obj.train_batch_metrics

        x = ClassificationNotification(epoch=100, loss=10, accuracy=1, precision=2, f1=3, recall=4)
        y = ClassificationNotification(
            epoch=100, loss=10, accuracy=1, precision=2, f1=3, recall=4, type="train"
        )

        #  ensures that our old way of creating notification objects is still compatable with new code
        assert x == y

        z = ClassificationNotification(
            epoch=100, loss=10, accuracy=1, precision=2, f1=3, recall=4, type="test"
        )

        assert x != z

        fakeExperiment = MagicMock(spec=Experiment)
        handler = CometMethodHandler(fakeExperiment)
        handler.update(y)
        fakeExperiment.train.assert_called_with()
        fakeExperiment.test.assert_not_called()

        fakeExperiment.reset_mock()
        handler.update(z)
        fakeExperiment.train.assert_not_called()
        fakeExperiment.test.assert_called_with()


if __name__ == "__main__":
    unittest.main()
