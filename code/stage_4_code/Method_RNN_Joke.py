"""
Concrete MethodModule class for a specific learning MethodModule
"""

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.method import method, methodConfig
from code.lib.notifier import (
    ClassificationNotification,
    MethodNotifier,
    MLEventType,
)
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection


class MethodJokeGeneration(method, nn.Module):
    training_loader: DataLoader
    testing_loader: DataLoader

    # it defines the the MLP model architecture, e.g.,
    # how many layers, size of variables in each layer, activation function, etc.
    # the size of the input/output portal of the model architecture should be consistent with our data input and desired output
    def __init__(
        self,
        config: methodConfig,
        manager: Optional[MethodNotifier] = None,
        metrics: Optional[MetricCollection] = None,
    ):
        nn.Module.__init__(self)
        method.__init__(self, config, manager, metrics)

        p = config["hyperparameters"]

        self.input_size = p["output_dim_0"]
        self.hidden_size = p["output_dim_1"]
        self.output_size = p["vocab_size"]
        self.batch_size = p["batch_size"]
        self.learning_rate = p["learning_rate"]
        self.max_epoch = p["max_epoch"]
        self.num_character_context = p["num_character_context"]
        self.vocab_size = p["vocab_size"]

        self.embedding = nn.Embedding(
            num_embeddings=p["vocab_size"] + 1,  # to skip index 0
            embedding_dim=p["output_dim_0"],
        )
        # print(f"{self.embedding.weight.shape=}")
        self.lstm = nn.LSTM(
            input_size=p["output_dim_0"], hidden_size=p["output_dim_1"], batch_first=True
        )
        self.fc = nn.Linear(p["num_character_context"] * p["output_dim_1"], p["vocab_size"])

    def forward(self, input_data, hidden, cell):
        # print(f'{input_data.shape=}')
        embed = self.embedding(input_data)

        output = F.relu(embed)
        output, (hidden, cell) = self.lstm(output, (hidden, cell))
        # print(f'{output.shape=}')
        # print(f'{hidden.shape=}')
        # print(f'{cell.shape=}')
        output = output.reshape((output.shape[0], -1))
        # print(f'{output.shape=}')
        output = F.relu(output)
        logits = self.fc(output)
        # print(f'{logits.shape=}')

        return logits, (hidden, cell)

    def init_state(self, num_samples: int):
        return (
            torch.zeros(1, num_samples, self.hidden_size),
            torch.zeros(1, num_samples, self.hidden_size),
        )

    # backward error propagation will be implemented by pytorch automatically
    # so we don't need to define the error backpropagation function here

    def train_model(self):

        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        loss_function = nn.CrossEntropyLoss()

        names = {
            "MulticlassAccuracy": "accuracy",
            "MulticlassF1Score": "f1",
            "MulticlassPrecision": "precision",
            "MulticlassRecall": "recall",
        }

        for epoch in range(self.max_epoch):
            state_h, state_c = self.init_state(self.batch_size)

            for idx, batch in enumerate(self.training_loader):

                y_pred, (state_h, state_c) = self.forward(batch["context"], state_h, state_c)

                state_h = state_h.detach()
                state_c = state_c.detach()

                train_loss = loss_function(y_pred, batch["label"])

                optimizer.zero_grad()
                train_loss.backward()

                optimizer.step()

                if self.notification_manager is not None:
                    self.batch_metrics.update(y_pred, batch["label"])

            accumulated_loss = self.batch_metrics.compute()

            # print(f"{train_loss.item()=}")
            if epoch % 10 == 0:
                print(
                    "Epoch:",
                    epoch,
                    "Loss:",
                    train_loss.item(),
                    "Metrics:",
                    accumulated_loss.items(),
                )
            self.notification_manager.notify(
                MLEventType("method"),
                ClassificationNotification(
                    epoch=epoch,
                    loss=train_loss.item(),
                    **{names[n]: m.compute() for n, m in self.batch_metrics.items()},
                ),
            )

            for metric in self.batch_metrics.values():
                metric.reset()

    def test_model(self):

        y_pred = None
        self.eval()

        with torch.no_grad():
            for batch in self.testing_loader:
                y_pred, _ = self.forward(batch["context"], *self.init_state(self.batch_size))
                self.batch_metrics.update(y_pred, batch["label"])
            self.batch_metrics.compute()
            # print(f"{accumulated_loss.items()=}")

        return {"pred_y": torch.argmax(y_pred, axis=1), "true_y": batch["label"]}

    def run(self):
        print("method running...")
        print("--start training...")
        self.train_model()
        print("--start testing...")

        return self.test_model()
