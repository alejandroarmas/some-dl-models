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


class MethodCNN(method, nn.Module):
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

        self.conv_layer1 = nn.Conv2d(
            in_channels=p["conv_channels_in_dim"],
            out_channels=p["conv_channels_out_dim_0"],
            kernel_size=p["conv_kernel_size"],
        )
        self.max_pool = nn.MaxPool2d(
            kernel_size=p["pool_kernel_size"],
            stride=p["pool_stride"],
        )
        self.conv_layer2 = nn.Conv2d(
            in_channels=p["conv_channels_out_dim_0"],
            out_channels=p["conv_channels_out_dim_1"],
            kernel_size=p["conv_kernel_size"],
        )

        self.input_dim_0 = (
            p["conv_channels_out_dim_1"] * p["conv_kernel_size"] * p["conv_kernel_size"]
        )
        self.fc1 = nn.Linear(self.input_dim_0, p["output_dim_0"])
        self.fc2 = nn.Linear(p["output_dim_0"], p["output_dim_1"])
        self.fc3 = nn.Linear(p["output_dim_1"], p["output_dim_2"])
        self.learning_rate = p["learning_rate"]
        self.max_epoch = p["max_epoch"]
        self.batch_size = p["batch_size"]

    # Progresses data across layers
    def forward(self, x):
        out = self.max_pool(F.relu(self.conv_layer1(x)))
        out = self.max_pool(F.relu(self.conv_layer2(out)))

        out = out.view(self.batch_size, self.input_dim_0)

        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

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

            for idx, batch in enumerate(self.training_loader):

                y_pred = self.forward(batch["image"])

                train_loss = loss_function(y_pred, batch["label"])

                optimizer.zero_grad()
                train_loss.backward()

                optimizer.step()

                if self.notification_manager is not None:
                    self.batch_metrics.update(batch["label"], y_pred.max(1)[1])

            accumulated_loss = self.batch_metrics.compute()

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
                y_pred = self.test(batch["image"])
                self.batch_metrics.update(batch["label"], y_pred.max(1)[1])
            accumulated_loss = self.batch_metrics.compute()
            print(f"{accumulated_loss.items()=}")

        return {"pred_y": y_pred.max(1)[1], "true_y": batch["label"]}

    def run(self):
        print("method running...")
        print("--start training...")
        self.train_model()
        print("--start testing...")

        return self.test_model()
