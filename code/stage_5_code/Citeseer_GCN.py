from code.base_class.method import method, methodConfig
from code.base_class.notifier import MLEventType
from code.lib.notifier import ClassificationNotification, MethodNotifier
from typing import Optional

import torch
import torch.nn as nn2
from torch import nn
from torch.nn import Dropout
from torch_geometric.nn.conv import GCNConv  # type: ignore
from torchmetrics import MetricCollection


class Citeseer_GCN(method, nn.Module):
    def __init__(
        self,
        config: methodConfig,
        manager: Optional[MethodNotifier] = None,
        metrics: Optional[MetricCollection] = None,
    ):
        nn.Module.__init__(self)
        method.__init__(self, config, manager, metrics)
        p = config["hyperparameters"]
        self.learning_rate = p["learning_rate"]
        self.input_dim = p["input_dim"]
        self.output_dim = p["output_dim"]
        self.max_epoch = p["max_epoch"]
        self.weight_decay = p["weight_decay"]
        self.dropout = p["dropout"]


        # note: potentially add a dropout layer between conv1 and conv2?
        self.act = nn.ReLU()
        self.conv1 = GCNConv(p["input_dim"], p["hidden_dim_1"])
        self.drop1 = Dropout(self.dropout)
        self.conv2 = GCNConv(p["hidden_dim_1"], p["hidden_dim_2"])
        self.drop2 = Dropout(self.dropout)
        self.fc = nn2.Linear(p["hidden_dim_1"], self.output_dim)

    def forward(self, X, edges):
        out = X
        out = self.act(self.conv1(out, edges))
        #out = self.drop1(out)
        #out = self.act(self.conv2(out, edges))
        #out = self.drop2(out)
        out = self.fc(out)
        return out

    def train_model(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        loss_function = nn.CrossEntropyLoss()
        names = {
            "MulticlassAccuracy": "accuracy",
            "MulticlassF1Score": "f1",
            "MulticlassPrecision": "precision",
            "MulticlassRecall": "recall",
        }

        print("--start training...")
        data = self.data
        X = data["graph"]["X"]
        edges = torch.LongTensor(data["graph"]["edge"].T)
        labels = data["graph"]["y"]
        labels_onehot = torch.nn.functional.one_hot(labels).float()
        train_idx, val_idx = (
            data["train_test_val"]["train_idx"],
            data["train_test_val"]["val_idx"],
        )
        self.train()
        for epoch in range(self.max_epoch):
            # The GNN will parse and predict all nodes at once, but only the loss for
            # the training set will be used to backpropagate
            y_pred_dist = self(X, edges)
            y_pred = y_pred_dist.max(1)[1]

            train_loss = loss_function(y_pred_dist[train_idx], labels_onehot[train_idx])
            val_loss = loss_function(y_pred_dist[val_idx], labels_onehot[val_idx])

            optimizer.zero_grad()

            train_loss.backward()

            optimizer.step()
            if self.notification_manager is not None:
                self.train_batch_metrics.update(y_pred[train_idx], labels[train_idx])
                self.val_batch_metrics.update(y_pred[val_idx], labels[val_idx])

            train_accumulated_loss = self.batch_metrics.compute()
            val_accumulated_loss = self.val_batch_metrics.compute()

            if epoch % 10 == 0:
                print(
                    "Epoch:",
                    epoch,
                    "Train Loss:",
                    train_loss.item(),
                    "Val Loss:",
                    val_loss.item(),
                    "Train Metrics:",
                    train_accumulated_loss.items(),
                    "Val Metrics:",
                    val_accumulated_loss.items(),
                )
            self.notification_manager.notify(
                MLEventType("method"),
                ClassificationNotification(
                    epoch=epoch,
                    type="train",
                    loss=train_loss.item(),
                    **{names[n]: m.compute() for n, m in self.train_batch_metrics.items()},
                ),
            )
            self.notification_manager.notify(
                MLEventType("method"),
                ClassificationNotification(
                    epoch=epoch,
                    type="validation",
                    loss=val_loss.item(),
                    **{names[n]: m.compute() for n, m in self.val_batch_metrics.items()},
                ),
            )
            for metric in self.train_batch_metrics.values():
                metric.reset()
            for metric in self.val_batch_metrics.values():
                metric.reset()

    def test_model(self):
        y_pred = None
        self.eval()
        data = self.data
        X = data["graph"]["X"]
        edges = torch.LongTensor(data["graph"]["edge"].T)
        labels = data["graph"]["y"]
        test_idx = data["train_test_val"]["test_idx"]
        names = {
            "MulticlassAccuracy": "accuracy",
            "MulticlassF1Score": "f1",
            "MulticlassPrecision": "precision",
            "MulticlassRecall": "recall",
        }
        with torch.no_grad():
            y_pred = self.test(X, edges)
            count = 0
            total = 0
            for pred, real in zip(y_pred[test_idx], labels[test_idx]):
                if pred == real:
                    count += 1
                total += 1
            self.test_batch_metrics.update(y_pred[test_idx], labels[test_idx])
            accumulated_loss = self.test_batch_metrics.compute()
            print(f"{accumulated_loss.items()=}")

        self.notification_manager.notify(
            MLEventType("method"),
            ClassificationNotification(
                epoch=self.max_epoch,
                type="test",
                loss=0,
                **{names[n]: m.compute() for n, m in self.test_batch_metrics.items()},
            ),
        )
        return {"pred_y": y_pred[test_idx], "true_y": labels[test_idx]}

    def test(self, X, edges):
        y_pred = self.forward(X, edges)

        return y_pred.max(1)[1]

    def run(self):
        self.train_model()

        return self.test_model()
