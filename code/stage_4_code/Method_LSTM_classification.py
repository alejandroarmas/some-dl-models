from code.base_class.method import method, methodConfig
from code.lib.notifier import (
    ClassificationNotification,
    MethodNotifier,
    MLEventType,
)
from code.stage_4_code.Dataset_Loader_Classification import (
    Classification_Vocabulary,
)
from typing import Optional, Tuple

import numpy as np
import torch
import torchtext  # type: ignore
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection


class MethodLSTMClassification(method, nn.Module):
    training_loader: DataLoader
    testing_loader: DataLoader

    dataset_vocab: Classification_Vocabulary

    def __init__(
        self,
        config: methodConfig,
        manager: Optional[MethodNotifier] = None,
        metrics: Optional[MetricCollection] = None,
    ):
        nn.Module.__init__(self)
        method.__init__(self, config, manager, metrics)
        p = config["hyperparameters"]
        self.input_size = p["input_size"]
        self.vocab_size = p["vocab_size"]

        self.tokenizer = torchtext.data.get_tokenizer("basic_english")

        # build embedding layer with random embeddings
        self.embedding_matrix = torch.tensor(np.random.random((self.vocab_size, self.input_size)))
        self.embedding_matrix[0] = torch.zeros(self.input_size, dtype=torch.float)
        self.embedding_matrix[1] = torch.zeros(self.input_size, dtype=torch.float)

        self.embedding = nn.Embedding.from_pretrained(self.embedding_matrix, freeze=False).float()
        self.lstm = nn.LSTM(
            input_size=p["input_size"],
            hidden_size=p["hidden_size"],
            num_layers=p["num_layers"],
            bias=True,
            batch_first=True,  # defines order as (batch, sequence, features)
            dropout=p["dropout"],
            bidirectional=False,
        ).float()
        # self.dense = nn.Sequential(
        #     nn.Linear(p["hidden_size"], p["dense_size_1"]),
        #     nn.Sigmoid(),
        # )
        # define the output layer
        self.output = nn.Sequential(nn.Linear(p["hidden_size"], p["output_dim_1"]), nn.Sigmoid())
        self.output.requires_grad_(True)
        self.num_layers = p["num_layers"]
        self.num_directions = 1  # 2 if bidirectional
        self.learning_rate = p["learning_rate"]
        self.max_epoch = p["max_epoch"]
        self.batch_size = p["batch_size"]
        self.hidden_size = p["hidden_size"]

    def forward(self, batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        tokenized_input = [self.tokenizer(b) for b in batch]
        input_indices = [
            torch.tensor(
                [self.dataset_vocab.get_vocab()[token] for token in document],
                dtype=torch.long,
            )
            for document in tokenized_input
        ]
        padded_input = nn.utils.rnn.pad_sequence(
            input_indices,
            padding_value=self.dataset_vocab.get_vocab()["<pad>"],
            batch_first=True,
        )
        # print("pads", padded_input, padded_input.size())
        embedded_input = self.embedding(padded_input)
        # print("embeds",embedded_input, embedded_input.size())
        # pads everything in the batch to the size of the largest object in the batch
        # first_hidden = torch.rand(
        #     (self.num_layers * self.num_directions, self.batch_size, self.hidden_size),
        #     dtype=torch.float,
        # )
        output, (hn, cn) = self.lstm(embedded_input)
        # print(hn) #nans
        # fc = self.dense(hn)
        out = self.output(hn)
        # print(out) #nan
        out = torch.squeeze(out, dim=0)
        out = torch.squeeze(out, dim=1)
        # print(out) #nan
        return out

    def train_model(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        loss_function = nn.BCELoss()
        names = {
            "BinaryAccuracy": "accuracy",
            "BinaryF1Score": "f1",
            "BinaryPrecision": "precision",
            "BinaryRecall": "recall",
        }

        print("--start training...")

        for epoch in range(self.max_epoch):
            for idx, batch in enumerate(self.training_loader):
                y_pred = self.forward(batch["contents"])
                y_pred = torch.unsqueeze(y_pred, dim=1)

                train_loss = loss_function(y_pred, batch["label"].float())
                optimizer.zero_grad()

                train_loss.backward()

                optimizer.step()
                if self.notification_manager is not None:
                    self.batch_metrics.update(batch["label"], torch.round(y_pred))
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
                y_pred = self.test(batch["contents"])
                self.batch_metrics.update(batch["label"], y_pred)
            accumulated_loss = self.batch_metrics.compute()
            print(f"{accumulated_loss.items()=}")

        return {"pred_y": y_pred, "true_y": batch["label"]}

    def test(self, X):
        # do the testing, and result the result
        y_pred = self.forward(X)
        # convert the probability distributions to the corresponding labels
        # instances will get the labels corresponding to the largest probability
        return torch.unsqueeze(torch.round(y_pred), dim=1)

    def run(self):
        print("method running...")
        self.train_model()
        print("--start testing...")

        return self.test_model()
