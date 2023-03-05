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
from torchtext.vocab import Vocab  # type: ignore


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
        self.embedding_grad_epoch = p["embedding_grad_epoch"]

    # input is the raw text of a document
    def prepareEmbedding(self, vocabulary: Vocab):
        glove = torchtext.vocab.GloVe(name="6B", dim=self.input_size)
        indices = []
        for entry in vocabulary.get_stoi().keys():
            try:
                # print(entry)
                if entry == "<unk>" or entry == "<pad>":
                    vec = torch.zeros(self.input_size, dtype=torch.float)
                else:
                    index = glove.stoi[entry]
                    vec = glove.vectors[index]
                indices.append(vocabulary.__getitem__(entry))
                self.embedding_matrix[vocabulary.__getitem__(entry)] = vec
            except KeyError:
                pass
                # print("Error", entry)
        percent_prelearned = len(indices) / vocabulary.__len__()
        self.prelearned_indices = torch.LongTensor(indices)
        print(f"{percent_prelearned=}")
        self.embedding = nn.Embedding.from_pretrained(self.embedding_matrix, freeze=False).float()
        print(self.embedding_matrix.size())

    def forward(
        self, batch: torch.Tensor, hx: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Some kind of buggy behavior with ONNX causes None objects to appear in batches of size 1

        tokenized_input = [self.tokenizer(b) for b in batch]
        # print("tokenized", tokenized_input, len(tokenized_input))
        input_indices = [
            torch.tensor(
                [self.dataset_vocab.get_vocab().__getitem__(token) for token in document],
                dtype=torch.long,
            )
            for document in tokenized_input
        ]
        padded_input = nn.utils.rnn.pad_sequence(
            input_indices,
            padding_value=self.dataset_vocab.get_vocab().__getitem__("<pad>"),
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
        print("training type: ", type(self.training_loader))

        print("-----loading pretrained vectors------")
        # self.prepareEmbedding(self.dataset_vocab.get_vocab())
        print("--start training...")

        for epoch in range(self.max_epoch):
            for idx, batch in enumerate(self.training_loader):
                y_pred = self.forward(batch["contents"])
                y_pred = torch.unsqueeze(y_pred, dim=1)
                if torch.isnan(y_pred).any():
                    print(y_pred)
                train_loss = loss_function(y_pred, batch["label"].float())
                optimizer.zero_grad()

                train_loss.backward()

                # # zeroes out the gradient of all prelearned vectors until the noted epoch has passed
                # if epoch < self.embedding_grad_epoch:
                #     self.embedding.weight.grad[self.prelearned_indices] = 0
                # clip = 0.3 #0.3 worked
                # torch.nn.utils.clip_grad_norm_(self.parameters(), clip)
                optimizer.step()
                if self.notification_manager is not None:
                    self.batch_metrics.update(batch["label"], torch.round(y_pred))
            accumulated_loss = self.batch_metrics.compute()

            if epoch == self.embedding_grad_epoch:
                print("unfreezing pretrained vectors")
            self.validate(epoch)
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
                    type="train",
                    **{names[n]: m.compute() for n, m in self.batch_metrics.items()},
                ),
            )
            for metric in self.batch_metrics.values():
                metric.reset()

    def validate(self, epoch):
        names = {
            "BinaryAccuracy": "accuracy",
            "BinaryF1Score": "f1",
            "BinaryPrecision": "precision",
            "BinaryRecall": "recall",
        }
        loss_function = nn.BCELoss()
        for i in range(10):
            batch = next(iter(self.testing_loader))
            # was in for loop before
            with torch.no_grad():
                y_pred = self.forward(batch["contents"])
                y_pred = torch.unsqueeze(y_pred, dim=1)
                y_pred = torch.round(y_pred)

                test_loss = loss_function(y_pred, batch["label"].float())

                # # zeroes out the gradient of all prelearned vectors until the noted epoch has passed
                # if epoch < self.embedding_grad_epoch:
                #     self.embedding.weight.grad[self.prelearned_indices] = 0
                # clip = 0.3 #0.3 worked
                # torch.nn.utils.clip_grad_norm_(self.parameters(), clip)
                if self.notification_manager is not None:
                    self.test_batch_metrics.update(batch["label"], torch.round(y_pred))
                # was in for loop before
            accumulated_loss = self.test_batch_metrics.compute()
            print(accumulated_loss)
        self.notification_manager.notify(
            MLEventType("method"),
            ClassificationNotification(
                epoch=epoch,
                loss=test_loss.item(),
                type="test",
                **{names[n]: m.compute() for n, m in self.test_batch_metrics.items()},
            ),
        )
        for metric in self.test_batch_metrics.values():
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
