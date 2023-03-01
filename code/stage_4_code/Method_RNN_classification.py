from code.base_class.method import method, methodConfig
from code.lib.notifier import (
    ClassificationNotification,
    MethodNotifier,
    MLEventType,
)
from typing import Optional, Tuple

import torch
import torchtext  # type: ignore
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection


class MethodRNNClassification(method, nn.Module):
    training_loader: DataLoader
    testing_loader: DataLoader

    def __init__(
        self,
        config: methodConfig,
        manager: Optional[MethodNotifier] = None,
        metrics: Optional[MetricCollection] = None,
    ):
        method.__init__(self, config, manager, metrics)
        self.p = config["hyperparameters"]
        p = config["hyperparameters"]
        # Build vocab
        self.glove = torchtext.vocab.GloVe(name="6B", dim=50)
        self.embedding = nn.Embedding.from_pretrained(self.glove.vectors, freeze=True)
        self.tokenizer = torchtext.data.get_tokenizer("basic_english")
        self.rnn = nn.RNN(
            input_size=p["input_size"],
            hidden_size=p["hidden_size"],
            num_layers=p["num_layers"],
            nonlinearity=p["nonlinearity"],
            bias=p["bias"],
            batch_first=p["batch_first"],
            dropout=p["dropout"],
            bidirectional=False,
        )
        self.learning_rate = p["learning_rate"]
        self.max_epoch = p["max_epoch"]
        self.batch_size = p["batch_size"]

    # input is the raw text of a document
    def forward(
        self, input: torch.Tensor, hx: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        #  torch_tensor_first_word = torch.tensor(glove.stoi[tokenized_sentence[0]], dtype=torch.long)
        #  embeddings_for_first_word = my_embeddings(torch_tensor_first_word)
        tokenized_input = self.tokenizer(input)
        if len(tokenized_input) > self.p["input_size"]:  # truncate
            beginning_len = self.p["input_size"] // 2
            ending_len = self.p["input_size"] - beginning_len
            tokenized_input = tokenized_input[:beginning_len] + tokenized_input[ending_len:]
            print(f"Truncated {tokenized_input=}")

        embedded_input = self.embedding(
            torch.tensor([self.glove.stoi[t] for t in tokenized_input], dtype=torch.long)
        )
        # problem: how tf do I know how big the input vectors are?
        # if len(embedded_input) < self.p["input_size"]:
        #     # pad
        #     embedded_input = torch.cat((embedded_input, torch.zeros(self.p)))
        first_hidden = torch.zeros((self.p["hidden_size"]))
        out, hidden = self.rnn(embedded_input, first_hidden)

        return out, hidden

    def train(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        loss_function = nn.BCELoss()
        names = {
            "BinaryAccuracy": "accuracy",
            "BinaryF1Score": "f1",
            "BinaryPrecision": "precision",
            "BinaryRecall": "recall",
        }
        print("training type: ", type(self.training_loader))
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
