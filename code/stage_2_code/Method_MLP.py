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
from typing import Any, Optional

import numpy as np
import torch
from torch import nn


class Method_MLP(method, nn.Module):
    data = None
    metrics: dict[str, Any]

    # it defines the the MLP model architecture, e.g.,
    # how many layers, size of variables in each layer, activation function, etc.
    # the size of the input/output portal of the model architecture should be consistent with our data input and desired output
    def __init__(
        self,
        config: methodConfig,
        manager: Optional[MethodNotifier] = None,
        metrics: Optional[dict] = None,
    ):
        nn.Module.__init__(self)
        method.__init__(self, config, manager, metrics)
        # check here for nn.Linear doc: https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
        self.fc_layer_1 = nn.Linear(
            config["hyperparameters"]["input_dim"], config["hyperparameters"]["hidden_dim_0"]
        )
        # check here for nn.ReLU doc: https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html
        self.activation_func_1 = nn.Tanh()
        self.fc_layer_2 = nn.Linear(
            config["hyperparameters"]["hidden_dim_0"], config["hyperparameters"]["hidden_dim_1"]
        )
        self.activation_func_2 = nn.Tanh()
        self.fc_layer_3 = nn.Linear(
            config["hyperparameters"]["hidden_dim_1"], config["hyperparameters"]["output_dim"]
        )
        # check here for nn.Softmax doc: https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html
        column = 1
        self.activation_func_3 = nn.Softmax(dim=column)
        self.learning_rate = config["hyperparameters"]["learning_rate"]
        self.max_epoch = config["hyperparameters"]["max_epoch"]

    # it defines the forward propagation function for input x
    # this function will calculate the output layer by layer

    def forward(self, x):
        """Forward propagation"""
        # hidden layer embeddings
        h1 = self.activation_func_1(self.fc_layer_1(x))
        h2 = self.activation_func_2(self.fc_layer_2(h1))
        # outout layer result
        # self.fc_layer_2(h) will be a nx2 tensor
        # n (denotes the input instance number): 0th dimension; 2 (denotes the class number): 1st dimension
        # we do softmax along dim=1 to get the normalized classification probability distributions for each instance
        y_pred = self.activation_func_3(self.fc_layer_3(h2))
        return y_pred

    # backward error propagation will be implemented by pytorch automatically
    # so we don't need to define the error backpropagation function here

    def train_model(self, X, y):
        # check here for the torch.optim doc: https://pytorch.org/docs/stable/optim.html
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        # check here for the nn.CrossEntropyLoss doc: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        loss_function = nn.CrossEntropyLoss()
        # for training accuracy investigation purpose

        # it will be an iterative gradient updating process
        # we don't do mini-batch, we use the whole input as one batch
        # you can try to split X and y into smaller-sized batches by yourself
        for epoch in range(
            self.max_epoch
        ):  # you can do an early stop if self.max_epoch is too much...
            # get the output, we need to covert X into torch.tensor so pytorch algorithm can operate on it
            y_pred = self.forward(torch.FloatTensor(np.array(X)))
            # convert y to torch.tensor as well
            y_true = torch.LongTensor(np.array(y))
            # calculate the training loss
            train_loss = loss_function(y_pred, y_true)

            # check here for the gradient init doc: https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html
            optimizer.zero_grad()
            # check here for the loss.backward doc: https://pytorch.org/docs/stable/generated/torch.Tensor.backward.html
            # do the error backpropagation to calculate the gradients
            train_loss.backward()
            # check here for the opti.step doc: https://pytorch.org/docs/stable/optim.html
            # update the variables according to the optimizer and the gradients calculated by the above loss.backward function
            optimizer.step()

            if self.notification_manager is not None:
                for _, m in self.batch_metrics.items():
                    m.update(y_true, y_pred.max(1)[1])

                self.notification_manager.notify(
                    MLEventType("method"),
                    ClassificationNotification(
                        epoch=epoch,
                        loss=train_loss.item(),
                        **{n: m.compute() for n, m in self.batch_metrics.items()},
                    ),
                )

            if epoch % 100 == 0 and self.batch_metrics is not None:
                print(
                    "Epoch:",
                    epoch,
                    "Loss:",
                    train_loss.item(),
                    "Metrics:",
                    {k: m.compute().item() for k, m in self.batch_metrics.items()},
                )

    def test(self, X):
        # do the testing, and result the result
        y_pred = self.forward(torch.FloatTensor(np.array(X)))
        # convert the probability distributions to the corresponding labels
        # instances will get the labels corresponding to the largest probability
        return y_pred.max(1)[1]

    def run(self):
        print("method running...")
        print("--start training...")
        self.train_model(self.data["train"]["X"], self.data["train"]["y"])
        print("--start testing...")
        pred_y = self.test(self.data["test"]["X"])
        return {"pred_y": pred_y, "true_y": self.data["test"]["y"]}
