from code.base_class.setting import setting
from typing import Tuple

import torch


class Setting_Equal_Sampling_Split(setting):
    def load_run_save_evaluate(self):

        loaded_data = self._dataset.load()

        (
            loaded_data["train_test_val"]["train_idx"],
            loaded_data["train_test_val"]["test_idx"],
            loaded_data["train_test_val"]["val_idx"],
        ) = self.generate_split_indices(loaded_data)

        self._method.to(self.device)

        self._method.data = loaded_data

        # self._method.training_loader = training_dataloader
        # self._method.testing_loader = testing_dataloader

        output_result = self._method.run()

        self._result.data = output_result
        self._result.save()

        self._artifacts.serialize()

        self._evaluate.data = output_result

        return self._evaluate.evaluate(), None

    #
    def generate_split_indices(
        self, data
    ) -> Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor]:
        labels = data["graph"]["y"]
        num_classes = 7
        train_size = self.params["train_size"]
        test_size = self.params["test_size"]
        train_counts, test_counts = [0] * num_classes, [0] * num_classes
        train_indices, test_indices, val_indices = [], [], []
        for i in range(labels.size()[0]):
            label = labels[i]
            if (
                train_counts[label] < train_size / num_classes
            ):  # Room in training set for that class
                train_indices.append(i)
                train_counts[label] += 1
            elif test_counts[label] < test_size / num_classes:  # Room in testing set for that class
                test_indices.append(i)
                test_counts[label] += 1
            else:
                val_indices.append(i)  # validation set used as an overflow
        return (
            torch.LongTensor(train_indices),
            torch.LongTensor(test_indices),
            torch.LongTensor(val_indices),
        )
