"""
Concrete SettingModule class for a specific experimental SettingModule
"""

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.setting import setting
from code.stage_3_code.Dataset_Loader import LoadedDataset

from torch.utils.data import DataLoader


class Setting_Train_Test_Split(setting):
    def load_run_save_evaluate(self):

        # load dataset
        loaded_data = self._dataset.load(True)

        self._method.to(self.device)

        # run MethodModule
        self._method.data = {
            "train": {"X": loaded_data["X_train"], "y": loaded_data["y_train"]},
            "test": {"X": loaded_data["X_test"], "y": loaded_data["y_test"]},
        }

        training_dataset = LoadedDataset(
            loaded_data["X_train"],
            loaded_data["y_train"],
        )
        testing_dataset = LoadedDataset(
            loaded_data["X_test"],
            loaded_data["y_test"],
        )

        testing_dataloader = DataLoader(
            testing_dataset, batch_size=self._method.batch_size, shuffle=True, num_workers=0
        )
        training_dataloader = DataLoader(
            training_dataset, batch_size=self._method.batch_size, shuffle=True, num_workers=0
        )

        self._method.training_loader = training_dataloader
        self._method.testing_loader = testing_dataloader

        print(f'{self._method.data["train"]["X"].shape=}')
        learned_result = self._method.run()

        # save raw ResultModule
        self._result.data = learned_result
        self._result.save()

        self._artifacts.serialize()

        self._evaluate.data = learned_result

        return self._evaluate.evaluate(), None
