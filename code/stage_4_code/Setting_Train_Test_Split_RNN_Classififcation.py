from code.base_class.setting import setting
from code.stage_4_code.Dataset_Loader_Classification import (
    Classififcation_Dataset,
)

from torch.utils.data import DataLoader


class Setting_Train_Test_Split(setting):
    def load_run_save_evaluate(self):

        loaded_data = self._dataset.load()

        self._method.to(self.device)

        self._method.data = {
            "train": {"X": loaded_data["X_train"], "y": loaded_data["y_train"]},
            "test": {"X": loaded_data["X_test"], "y": loaded_data["y_test"]},
        }

        training_set = Classififcation_Dataset(
            loaded_data["X_train"],
            loaded_data["y_train"],
            base_dir=self._dataset.dataset_source_folder_path,
        )

        testing_set = Classififcation_Dataset(
            loaded_data["X_test"],
            loaded_data["y_test"],
            base_dir=self._dataset.dataset_source_folder_path,
        )

        training_dataloader = DataLoader(
            training_set, batch_size=self._method.batch_size, shuffle=True, num_workers=0
        )

        testing_dataloader = DataLoader(
            testing_set, batch_size=self._method.batch_size, shuffle=True, num_workers=0
        )

        self._method.training_loader = training_dataloader
        self._method.testing_loader = testing_dataloader

        output_result = self._method.run()

        self._result.data = output_result
        self._result.save()

        self._artifacts.serialize()

        self._evaluate.data = output_result

        return self._evaluate.evaluate(), None
