"""
Concrete SettingModule class for a specific experimental SettingModule
"""
from code.base_class.setting import setting
from code.stage_3_code.Dataset_Loader import LoadedDataset

import torch
from torch.utils.data import DataLoader
from torchvision import transforms  # type: ignore

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD


class Setting_Train_Test_Split_ORL(setting):
    def load_run_save_evaluate(self):

        # load dataset
        loaded_data = self._dataset.load()

        self._method.to(self.device)

        # run MethodModule
        self._method.data = {
            "train": {"X": loaded_data["X_train"], "y": loaded_data["y_train"]},
            "test": {"X": loaded_data["X_test"], "y": loaded_data["y_test"]},
        }
        loaded_data["y_train"] = torch.sub(loaded_data["y_train"], 1)
        loaded_data["y_test"] = torch.sub(loaded_data["y_test"], 1)

        transformations = transforms.Compose(
            [
                transforms.ToPILImage(),
                # Randomly rotate some images by 20 degrees
                transforms.RandomRotation(20),
                # Randomly horizontal flip the images
                transforms.RandomHorizontalFlip(0.1),
                # Randomly adjust color jitter of the images
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                transforms.RandomAdjustSharpness(
                    sharpness_factor=2, p=0.1
                ),  # Randomly adjust sharpness
                # transforms.Resize(32),
                transforms.ToTensor(),
                transforms.RandomErasing(p=0.75, scale=(0.02, 0.1), value=1.0, inplace=False),
                # These values are mostly used by researchers as found to very useful in fast convergence
                transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261]),
                transforms.Grayscale(),
            ]
        )

        training_dataset = LoadedDataset(
            loaded_data["X_train"], loaded_data["y_train"], transform=transformations
        )
        testing_dataset = LoadedDataset(
            loaded_data["X_test"], loaded_data["y_test"], transform=transformations
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
