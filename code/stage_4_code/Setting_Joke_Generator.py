"""
Concrete SettingModule class for a specific experimental SettingModule
"""

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.setting import setting
from code.stage_4_code.JokeGenerationDataset import NLPDataset, TextTokenizer

from torch.utils.data import DataLoader


class Setting_Train_Test_Split(setting):
    def load_run_save_evaluate(self):

        # load dataset

        self._method.to(self.device)

        num_character_context = self._method.num_character_context

        tokenizer = TextTokenizer(self._dataset, ["<Pad>"], num_character_context)

        train_dataset: NLPDataset = tokenizer.dataset("test")
        testing_dataset: NLPDataset = tokenizer.dataset("train")

        training_dataloader = DataLoader(
            train_dataset,
            batch_size=self._method.batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True,
        )
        testing_dataloader = DataLoader(
            testing_dataset,
            batch_size=self._method.batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True,
        )

        self._method.training_loader = training_dataloader
        self._method.testing_loader = testing_dataloader

        learned_result = self._method.run()

        # save raw ResultModule
        self._result.data = learned_result
        self._result.save()

        self._evaluate.data = learned_result

        return self._evaluate.evaluate(), None
