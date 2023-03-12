"""
Concrete SettingModule class for a specific experimental SettingModule
"""

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.setting import setting
from code.stage_4_code.JokeGenerationDataset import NLPDataset, TextTokenizer

import torch
from torch.utils.data import DataLoader


class Setting_Train_Test_Split(setting):
    def load_run_save_evaluate(self):

        # load dataset

        self._method.to(self.device)

        num_character_context = self._method.num_character_context

        tokenizer = TextTokenizer(self._dataset, num_character_context)

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

        text = "What did the turkey say about the television program from the"

        def predict(tokenizer: TextTokenizer, model, text, next_words=100):
            model.eval()

            num_words = model.num_character_context

            words: list[str] = text.split(" ")

            word_set = set(tokenizer.vocab.vocabulary())

            for word in words:
                assert word in word_set

            state_h, state_c = model.init_state(num_samples=32)

            for i in range(0, next_words):

                j = i + num_words
                context = words[i:j]
                print(context)
                input = torch.tensor(
                    [[tokenizer.vocab.stoi(word) for word in context] for _ in range(32)]
                )
                y_pred, (state_h, state_c) = model(input, state_h, state_c)
                generated_word = tokenizer.vocab.itos(int(torch.argmax(y_pred).item()))
                print(generated_word)
                words.append(generated_word)
            return words

        print(f"{predict(tokenizer, self._method, text)=}")

        return self._evaluate.evaluate(), None
