import unittest
from code.base_class.dataset import datasetConfig
from code.stage_4_code.JokeGenerationDataset import (
    JokeFilePreprocess,
    NLPDataset,
    TextTokenizer,
)

"""
    To run this class, perform `python -m code.tests.test_example`
    in the command line.
"""
import torch
from torch.utils.data import DataLoader


class TestNLPDataset(unittest.TestCase):
    def test_assert_error(self):

        d_config = datasetConfig(
            {
                "name": "jokes",
                "description": "...data description...",
                "source_folder_path": "data/stage_4_data/text_generation/",
                "source_file_name": "data",
                "device": "cpu",
            }
        )
        num_character_context = 10
        num_batch_size = 32

        loader = JokeFilePreprocess(d_config)
        tokenizer = TextTokenizer(loader, ["<n>"], num_character_context)

        train_dataset: NLPDataset = tokenizer.dataset("train")

        train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)

        max = 0
        for batch in train_dataloader:

            if torch.max(batch["context"]) > max:
                max = torch.max(batch["context"])

            self.assertLessEqual(
                batch["context"].shape, torch.Size([num_batch_size, num_character_context])
            )


if __name__ == "__main__":
    unittest.main()
