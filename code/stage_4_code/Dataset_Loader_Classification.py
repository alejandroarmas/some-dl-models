import os
import re
import string
from code.base_class.dataset import dataset, datasetConfig
from code.lib.notifier import DatasetNotifier
from itertools import islice
from typing import Optional

import numpy as np
import torchtext  # type: ignore
from torch.utils.data import Dataset


class Classififcation_Dataset(Dataset):
    def __init__(self, X, y, base_dir: str):
        self.base_dir = base_dir
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        contents = self.X[idx]
        label = self.y[idx]

        return {"contents": contents, "label": label}


class Classification_Loader(dataset):
    def __init__(self, config: datasetConfig, manager: Optional[DatasetNotifier] = None):
        super().__init__(config, manager)
        self.train_pos_dir_path = f"{self.dataset_source_folder_path}/train/pos/"
        self.train_neg_dir_path = f"{self.dataset_source_folder_path}/train/neg/"
        self.test_pos_dir_path = f"{self.dataset_source_folder_path}/test/pos/"
        self.test_neg_dir_path = f"{self.dataset_source_folder_path}/test/neg/"
        self.translator = str.maketrans("", "", string.punctuation)

    def preprocess(self, inputText: str) -> str:
        # remove punctuation
        # print(inputText, type(inputText))
        inputText = inputText.lower()
        cleaner = re.compile("<.*?>")
        inputText = re.sub(cleaner, " ", inputText)
        # TODO: remove HTML tags (br)
        return inputText.translate(self.translator)

    # Loads Train and Test sets from disk, with X being filepaths from the base directory and Y being a one hot encoded
    # label (neg pos)
    def load(self, cutoff_value: Optional[int] = None):
        train_pos_gen = (
            f"/train/pos/{f}"
            for f in os.listdir(self.train_pos_dir_path)
            if os.path.isfile(os.path.join(self.train_pos_dir_path, f))
        )
        train_neg_gen = (
            f"/train/neg/{f}"
            for f in os.listdir(self.train_neg_dir_path)
            if os.path.isfile(os.path.join(self.train_neg_dir_path, f))
        )
        test_pos_gen = (
            f"/test/pos/{f}"
            for f in os.listdir(self.test_pos_dir_path)
            if os.path.isfile(os.path.join(self.test_pos_dir_path, f))
        )
        test_neg_gen = (
            f"/test/neg/{f}"
            for f in os.listdir(self.test_neg_dir_path)
            if os.path.isfile(os.path.join(self.test_neg_dir_path, f))
        )
        neg = np.array([0])
        pos = np.array([1])
        train_pos_files = list(islice(train_pos_gen, cutoff_value))
        train_neg_files = list(islice(train_neg_gen, cutoff_value))
        test_pos_files = list(islice(test_pos_gen, cutoff_value))
        test_neg_files = list(islice(test_neg_gen, cutoff_value))

        train_pos = []
        test_pos = []
        train_neg = []
        test_neg = []
        print("------loading positive files from training--------")
        for name in train_pos_files:
            with open(f"{self.dataset_source_folder_path}{name}", "r", encoding="utf-8") as f:
                train_pos.append(self.preprocess(f.read()))
        print("------loading negative files from training--------")
        for name in train_neg_files:
            with open(f"{self.dataset_source_folder_path}{name}", "r", encoding="utf-8") as f:
                train_neg.append(self.preprocess(f.read()))
        print("------loading positive files from testing--------")
        for name in test_pos_files:
            with open(f"{self.dataset_source_folder_path}{name}", "r", encoding="utf-8") as f:
                test_pos.append(self.preprocess(f.read()))
        print("------loading negative files from testing--------")
        for name in test_neg_files:
            with open(f"{self.dataset_source_folder_path}{name}", "r", encoding="utf-8") as f:
                test_neg.append(self.preprocess(f.read()))

        return {
            "X_train": train_pos + train_neg,
            "y_train": np.concatenate(
                (np.tile(pos, (len(train_pos), 1)), np.tile(neg, (len(train_neg), 1))),
                axis=0,
            ),
            "X_test": test_pos + test_neg,
            "y_test": np.concatenate(
                (np.tile(pos, (len(test_pos), 1)), np.tile(neg, (len(test_neg), 1))),
                axis=0,
            ),
        }


class Classification_Vocabulary:
    loaded_data: dict

    def __init__(self, dataset: Classification_Loader, cutoff_value: Optional[int] = None):
        self.dataset = dataset
        self.tokenizer = torchtext.data.get_tokenizer("basic_english")
        self.loaded_data = self.loaded_data = self.dataset.load(cutoff_value)
        self.vocab = torchtext.vocab.build_vocab_from_iterator(
            [self.yield_tokens(self.loaded_data["X_train"])],
            specials=["<unk>", "<pad>"],
            min_freq=3,
        )
        self.vocab.set_default_index(self.vocab.__getitem__("<unk>"))
        print("-------done building vocab--------")
        print(f"{self.vocab.__len__()=}")

    def yield_tokens(self, data):
        for document in data:
            for token in self.tokenizer(document):
                yield token

    def get_loaded_data(self):
        return self.loaded_data

    def get_vocab(self):
        return self.vocab
