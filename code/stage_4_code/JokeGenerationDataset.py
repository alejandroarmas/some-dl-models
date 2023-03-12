"""
Concrete IO class for a specific dataset
"""

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

import random
from abc import ABC, abstractmethod
from code.base_class.dataset import dataset, datasetConfig
from code.lib.notifier import DatasetNotifier
from typing import Callable, Literal, Optional

import pandas as pd
import torch
from torch.utils.data import Dataset


class NLPDataset(Dataset):
    """NLPDataset is a specialization of the Utility provided in Pytorch to load batched
        files

    Args:
        Dataset (_type_): _description_
    """

    def __init__(self, X: torch.Tensor, y: torch.Tensor, transform=None, train=True, toByte=False):

        self.X = X
        self.y = y
        self.transform: Optional[Callable[..., torch.Tensor]] = transform
        self.train = train
        self.toByte = toByte

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):

        context: torch.Tensor = self.X[idx]
        label: torch.Tensor = self.y[idx]

        sample = {"context": context, "label": label}

        # convert tensor to bytes (if needed)
        if self.transform is not None:
            sample["context"] = (
                self.transform(sample["context"].byte())
                if self.transform and self.toByte
                else self.transform(sample["context"])
            )

        return sample


class JokeFilePreprocess(dataset):
    def __init__(self, config: datasetConfig, manager: Optional[DatasetNotifier] = None):
        super().__init__(config, manager)

    def load(self) -> list[str]:
        filePath = self.dataset_source_folder_path + self.dataset_source_file_name
        filePathClean = self.dataset_source_folder_path + "data_clean.csv"

        print("\nloading data...\n")
        joke_csv: pd.DataFrame = pd.read_csv(filePath, header=None)

        # remove ID column from dataset
        jokes: pd.Series = joke_csv.iloc[:, 1]

        # drop rows that have links
        http_rows = jokes[jokes[:].str.contains("http", case=False)]
        jokes = jokes.drop(http_rows.index)

        # drop rows that contain parentheses
        paren_rows = jokes[jokes[:].str.contains(r"\(|\)", regex=True)]
        jokes = jokes.drop(paren_rows.index)

        # drop any rows that contain '\r'
        jokes[:] = jokes[:].replace(to_replace=r".*\r.*\n?", value="", regex=True)

        # drop rows that contain '&g'
        jokes[:] = jokes[:].replace(to_replace=r".*&g.*\n?", value="", regex=True)

        # remove characters
        jokes[:] = jokes[:].replace(
            to_replace=r"[\\/()\[\]!.?*-\:#&^%;><_~`]", value="", regex=True
        )

        # remove digits
        jokes[:] = jokes[:].replace(to_replace=r"\d", value="", regex=True)

        # write to file
        print("\nwriting 'clean' data...\n")
        jokes.to_csv(filePathClean, index=False, header=False)

        data: list[str] = []

        with open(filePathClean, "r") as f:
            while line := f.readline().rstrip():
                comma_index: int = (
                    line.find(",") + 2
                )  # + 1 from the character after comma, and another + 1 for ' symbol
                end_index = len(line) - 1
                cleaned_line = line[comma_index:end_index]
                data.append(cleaned_line)

        return data


class Vocabulary(ABC):
    @abstractmethod
    def stoi(self, s: str) -> int:
        ...

    @abstractmethod
    def itos(self, i: int) -> str:
        ...

    @abstractmethod
    def vocabulary(self) -> list[str]:
        ...


class WordVocabulary(Vocabulary):
    def __init__(self, sentences: list[str], extra_keys: Optional[list[str]] = None):

        words: list[str] = sorted(list(set([word for s in sentences for word in s.split(" ")])))

        self.stoi_dict: dict[str, int] = {word: i for i, word in enumerate(words)}
        num_keys: int = len(self.stoi_dict)

        if extra_keys:
            for key in extra_keys:
                self.stoi_dict[key] = num_keys
                num_keys += 1

        self.itos_dict: dict[int, str] = {i: s for s, i in self.stoi_dict.items()}

    def stoi(self, s: str) -> int:
        return self.stoi_dict[s]

    def itos(self, i: int) -> str:
        return self.itos_dict[i]

    def vocabulary(self) -> list[str]:
        return list(self.stoi_dict.keys())


class Tokenizer(ABC):
    @abstractmethod
    def dataset(self, type: Literal["test", "train", "validation"]) -> NLPDataset:
        ...


class TextTokenizer(Tokenizer):

    sentences: list[str]
    blocksize: int
    vocab: WordVocabulary

    def __init__(self, loader: dataset, block_sz: int = 3):
        self.blocksize = block_sz
        self.sentences: list[str] = loader.load()
        self.vocab = WordVocabulary(sentences=self.sentences, extra_keys=["<End>"])

        random.shuffle(self.sentences)
        self.idx_1 = int(len(self.sentences) * 0.8)
        self.idx_2 = int(len(self.sentences) * 0.9)

    def dataset(self, type: Literal["test", "train", "validation"]) -> NLPDataset:

        X: list[list[int]] = []
        Y: list[int] = []

        match type:
            case "train":
                l: int = 0
                r: int = self.idx_1
            case "test":
                l: int = self.idx_1  # type: ignore[no-redef]
                r: int = self.idx_2  # type: ignore[no-redef]
            case "validation":
                l: int = self.idx_2  # type: ignore[no-redef]
                r: int = len(self.sentences) - 1  # type: ignore[no-redef]

        for sentence in self.sentences[l:r]:
            context: list[int] = [len(self.vocab.vocabulary()) for _ in range(self.blocksize)]
            for word in sentence.split(" "):
                ix = self.vocab.stoi(word)
                X.append(context)
                Y.append(ix)
                context = context[1:] + [ix]

        return NLPDataset(torch.tensor(X), torch.tensor(Y))
