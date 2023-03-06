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
import re
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

        data = []
    
        filePath = self.dataset_source_folder_path + self.dataset_source_file_name
        filePathClean = self.dataset_source_folder_path + "data_clean.csv"

        print("\nloading data...\n")
        df = pd.read_csv(filePath, header=None, names=range(7))
        #df = df.apply(lambda row: ' '.join(row.values.astype(str)), axis=1)

        # remove ID column from dataset
        df = df.iloc[:, 1] 

        # drop rows that have links 
        http_rows = df[df[:].str.contains('http', case=False)]
        df = df.drop(http_rows.index)

        # drop rows that contain parentheses
        paren_rows = df[df[:].str.contains('\(|\)', regex=True)]
        df = df.drop(paren_rows.index)

        # drop any rows that contain '\r'
        df[:] = df[:].replace(to_replace=r'.*\r.*\n?', value='', regex=True)

        # drop rows that contain '&g'
        df[:] = df[:].replace(to_replace=r'.*&g.*\n?', value='', regex=True)

        # remove sequences which start with 'nan'
        # df[:] = df[:].replace(to_replace=r'nan.*', value='', regex=True)

        # remove characters
        df[:] = df[:].replace(to_replace = r"[\\/()\[\]!.?*-\:#&^%;><_~`]", value='', regex=True)

        # remove digits
        df[:] = df[:].replace(to_replace = r'\d', value='', regex=True)

        # append <EOS> to each line
        # df[:] = df[:] + "eos"
        
        # write to file
        print("\nwriting 'clean' data...\n")
        df.to_csv(filePathClean, index=False, header=False)
        
        # fill data list
        with open(filePathClean, "r") as f:
            for line in f:
                cleanerLine = re.sub(r'[\'"]', '', line)              # remove all double or single quotations
                #cleanerLine = re.sub(r'\s{2,}', ' ', line.strip())    # remove extra white spaces
                data.append(cleanerLine)
        
        #convert to lowercase
        data = [w.lower() for w in data]

        for line in data:
            print(line)

        return data


class Tokenizer(ABC):
    @abstractmethod
    def stoi(self, s: str) -> int:
        ...

    @abstractmethod
    def itos(self, i: int) -> str:
        ...

    @abstractmethod
    def vocab_characters(self) -> list[str]:
        ...

    @abstractmethod
    def dataset(self, type: Literal["test", "train", "validation"]) -> NLPDataset:
        ...


class TextTokenizer(Tokenizer):

    words: list[str]
    blocksize: int

    def __init__(self, loader: dataset, extra_keys: Optional[list[str]] = None, block_sz: int = 3):
        self.sentences: list[str] = loader.load()
        self.blocksize = block_sz
        random.shuffle(self.sentences)
        self.idx_1 = int(len(self.sentences) * 0.8)
        self.idx_2 = int(len(self.sentences) * 0.9)

        self.chars: list[str] = sorted(list(set("".join(self.sentences))))
        self.stoi_dict: dict[str, int] = {c: i for i, c in enumerate(self.chars)}
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

    def vocab_characters(self) -> list[str]:
        return self.chars

    def dataset(self, type: Literal["test", "train", "validation"]) -> NLPDataset:

        X: list[list[int]] = []
        Y: list[list[int]] = []

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

        for word in self.sentences[l:r]:
            context: list[int] = [len(self.vocab_characters()) for _ in range(self.blocksize)]
            for c in word:
                one_hot = [0 for _ in range(len(self.vocab_characters()) + 1)]
                ix = self.stoi(c)
                one_hot[ix] = 1
                X.append(context)
                Y.append(one_hot)
                context = context[1:] + [ix]
        return NLPDataset(torch.tensor(X), torch.tensor(Y))