import os
import csv
import pickle
from collections import Counter

import torch
import gensim
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset

from ..preprocessing.tokenizers import NLTK_tokenizer, Tokenizer
from ..preprocessing.vocabulars import get_word2vec, Text2Vector


class CSVDataset(Dataset):
    def __init__(self, data_path: str) -> None:
        self._data = []
        with open(data_path, "r") as f:
            reader = csv.reader(f, delimiter="\t")
            for i, row in enumerate(reader):
                if i == 0:
                    continue
                self._data.append(
                    [
                        row[1],  # reference text
                        row[2],  # translated text
                        row[3],  # similarity
                        row[5],  # reference toxicity
                        row[6],  # translation toxicity
                    ]
                )

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index):
        # TODO
        return self._data[index]


class ToxicityLevelDataset(CSVDataset):
    def __init__(
        self,
        data_path: str,
        tokenizer: Tokenizer,
        text2vector: Text2Vector,
        verbose=True,
    ) -> None:
        super().__init__(data_path)
        _toxic_level = []
        self._texts = []
        for data_row in self._data:
            _toxic_level.append(float(data_row[3]))
            self._texts.append(data_row[0])
            _toxic_level.append(float(data_row[4]))
            self._texts.append(data_row[1])

        texts = tokenizer.forward(self._texts, verbose=verbose)
        self._tokenized_texts = []
        self._toxic_level = []
        for i in range(len(texts)):
            if len(texts[i]) > 0:
                self._toxic_level.append(_toxic_level[i])
                self._tokenized_texts.append(texts[i])

        self.text2vec = text2vector
        if not self.text2vec.ready:
            self.text2vec.build(self._tokenized_texts, self._toxic_level)

    def __len__(self):
        return len(self._tokenized_texts)

    def __getitem__(self, index):
        vector_form = self.text2vec.forward(self._tokenized_texts[index])
        return vector_form, self._toxic_level[index]


class SimpleParaphrasingDataset(CSVDataset):
    def __init__(self, data_path: str, tokenizer: Tokenizer, prefix: str = "") -> None:
        super().__init__(data_path)
        self.prefix = prefix
        self.tokenizer = tokenizer

    def __getitem__(self, index):

        ref_tokens = self.tokenizer.forward(self.prefix + self._data[index][0])
        trn_tokens = self.tokenizer.forward(self.prefix + self._data[index][1])
        # print(self._data[index][0], self._data[index][1])
        # print(ref_tokens, trn_tokens)
        return ref_tokens, trn_tokens


class TransfosrmerDataset(CSVDataset):
    def __init__(self, data_path: str, tokenizer: Tokenizer) -> None:
        super().__init__(data_path)
        self.toxic_level = []
        self._texts = []
        for data_row in self._data:
            self.toxic_level.append(float(data_row[3]))
            self._texts.append(data_row[0])
            self.toxic_level.append(float(data_row[4]))
            self._texts.append(data_row[1])

        # if binary:
        #     self.toxic_level = list(map(lambda x: x > 0.5))

        self.tokenizers = tokenizer

    def __len__(self):
        return len(self.toxic_level)

    def __getitem__(self, index):
        item = self.tokenizers.forward(self._texts[index])

        item = {key: torch.tensor(val) for key, val in item.items()}
        item["labels"] = self.toxic_level[index]
        return item


class BinaryToxicityLevelDataset(ToxicityLevelDataset):
    def __init__(
        self,
        data_path: str,
        tokenizer: Tokenizer,
        text2vector: Text2Vector,
        threshold: float,
        verbose=True,
    ) -> None:
        super().__init__(data_path, tokenizer, text2vector, verbose)
        self.threshold = threshold

    def __getitem__(self, index):
        vector, toxic_level = super().__getitem__(index)
        return vector, int(toxic_level > self.threshold)


def build_dataset(dataset_config: dict, **preprocessing_args):
    dataset_params = {
        i: dataset_config[i] for i in dataset_config if i not in ["name", "save_path"]
    }
    dataset = eval(f"{dataset_config.name}")(**preprocessing_args, **dataset_params)

    return dataset


if __name__ == "__main__":
    dataset = CSVDataset("data/filtered.tsv")
    print(dataset.data)
    print("Aboba")
