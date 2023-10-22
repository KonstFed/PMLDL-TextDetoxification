import csv
import pickle

from torch.utils.data import Dataset
from preprocessing.tokenizers import NLTK_tokenizer
from preprocessing.vocabulars import get_word2vec
import gensim


class CachedDataset:
    @classmethod
    def load(path: str):
        raise NotImplementedError

    def save(self, path: str):
        raise NotImplementedError


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

    def save(self, path) -> None:
        raise NotImplementedError

    def load(self, path) -> None:
        raise NotImplementedError


class ToxicityLevelDataset(CSVDataset, CachedDataset):
    def __init__(self, data_path: str, nltk_args, word2vec_args, verbose=True) -> None:
        """Dataset for predicting toxicity level of sentence for filtered.tsv data.

        Args:
            data_path (str): path to .tsv data
            verbose (bool, optional): if True print some logs and tqdm. Defaults to True.
        """
        super().__init__(data_path)
        self._toxic_level = []
        self._texts = []
        for data_row in self._data:
            self._toxic_level.append(float(data_row[3]))
            self._texts.append(data_row[0])
            self._toxic_level.append(float(data_row[4]))
            self._texts.append(data_row[1])

        self.tokenizer = NLTK_tokenizer(**nltk_args)
        self._tokenized_texts = self.tokenizer.forward(self._texts, verbose=verbose)

        if verbose:
            print("Begin training of Word2vec")
        self.to_emb = get_word2vec(self._tokenized_texts, args=word2vec_args)
        if verbose:
            print("Done training Word2vec")
        # for sentence in tokenized_texts:
        #     self._emb_texts.append(list(map(lambda x: self.to_emb.wv[x], sentence)))

    def __getitem__(self, index):
        texts_emb = [self.to_emb.wv[x] for x in self._tokenized_texts[index]]
        return texts_emb, self._toxic_level[index]

    def __len__(self):
        return len(self._toxic_level)

    def save(self, path) -> None:
        data2save = (self._tokenized_texts, self._toxic_level)
        with open(path + "/texts.obj", "wb") as f:
            pickle.dump(data2save, f)
        self.to_emb.save(path + "/word2vec.model")

    @classmethod
    def load(cls, path) -> None:
        self = cls.__new__(cls)
        with open(path + "/texts.obj", "rb") as f:
            self._tokenized_texts, self._toxic_level = pickle.load(f)
        self.to_emb = gensim.models.Word2Vec.load(path + "/word2vec.model")
        return self


def build_dataset(dataset_config: dict):
    if "load_path" in dataset_config:
        dataset = eval(f"{dataset_config.name}.load")(dataset_config.load_path)
    else:
        dataset_params = {
            i: dataset_config[i]
            for i in dataset_config
            if i not in ["name", "save_path"]
        }
        dataset: CachedDataset = eval(f"{dataset_config.name}")(**dataset_params)
        if "save_path" in dataset_config:
            dataset.save(dataset_config.save_path)
    return dataset


if __name__ == "__main__":
    dataset = CSVDataset("data/filtered.tsv")
    print(dataset.data)
    print("Aboba")