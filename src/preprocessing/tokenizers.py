from multiprocessing import Pool
import re

import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from tqdm import tqdm

from .utils import Cached


# from transformers import AutoTokenizer
class Tokenizer(Cached):
    def forward(
        self,
        data: list[str],
        verbose=True
    ) -> list[list[str]]:
        """Tokenize list of sentence

        Args:
            data (list[str]): list of sentence
            verbose (bool, optional): Need to print additional logs and tqdm bar. Defaults to True.

        Returns:
            list[list[str]]: list of sentences. Where every sentence is list of words, which are string
        """


def _download_if_non_existent(res_path, res_name):
    try:
        nltk.data.find(res_path)
    except LookupError:
        print(f"resource {res_name} not found in {res_path}")
        print("Downloading now ...")
        nltk.download(res_name)


class NLTK_tokenizer(Tokenizer):
    def __init__(self, num_workers: int = 1) -> None:
        _download_if_non_existent("corpora/stopwords", "stopwords")
        self.rm_symb_pattern_ = r"[();.\/\\$%,!&?*'\"\’—–~…\”-]"
        self.stopwords = stopwords.words("english")
        self._stemmer = PorterStemmer()
        self.num_workers = num_workers

    def _tokenize_sentence(self, sentence: str) -> list[str]:
        row: str = re.sub(self.rm_symb_pattern_, " ", sentence)
        row = re.sub(r" +", " ", row).strip()
        tokens = row.split(" ")
        tokens = list(filter(lambda x: x not in self.stopwords, tokens))
        tokens = list(map(self._stemmer.stem, tokens))
        return tokens

    def forward(self, data: list[str], verbose=False) -> list[list[str]]:
        with Pool(self.num_workers) as p:
            if verbose:
                out = list(tqdm(p.imap(self._tokenize_sentence, data), total=len(data)))
            else:
                out = list(p.imap(self._tokenize_sentence, data))

        # out = list(filter(lambda x: len(x) > 0, out))
        return out


if __name__ == "__main__":
    tokenizer = NLTK_tokenizer()
    print(tokenizer.forward(["what are you doing?"]))
