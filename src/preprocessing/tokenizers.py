from multiprocessing import Pool
import re

import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from tqdm import tqdm

# from transformers import AutoTokenizer


def _download_if_non_existent(res_path, res_name):
    try:
        nltk.data.find(res_path)
    except LookupError:
        print(f"resource {res_name} not found in {res_path}")
        print("Downloading now ...")
        nltk.download(res_name)


class NLTK_tokenizer:
    def __init__(self, num_workers: int = 1) -> None:
        _download_if_non_existent("corpora/stopwords", "stopwords")
        self.rm_symb_pattern_ = r"[();.\/\\$%,!&?*]"
        self.stopwords = stopwords.words("english")
        self._stemmer = PorterStemmer()
        self.num_workers = num_workers

    def _tokenize_sentence(self, sentence: str) -> list[str]:
        row: str = re.sub(self.rm_symb_pattern_, " ", sentence)
        row = re.sub(r" +", " ", row).strip()
        tokens = row.split(" ")
        # tokens = list(filter(lambda x: x not in self.stopwords, tokens))
        tokens = list(map(self._stemmer.stem, tokens))
        return tokens

    def forward(self, data: list[str], verbose=False) -> list[list[str]]:
        with Pool(self.num_workers) as p:
            if verbose:
                out = list(tqdm(p.imap(self._tokenize_sentence, data), total=len(data)))
            else:
                out = list(p.imap(self._tokenize_sentence, data))
        return out


def get_t5_tokenizer(checkpoint: str = "t5-small"):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    return tokenizer


if __name__ == "__main__":
    tokenizer = NLTK_tokenizer()
    print(tokenizer.forward(["what are you doing?"]))