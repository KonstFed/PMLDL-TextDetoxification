from collections import Counter

import numpy as np
from gensim.models import Word2Vec

from .utils import Cached

def get_word2vec(texts: list[list[str]], args=None):
    if args is None:
        args = {"min_count": 1, "vector_size": 500, "window": 5, "sg": 0}
    model = Word2Vec(**args)
    model.build_vocab(texts, progress_per=1000)
    model.train(
        texts, total_examples=model.corpus_count, report_delay=1, epochs=model.epochs
    )
    return model


class Text2Vector(Cached):
    ready = False
    def __init__(self, load_path: str = None, save_path: str = None) -> None:
        if load_path is not None:
            self = Text2Vector.load(load_path)
        self.save_path = save_path

    def build(self, data) -> None:
        self.ready = True
        self.save(self.save_path)

    @classmethod
    def load(cls, path: str):
        object: Text2Vector = super().load(path)
        object.ready = True
        object.save_path = None
        return object

    def forward(self, text: list[str]) -> np.ndarray:
        raise NotImplementedError


class BoW(Text2Vector):
    def __init__(self, size: int, load_path: str = None, save_path: str = None) -> None:
        super().__init__(load_path, save_path)
        self.size = size
        self.counter = Counter()

    def build(self, data: list[list[str]]) -> None:
        for sentence in data:
            for word in sentence:
                self.counter[word] += 1

        t = self.counter.most_common(self.size)
        self.mapping = {x[0]: idx for idx, x in enumerate(t)}
        if self.save_path is not None:
            self.save(self.save_path)
        self.ready = True
    
    def forward(self, text: list[str]) -> np.ndarray:
        out = np.zeros(self.size)
        for word in text:
            idx = self.mapping.get(word, -1)
            if idx != -1:
                out[idx] += 1
        return out
