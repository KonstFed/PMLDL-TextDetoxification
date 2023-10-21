import pickle

from src.preprocessing.tokenizers import NLTK_tokenizer, get_t5_tokenizer
from src.data.dataset import ToxicityLevelDataset
from src.preprocessing.vocabulars import get_word2vec

dataset = ToxicityLevelDataset("data/filtered.tsv")
tokenizer = NLTK_tokenizer()
texts = tokenizer.forward(dataset._texts, verbose=True)
with open("models/preprocessing/tokens.obj", "wb") as f:
    pickle.dump(texts, f)

word2vec = get_word2vec(texts)
word2vec.save("models/preprocessing/word2vec.model")
print(texts[0][0])
print(word2vec.wv[texts[0][0]])