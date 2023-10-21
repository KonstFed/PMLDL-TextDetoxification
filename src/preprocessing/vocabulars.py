from gensim.models import Word2Vec

def get_word2vec(texts: list[list[str]], args=None):
    if args is None:
        args = {
            "min_count": 1,
            "vector_size": 500,
            "window": 5,
            "sg": 0
        }
    model = Word2Vec(**args)
    model.build_vocab(texts, progress_per=1000)
    model.train(texts, total_examples=model.corpus_count, report_delay=1, epochs=model.epochs)
    return model