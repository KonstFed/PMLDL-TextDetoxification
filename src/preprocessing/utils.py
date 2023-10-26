import pickle

class Cached:        
    @classmethod
    def load(cls, path: str):
        with open(path, "rb") as f:
            return pickle.load(f)

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)