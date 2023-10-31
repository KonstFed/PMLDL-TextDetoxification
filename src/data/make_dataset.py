import os
import pandas as pd

def train_test_split(path: str, split_proportion: float):
    folder = "/".join(path.split("/")[:-1])

    df = pd.read_csv(path, sep='\t')
    total_rows = df.shape[0]
    split_index = int(total_rows * split_proportion)
    df = df.sample(frac=1, random_state=38)
    df_train = df[:split_index]
    df_test = df[split_index:]
    df_train.to_csv(folder + "/train.tsv", sep='\t', index=False)
    df_test.to_csv(folder + "/test.tsv", sep='\t', index=False)



if __name__ == "__main__":
    os.makedirs("models/preprocessing/toxic_dataset", exist_ok=True)

    train_test_split("data/filtered.tsv", split_proportion=0.8)