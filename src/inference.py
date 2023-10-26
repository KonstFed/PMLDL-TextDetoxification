import argparse
import random

import numpy as np
import torch
from addict import Dict
import yaml

from models import build_model
from preprocessing import build_preprocessing
from preprocessing.tokenizers import Tokenizer
from preprocessing.vocabulars import Text2Vector

def get_args():
    parser = argparse.ArgumentParser(
        description="Training CLI for text Detoxification",
    )
    parser.add_argument(
        "-c", "--config", help="path to config model to train", required=True
    )
    parser.add_argument(
        'text', help="Text to detoxify"
    )

    args = parser.parse_args()
    with open(args.config, "r") as file:
        config = yaml.safe_load(file)
    return Dict(config), args.text


def inference(config, input: str):
    tokenizer, text2vec = build_preprocessing(config.preprocessing)
    text2vec = text2vec.load(config.preprocessing.text2vector.load_path)
    tokens = tokenizer.forward([input])[0]
    print(tokens)
    input_vector = text2vec.forward(tokens)
    print(input_vector.sum())
    with torch.no_grad():
        input_vector = torch.tensor(input_vector, dtype=torch.float32)

        model = build_model(config.model, config.training)
        model.eval()
        # dataset.save("models/preprocessing/toxic_dataset")
        return model(input_vector)


if __name__ == "__main__":
    config, input = get_args()
    print(inference(config, input))
