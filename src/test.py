import argparse
import os

from addict import Dict
import yaml
from tqdm import trange
import csv
import torch

from .inference import ParaphrasingTransformerPipeline
from .models import build_model
from .preprocessing import build_preprocessing
from .data.dataset import build_dataset

def get_args():
    parser = argparse.ArgumentParser(
        description="Training CLI for text Detoxification",
    )
    parser.add_argument(
        "-c", "--config", help="path to config model to train", required=True
    )
    parser.add_argument(
        "-o", "--output", help="path to folder where put logs", required=False, default="."
    )
    parser.add_argument(
        "-b", "--batch-size", help="batch size for input", default=32, type=int
    )
    parser.add_argument("path", help="Path to file to process")

    args = parser.parse_args()
    with open(args.config, "r") as file:
        config = yaml.safe_load(file)
    return Dict(config), args.path, args.output, args.batch_size

def _load_data(path) -> list[str]:
    texts = []
    with open(path, "r") as file:
        reader = csv.reader(file, delimiter='\t')
        header = next(reader)
        for row in reader:
            texts.append(row[1])
    return texts


def test_transformer(config, test_data: list[str], save_folder: str, batch_size):
    pipeline = ParaphrasingTransformerPipeline(config)
    _out_path = os.path.join(save_folder, "output.tsv")
    with open(_out_path, "w", newline='') as file:
        writer = csv.writer(file, delimiter="\t")
        writer.writerow(["reference", "translation"])
        for idx in trange(0, len(test_data), batch_size):
            output = pipeline.forward(test_data[idx: idx + batch_size])
            writer.writerows([(test_data[idx + i], output[i]) for i in range(len(output))])
    build_model(config.model, config.training)


if __name__ == "__main__":
    config, path, folder, b_size = get_args()
    data = _load_data(path)
    test_transformer(config, data, folder, b_size)