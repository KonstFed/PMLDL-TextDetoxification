import argparse
import os

from addict import Dict
import yaml
from tqdm import tqdm
import csv

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
    parser.add_argument("path", help="Path to file to process")

    args = parser.parse_args()
    with open(args.config, "r") as file:
        config = yaml.safe_load(file)
    return Dict(config), args.path, args.output

def _load_data(path) -> list[str]:
    texts = []
    with open(path, "r") as file:
        reader = csv.reader(file, delimiter='\t')
        header = next(reader)
        for row in reader:
            texts.append(row[1])
    return texts

def test_transformer(config, test_data: list[str], save_folder: str):

    pipeline = ParaphrasingTransformerPipeline(config)
    _out_path = os.path.join(save_folder, "output.tsv")
    with open(_out_path, "w", newline='') as file:
        writer = csv.writer(file, delimiter="\t")
        writer.writerow(["reference", "translation"])
    
        for sentence in tqdm(test_data):
            output = pipeline.forward(sentence)
            writer.writerow([sentence, output[0]])
    build_model(config.model, config.training)


if __name__ == "__main__":
    config, path, folder = get_args()
    data = _load_data(path)
    test_transformer(config, data, folder)