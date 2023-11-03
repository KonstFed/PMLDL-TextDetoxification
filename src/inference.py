import argparse
import random

import numpy as np
import torch
from addict import Dict
import yaml
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from .models import build_model
from .preprocessing import build_preprocessing
from .preprocessing.tokenizers import Tokenizer
from .preprocessing.vocabulars import Text2Vector


def get_args():
    parser = argparse.ArgumentParser(
        description="Training CLI for text Detoxification",
    )
    parser.add_argument(
        "-c", "--config", help="path to config model to train", required=True
    )
    parser.add_argument("text", help="Text to detoxify")

    args = parser.parse_args()
    with open(args.config, "r") as file:
        config = yaml.safe_load(file)
    return Dict(config), args.text


class ToxicClassificationPipeline:
    def __init__(self, config) -> None:
        preprocessing = build_preprocessing(config.preprocessing)
        self.tokenizer, self.text2vec = (
            preprocessing["tokenizer"],
            preprocessing["text2vector"],
        )
        # self.text2vec = self.text2vec.load(config.preprocessing.text2vector.load_path)
        self.model = build_model(config.model, config.training)
        self.model.eval()

    def forward(self, input: str) -> float:
        tokens = self.tokenizer.forward([input])[0]
        input_vector = self.text2vec.forward(tokens)
        norm = np.linalg.norm(input_vector)
        if norm != 0:
            input_vector /= norm
        with torch.no_grad():
            input_vector = torch.tensor(input_vector, dtype=torch.float32)
            return self.model(input_vector).detach()


class BertPipeline:
    def __init__(self, config) -> None:
        self.tokenizer: Tokenizer = build_preprocessing(config.preprocessing)[
            "tokenizer"
        ]
        self.model = build_model(config.model, config.training)
        self.model.eval()

    def forward(self, input: str) -> float:
        tokens = self.tokenizer.forward([input])

        out = self.model(tokens)
        p = torch.nn.functional.sigmoid(out["logits"].detach())
        return p


class ParaphrasingTransformerPipeline:
    def __init__(self, config) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(config.model.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(config.model.pretrained_path)
        self.config = config
        self._tokenizer_args = {
            k: v
            for k, v in config.preprocessing.tokenizer.items()
            if k not in ["name", "model_name"]
        }
        self.model.eval()

    def forward(self, input: str) -> str:
        tokens = self.tokenizer(input, return_tensors="pt", **self._tokenizer_args)
        # tokens = {k: torch.tensor(v) for k, v in tokens.items()}
        out = self.model.generate(
            input_ids=tokens["input_ids"],
            attention_mask=tokens["attention_mask"],
            max_length=50,
            num_return_sequences=1,
        )
        preds = [
            self.tokenizer.decode(
                gen_id, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            for gen_id in out
        ]

        return preds


def inference(config, input: str):
    if config.model.name in ["DistilBert"]:
        pipeline = BertPipeline(config)
    elif config.model.name == "T5model":
        pipeline = ParaphrasingTransformerPipeline(config)
    else:
        pipeline = ToxicClassificationPipeline(config)
    return pipeline.forward(input)
    # preprocessing = build_preprocessing(config.preprocessing)
    # # text2vec = text2vec.load(config.preprocessing.text2vector.load_path)
    # tokens = tokenizer.forward([input])[0]
    # print(tokens)
    # input_vector = text2vec.forward(tokens)
    # print(input_vector.sum())
    # with torch.no_grad():
    #     input_vector = torch.tensor(input_vector, dtype=torch.float32)

    #     model = build_model(config.model, config.training)
    #     model.eval()
    #     # dataset.save("models/preprocessing/toxic_dataset")
    #     return model(input_vector)


if __name__ == "__main__":
    config, input = get_args()
    print(inference(config, input))
