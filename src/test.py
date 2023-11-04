import argparse
import os

from addict import Dict
import yaml
from tqdm import trange
import csv
import torch
from sentence_transformers import SentenceTransformer

from .inference import ParaphrasingTransformerPipeline, build_pipeline
from .models import build_model
from .preprocessing import build_preprocessing
from .data.dataset import build_dataset


def get_args():
    parser = argparse.ArgumentParser(
        description="Training CLI for text Detoxification",
    )
    parser.add_argument(
        "-c", "--config", help="path to config model to test", required=True
    )
    parser.add_argument(
        "-t",
        "--toxic-config",
        help="path to config model to classify toxicity",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output",
        help="path to folder where put logs",
        required=False,
        default=".",
    )
    parser.add_argument(
        "-b", "--batch-size", help="batch size for input", default=32, type=int
    )
    parser.add_argument("path", help="Path to file to process")

    args = parser.parse_args()
    with open(args.config, "r") as file:
        config = yaml.safe_load(file)
    with open(args.toxic_config, "r") as file:
        toxic_cls_config = yaml.safe_load(file)
    return Dict(config), Dict(toxic_cls_config), args.path, args.output, args.batch_size


def _text_similarity(reference: list[str], translation: list[str]):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    ref_embs = model.encode(
        reference, show_progress_bar=True, device=device, convert_to_tensor=True
    )
    trn_embs = model.encode(
        translation, show_progress_bar=True, device=device, convert_to_tensor=True
    )
    assert len(ref_embs) == len(trn_embs)
    result = []
    print("Computing semantic similarity")
    print(ref_embs.shape)
    for i in trange(len(ref_embs)):
        similarity = torch.nn.functional.cosine_similarity(
            ref_embs[i].view(1, -1), trn_embs[i].view(1, -1)
        )
        similarity = float(similarity.detach().cpu())
        result.append(similarity)
    return result


def _load_data(path) -> list[str]:
    texts = []
    with open(path, "r") as file:
        reader = csv.reader(file, delimiter="\t")
        header = next(reader)
        for row in reader:
            texts.append(row[1])
    return texts


def _compute_toxicity(
    toxic_cls_config, reference: list[str], translation: list[str], batch_size: int
):
    cls_pipeline = build_pipeline(toxic_cls_config)
    ref_result = cls_pipeline.forward_multiple(reference, batch_size=batch_size)
    trn_result = cls_pipeline.forward_multiple(translation, batch_size=batch_size)
    ref_result = list(map(lambda x: float(x.detach()), ref_result))
    trn_result = list(map(lambda x: float(x.detach()), trn_result))

    return ref_result, trn_result


def _compute_metric(toxic_cls_config, path: str, folder: str = "."):
    references = []
    translations = []
    with open(path, "r") as file:
        reader = csv.reader(file, delimiter="\t")
        header = next(reader)
        for row in reader:
            ref, translation = row
            references.append(ref)
            translations.append(translation)

    ref_tox, trn_tox = _compute_toxicity(toxic_cls_config, references, translations, 64)
    similarities = _text_similarity(references, translations)
    _metric_path = os.path.join(folder, "metric.tsv")
    with open(_metric_path, "w") as file:
        writer = csv.writer(file, delimiter="\t")
        writer.writerow(
            ["reference", "translation", "similarity", "None", "ref_tox", "trn_tox"]
        )
        for i in range(len(similarities)):
            writer.writerow(
                (
                    references[i],
                    translations[i],
                    similarities[i],
                    "",
                    ref_tox[i],
                    trn_tox[i],
                )
            )

    score = 0
    for i in range(len(similarities)):
        score += (ref_tox[i] - trn_tox[i]) * similarities[i]
    score /= len(similarities)
    return score


def test_transformer(
    config, test_data: list[str], toxic_cls_config, save_folder: str, batch_size
):
    _out_path = os.path.join(save_folder, "output.tsv")

    # pipeline = ParaphrasingTransformerPipeline(config)
    # with open(_out_path, "w", newline='') as file:
    #     writer = csv.writer(file, delimiter="\t")
    #     writer.writerow(["reference", "translation"])
    #     for idx in trange(0, len(test_data), batch_size):
    #         output = pipeline.forward(test_data[idx: idx + batch_size])
    #         writer.writerows([(test_data[idx + i], output[i]) for i in range(len(output))])

    score = _compute_metric(toxic_cls_config, _out_path, folder=save_folder)
    print("Test score:", score)


if __name__ == "__main__":
    config, toxic_cls_config, path, folder, b_size = get_args()
    data = _load_data(path)
    test_transformer(config, data, toxic_cls_config, folder, b_size)
