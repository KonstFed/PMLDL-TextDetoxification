import argparse

from torch.utils.data import random_split, DataLoader
from torch import nn
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import lightning.pytorch as pl
import numpy as np
import torch
import random
from addict import Dict
import yaml

from .models import build_model
from .data.dataset import build_dataset
from .preprocessing import build_preprocessing


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


def get_args():
    parser = argparse.ArgumentParser(
        description="Training CLI for text Detoxification",
    )
    parser.add_argument(
        "-c", "--config", help="path to config model to train", required=True
    )
    args = parser.parse_args()
    with open(args.config, "r") as file:
        config = yaml.safe_load(file)
    return Dict(config)


def train(config):
    # torch.set_float32_matmul_precision('medium' | 'high')
    set_seed(config.training.seed)
    model = build_model(config.model, config.training)
    preprocessing = build_preprocessing(config.preprocessing)
    dataset = build_dataset(config.training.dataset, **preprocessing)
    # dataset.save("models/preprocessing/toxic_dataset")

    train_data, val_data, test_data = random_split(
        dataset, config.training.train_val_test_ratio
    )
    train_loader = DataLoader(
        train_data,
        collate_fn=model.collate_batch,
        shuffle=True,
        **config.training.dataloader
    )
    val_loader = DataLoader(
        val_data,
        collate_fn=model.collate_batch,
        shuffle=False,
        **config.training.dataloader
    )
    test_loader = DataLoader(
        test_data,
        collate_fn=model.collate_batch,
        shuffle=False,
        **config.training.dataloader
    )
    trainer = pl.Trainer(
        callbacks=[EarlyStopping(monitor="val loss", mode="min")],
        **config.training.trainer_args
    )
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.test(model, dataloaders=test_loader)

    if "save_path" in config.training:
        model.save(config.training.save_path)

if __name__ == "__main__":
    config = get_args()
    train(config)
