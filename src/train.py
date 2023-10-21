from torch.utils.data import random_split, DataLoader
from torch import nn
import lightning.pytorch as pl
import numpy as np
import torch
import random

from models.toxicity_regression.model import SimpleToxicClassification
from data.dataset import ToxicityLevelDataset

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

def train():
    model = SimpleToxicClassification(500, nn.BCELoss())
    # dataset = ToxicityLevelDataset("data/filtered.tsv", num_workers=4)
    # dataset.save("models/preprocessing/toxic_dataset")

    # need to check 508263
    dataset = ToxicityLevelDataset.load("models/preprocessing/toxic_dataset")

    train_data, test_data = random_split(dataset, [0.7, 0.3])
    test_data, val_data = random_split(test_data, [0.5, 0.5])
    train_loader = DataLoader(train_data, collate_fn=SimpleToxicClassification.collate_batch, batch_size=16,  shuffle=False, num_workers=1)
    val_loader = DataLoader(val_data, collate_fn=SimpleToxicClassification.collate_batch, batch_size=16,  shuffle=False, num_workers=1)
    trainer = pl.Trainer(max_epochs=2)
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

set_seed(10)
train()