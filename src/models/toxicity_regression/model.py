from typing import Any
from lightning.pytorch.utilities.types import STEP_OUTPUT
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import numpy as np

import lightning.pytorch as pl


class SimpleToxicClassification(pl.LightningModule):
    def __init__(self, input_size: int, loss):
        super().__init__()
        self.linear = nn.Sequential(nn.Linear(input_size, 300), nn.ReLU())
        self.classification = nn.Sequential(
            nn.Linear(300, 100), nn.ReLU(), nn.Linear(100, 1), nn.Sigmoid()
        )
        self.loss = loss

    def forward(self, input: list[torch.tensor]) -> torch.tensor:
        sentences_batch = input
        out = []
        for sentence in sentences_batch:
            embds = self.linear(sentence)
            emb = embds.mean(axis=0)
            out.append(emb.view(*emb.shape, 1))

        out = torch.cat(out, dim=1).permute(1, 0)
        out = self.classification(out)
        return out

    def training_step(self, batch, batch_idx):
        sentences_batch, label = batch
        out = []
        for sentence in sentences_batch:
            embds = self.linear(sentence)
            emb = embds.mean(axis=0)
            out.append(emb.view(*emb.shape, 1))

        out = torch.cat(out, dim=1).permute(1, 0)
        out = self.classification(out)
        loss = self.loss(out, label)
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=len(batch)
        )
        return loss

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        sentences_batch, label = batch
        out = []
        for sentence in sentences_batch:
            embds = self.linear(sentence)
            emb = embds.mean(axis=0)
            out.append(emb.view(*emb.shape, 1))

        out = torch.cat(out, dim=1).permute(1, 0)
        out = self.classification(out)
        loss = self.loss(out, label)
        self.log(
            "Validation loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=len(batch)
        )
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    @staticmethod
    def collate_batch(batch):
        texts = []
        labels = []
        for sentence, label in batch:
            if np.array(sentence).shape[0] == 0:
                print("DDD")
            sentence = torch.tensor(np.array(sentence), dtype=torch.float32)
            texts.append(sentence)
            labels.append(label)

        labels = torch.tensor(labels).float()
        return texts, labels.view(*labels.shape, 1)
