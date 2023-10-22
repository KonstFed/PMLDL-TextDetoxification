from typing import Any
from lightning.pytorch.utilities.types import STEP_OUTPUT
import torch
from torch import nn
import numpy as np

import lightning.pytorch as pl


class SimpleToxicClassification(pl.LightningModule):
    def __init__(self, input_dim: int, loss_args, optimizer_args):
        super().__init__()
        self.linear = nn.Sequential(nn.Linear(input_dim, 300), nn.ReLU())
        self.classification = nn.Sequential(
            nn.Linear(300, 100), nn.ReLU(), nn.Linear(100, 1), nn.Sigmoid()
        )
        loss_params = {i: loss_args[i] for i in loss_args if i not in ["name"]}
        self.loss = eval(f"nn.{loss_args.name}", {"nn": nn})(**loss_params)
        self._optimizer_args = optimizer_args

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
    
    def _count_loss(self, batch):
        sentences_batch, label = batch
        out = []
        for sentence in sentences_batch:
            embds = self.linear(sentence)
            emb = embds.mean(axis=0)
            out.append(emb.view(*emb.shape, 1))

        out = torch.cat(out, dim=1).permute(1, 0)
        out = self.classification(out)
        loss = self.loss(out, label)
        return loss
    
    def training_step(self, batch, batch_idx):
        loss = self._count_loss(batch)
        self.log(
            "train_loss", loss, prog_bar=True, batch_size=len(batch)
        )
        return loss

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        loss = self._count_loss(batch)
        
        self.log(
            "val loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=len(batch)
        )
        return loss
    
    def test_step(self, batch, batch_idx) -> STEP_OUTPUT:
        loss = self._count_loss(batch)
        self.log('test_loss', loss, batch_size=len(batch))
        return loss

    def configure_optimizers(self):
        optimizer_params = {i:self._optimizer_args[i] for i in self._optimizer_args if i not in ['name']}
        optimizer = eval(f"{self._optimizer_args.name}")(self.parameters(), **optimizer_params)
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
