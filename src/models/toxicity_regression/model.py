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

    def forward(self, input: torch.tensor, offsets) -> torch.tensor:
        out = self.linear(input)
        per_batch = []
        for i in range(len(offsets)):
            per_batch.append(out[offsets[i] : offsets[i + 1]].mean())
        out = torch.per_batch(out, dim=1).permute(1, 0)
        out = self.classification(out)
        return out

    def _count_loss(self, batch):
        sentences_batch, offsets, label = batch
        out = self.linear(sentences_batch)

        per_batch = []
        for i in range(len(offsets) - 1):
            _c = out[offsets[i] : offsets[i + 1]].mean(dim=0)
            per_batch.append(_c.view(*_c.shape, 1))

        out = torch.cat(per_batch, dim=1).permute(1, 0)
        out = self.classification(out)
        loss = self.loss(out, label)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._count_loss(batch)
        self.log("train_loss", loss, prog_bar=True, batch_size=len(batch))
        return loss

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        loss = self._count_loss(batch)

        self.log(
            "val loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=len(batch),
        )
        return loss

    def test_step(self, batch, batch_idx) -> STEP_OUTPUT:
        loss = self._count_loss(batch)
        self.log("test_loss", loss, batch_size=len(batch))
        return loss

    def configure_optimizers(self):
        optimizer_params = {
            i: self._optimizer_args[i]
            for i in self._optimizer_args
            if i not in ["name"]
        }
        optimizer = eval(f"{self._optimizer_args.name}")(
            self.parameters(), **optimizer_params
        )
        return optimizer

    @staticmethod
    def collate_batch(batch):
        texts = []
        labels = []
        offsets = [0]
        for sentence, label in batch:
            sentence = torch.tensor(np.array(sentence), dtype=torch.float32)
            texts.append(sentence)
            offsets.append(sentence.size(0))
            labels.append(label)

        texts = torch.cat(texts)
        offsets = torch.tensor(offsets, dtype=torch.long).cumsum(dim=0)
        labels = torch.tensor(labels).float()
        return texts, offsets, labels.view(*labels.shape, 1)


class LogisticRegression(pl.LightningModule):
    def __init__(self, input_dim: int, optimizer_args, **args) -> None:
        super().__init__()
        self._optimizer_args = optimizer_args
        self.lr = nn.Sequential(
            nn.Linear(input_dim, 4000),
            nn.ReLU(),
            nn.Linear(4000, 1000),
            nn.ReLU(),
            nn.Linear(1000, 1),
        )
        self.sigmoid = nn.Sigmoid()
        self.loss = nn.BCELoss()

    def forward(self, input: torch.tensor) -> torch.tensor:
        out = self.lr(input)
        out = self.sigmoid(out)
        return out

    def training_step(self, batch) -> STEP_OUTPUT:
        input, labels = batch
        out = self.forward(input)
        loss = self.loss(out, labels)
        return loss

    def test_step(self, batch) -> STEP_OUTPUT:
        loss = self.training_step(batch)
        self.log("test loss", loss, on_epoch=True)

    def validation_step(self, batch) -> STEP_OUTPUT:
        loss = self.training_step(batch)
        self.log("val loss", loss, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer_params = {
            i: self._optimizer_args[i]
            for i in self._optimizer_args
            if i not in ["name"]
        }
        optimizer = eval(f"{self._optimizer_args.name}")(
            self.parameters(), **optimizer_params
        )
        return optimizer

    @staticmethod
    def collate_batch(batch):
        inp_vectors = []
        labels = []
        for inp_v, label in batch:
            labels.append(label)
            inp_v = torch.tensor(inp_v)
            inp_vectors.append(inp_v.view(*inp_v.shape, 1))
        inp_vectors = torch.cat(inp_vectors, dim=1).float().permute(1, 0)
        labels = torch.tensor(labels, dtype=torch.float)
        return inp_vectors, labels.view(*labels.shape, 1)
