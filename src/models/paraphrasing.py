from typing import Any
from lightning.pytorch.utilities.types import STEP_OUTPUT
import torch

import lightning.pytorch as pl
from transformers import AutoModelForSeq2SeqLM, AdamW


class T5model(pl.LightningModule):
    def __init__(
        self,
        model_name: str,
        optimizer_args,
        pretrained_path: str = None,
        *args: Any,
        **kwargs: Any
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.optimizer_args = optimizer_args
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        if pretrained_path is not None:
            self = T5model.load_from_checkpoint(
                pretrained_path, model_name=model_name, optimizer_args=optimizer_args
            )
            return

    def forward(self, input_ids, attention_mask, target=None):
        output = self.model(
            input_ids=input_ids, attention_mask=attention_mask, labels=target
        )
        return output.loss, output.logits

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        target = batch["target"]
        loss, logits = self(input_ids, attention_mask, target)
        self.log("train loss", loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        target = batch["target"]
        loss, logits = self(input_ids, attention_mask, target)
        self.log("val loss", loss, prog_bar=True, on_epoch=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        target = batch["target"]
        loss, logits = self(input_ids, attention_mask, target)
        self.log("test loss", loss, prog_bar=True, on_epoch=True, logger=True)
        return loss

    @staticmethod
    def collate_batch(batch):
        input_ids = []
        attention_masks = []
        targets = []

        for ref, target in batch:
            input_ids.append(torch.tensor(ref["input_ids"]))
            attention_masks.append(torch.tensor(ref["attention_mask"]))
            targets.append(torch.tensor(target["input_ids"]))

        input_ids = torch.stack(input_ids)
        attention_masks = torch.stack(attention_masks)
        targets = torch.stack(targets)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_masks,
            "target": targets,
        }

    def configure_optimizers(self):
        return AdamW(self.parameters(), **self.optimizer_args)

    def save(self, path):
        self.model.save_pretrained(path)
