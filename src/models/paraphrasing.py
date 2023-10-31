from typing import Any
import torch

import lightning.pytorch as pl
from transformers import 

class T5model(pl.LightningModule):
    def __init__(self, model_name: str, optimizer_args, pretrained_path: str, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        