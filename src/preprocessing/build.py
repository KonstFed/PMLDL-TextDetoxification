from .tokenizers import *
from .vocabulars import *


def _build(local_conifg):
    params = {k: v for k, v in local_conifg.items() if k not in ["name"]}
    object = eval(local_conifg.name)(**params)
    return object


def build_preprocessing(preprocessing_config: dict) -> tuple[Tokenizer, Text2Vector]:
    values = {}
    for k in preprocessing_config.keys():
        values[k] = _build(preprocessing_config[k])
    # tokenizer = _build(preprocessing_config.tokenizer)
    # text2vector: Text2Vector = _build(preprocessing_config.text2vector)
        if preprocessing_config[k].get("load_path", None) is not None:
            values[k] = values[k].load(preprocessing_config[k]["load_path"])
    return values
