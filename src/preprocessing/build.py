from .tokenizers import *
from .vocabulars import *

def _build(local_conifg):
    params = {k:v for k, v in local_conifg.items() if k not in ["name"]}
    object = eval(local_conifg.name)(**params)
    return object

def build_preprocessing(preprocessing_config: dict):
    tokenizer = _build(preprocessing_config.tokenizer)
    text2vector: Text2Vector = _build(preprocessing_config.text2vector)
    if preprocessing_config.text2vector.get("load_path", None) is not None:
        text2vector = text2vector.load(preprocessing_config.text2vector.load_path)
    return tokenizer, text2vector
