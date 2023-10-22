from .toxicity_regression.model import *

def build_model(model_config: dict, training_config: dict):
    model_params = {i:model_config[i] for i in model_config if i not in ['name']}
    model = eval(f"{model_config.name}")(**model_params, **training_config)
    return model