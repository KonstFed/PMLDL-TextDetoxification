from .toxicity_classification import *

def build_model(model_config: dict, training_config: dict):
    if model_config.get("checkpoint_path", None) is not None:
        model = eval(f"{model_config.name}").load_from_checkpoint(model_config.checkpoint_path)
    else:
        model_params = {i:model_config[i] for i in model_config if i not in ['name']}
        model = eval(f"{model_config.name}")(**model_params, **training_config)
    return model



def load_model(model_config: dict, path: str):
    model = eval(f"{model_config.name}")
    return model.load_from_checkpoint(path)