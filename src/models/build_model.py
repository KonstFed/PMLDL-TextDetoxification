from .toxicity_regression.model import SimpleToxicClassification
import torch.optim as optim

def build_model(model_config: dict, training_config: dict):
    model_params = {i:model_config[i] for i in model_config if i not in ['name']}
    model = eval(f"{model_config.name}")(loss_args=training_config.loss, optimizer_args=training_config.optimizer, **model_params)
    return model