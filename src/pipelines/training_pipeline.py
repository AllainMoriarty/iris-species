import yaml
from copy import deepcopy
import sys
from src.data.prepare_dataset import prepare_dataset
from src.models.train import train_model
from src.models.evaluate import evaluate_model


def deep_update(base_dict, override_dict):
    for key, value in override_dict.items():
        if isinstance(value, dict) and key in base_dict:
            base_dict[key] = deep_update(base_dict[key], value)
        else:
            base_dict[key] = value
    return base_dict


def run_pipeline(base_config_path: str, experiment_config_path: str | None = None):
    with open(base_config_path) as f:
        config = yaml.safe_load(f)

    if experiment_config_path:
        with open(experiment_config_path) as f:
            exp_config = yaml.safe_load(f)

        config = deep_update(deepcopy(config), exp_config)

    prepare_dataset(config)
    train_model(config)
    evaluate_model(config)

if __name__ == "__main__":
    base_cfg = sys.argv[1]
    exp_cfg = sys.argv[2] if len(sys.argv) > 2 else None

    run_pipeline(base_cfg, exp_cfg)