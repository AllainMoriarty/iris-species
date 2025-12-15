import sys
import yaml
from src.data.prepare_dataset import prepare_dataset
from src.models.train import train_model
from src.models.evaluate import evaluate_model

def run_pipeline(config_path: str):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    prepare_dataset(config)
    train_model(config)
    evaluate_model(config)

if __name__ == "__main__":
    config_path = sys.argv[1]
    run_pipeline(config_path)