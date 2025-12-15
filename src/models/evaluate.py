import pandas as pd
import joblib
from src.utils.metrics import classification_metrics

def evaluate_model(config):
    model = joblib.load(config['artifacts']['model_path'])

    X_test = pd.read_csv("data/processed/X_test.csv")
    y_test = pd.read_csv("data/processed/y_test.csv").values.ravel()

    metrics = classification_metrics(model, X_test, y_test)

    with open(config['reports']['results_path'], "a") as f:
        f.write("\n## Test Metrics\n")
        for k, v in metrics.items():
            f.write(f"- {k}: {v}\n")

    return metrics