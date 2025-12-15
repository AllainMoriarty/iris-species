import os
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

def build_model(config):
    model_name = config["model"]["name"]

    if model_name == "logistic_regression":
        return LogisticRegression(**config["model"]["logistic_regression"])

    elif model_name == "knn":
        return KNeighborsClassifier(**config["model"]["knn"])

    elif model_name == "svm":
        return SVC(**config["model"]["svm"])

    else:
        raise ValueError(f"Unsupported model: {model_name}")

def train_model(config):
    model_name = config["model"]["name"]
    model_dir = config["artifacts"]["model_dir"]
    model_path = os.path.join(model_dir, f"{model_name}.joblib")

    os.makedirs(model_dir, exist_ok=True)

    X_train = pd.read_csv("data/processed/X_train.csv")
    y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()

    model = build_model(config)
    model.fit(X_train, y_train)

    joblib.dump(model, model_path)

    print(f"Model saved to: {model_path}")
    return model
