import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib

def train_model(config):
    X_train = pd.read_csv("data/processed/X_train.csv")
    y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()

    model = LogisticRegression(**config['model']['params'])
    model.fit(X_train, y_train)

    joblib.dump(model, config['artifacts']['model_path'])
    return model