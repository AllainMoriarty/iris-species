import joblib
import pandas as pd

def predict(model_path: str, X: pd.DataFrame):
    model = joblib.load(model_path)
    return model.predict(X)