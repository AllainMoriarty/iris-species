import os
from .load import load_raw_data
from .split import stratified_split

def prepare_dataset(config):
    df = load_raw_data(config['data']['raw_path'])

    X = df.drop(columns=[config['data']['target_column']])
    y = df[config['data']['target_column']]


    X_train, X_val, X_test, y_train, y_val, y_test = stratified_split(X, y, config['split']['train_size'], config['split']['val_size'], config['split']['test_size'], config['project']['random_seed'])

    os.makedirs(config['data']['processed_dir'], exist_ok=True)

    X_train.to_csv(f"{config['data']['processed_dir']}/X_train.csv", index=False)
    X_val.to_csv(f"{config['data']['processed_dir']}/X_val.csv", index=False)
    X_test.to_csv(f"{config['data']['processed_dir']}/X_test.csv", index=False)

    y_train.to_csv(f"{config['data']['processed_dir']}/y_train.csv", index=False)
    y_val.to_csv(f"{config['data']['processed_dir']}/y_val.csv", index=False)
    y_test.to_csv(f"{config['data']['processed_dir']}/y_test.csv", index=False)