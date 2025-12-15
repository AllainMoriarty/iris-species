from sklearn.model_selection import train_test_split

def stratified_split(X, y, train_size, val_size, test_size, random_state):
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=(1 - train_size), stratify=y, random_state=random_state)

    val_ratio = val_size / (val_size + test_size)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=(1 - val_ratio), stratify=y_temp, random_state=random_state)
    
    return X_train, X_val, X_test, y_train, y_val, y_test