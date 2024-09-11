import pandas as pd 


def get_data_usable():
    """create Datasets and return it
    """
    df_train = pd.read_csv("../../dataset/diabetes_train_usable.csv")
    df_test = pd.read_csv("../../dataset/diabetes_test_usable.csv")

    X_train = df_train.drop("diabetes", axis=1)
    y_train = df_train["diabetes"]
    X_test = df_test.drop("diabetes", axis=1)
    y_test = df_test["diabetes"]

    return X_train, X_test, y_train, y_test