import numpy as np


def my_train_test_split(X, y, test_ratio, random):
    shuffle_index = np.random.permutation(X.shape[1])
    X = X[shuffle_index]
    y = y[shuffle_index]
    test_size = len(X) * test_ratio
    X_train = X[test_size:]
    X_test = X[:test_size]
    y_train = y[test_size:]
    y_test = y[:test_size]
    return X_train, X_test, y_train, y_test


def r2_squared(y_test, y_predict):
    return 1 - (metric_mse(y_test, y_predict) / np.var(y_test))


def metric_mse(y_test, y_predict):
    return np.sum((y_test - y_predict) ** 2) / len(y_test)
