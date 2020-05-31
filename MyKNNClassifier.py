import numpy as np
import math
from collections import Counter


class MyKNNClassifier:

    def __init__(self, k):
        assert k >= 1, "k must be greater than 1"

        self.k = k
        self._X_train = None
        self._y_train = None

    def fit(self, X_train, y_train):
        assert X_train.shape[1] <=2, "the shape should be less than 2"
        assert X_train.shape[0] == len(y_train), \
            "the size of X_train must be equal to the size of y_train"

        self._X_train = X_train
        self._y_train = y_train
        return self

    def predict(self, X_test):
        assert self._X_train is not None and self._y_train is not None, \
            "must be fit before predict"
        assert self._X_train.shape[1] == X_test.shape[1], \
            "the feature number of X_train must be equal to X_test"

        y_predict = [self._predict(x_test) for x_test in X_test]
        return y_predict

    def _predict(self, x_test):
        distances = [math.sqrt(np.sum((x_train - x_test) ** 2)) for x_train in self._X_train]
        nearest = np.argsort(distances)
        topK = [self._y_train[arg] for arg in nearest[:self.k]]
        votes = Counter(topK).most_common(1)[0][0]
        return votes

    def score(self, X_test, y_test):
        y_predict = self.predict(X_test)
        return np.sum(y_predict == y_test) / len(y_test)

    def __repr__(self):
        return "MyKNNClassifier(k=%d)" % self.k
