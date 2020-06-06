import numpy as np
from .MyMetrics import r2_squared


class MyPolyLinearRegression:

    def __init__(self):
        self.intercept_ = None
        self.coefficients_ = None
        self._theta = None

    def fit(self, X_train, y_train):
        assert len(X_train) == len(y_train), \
            "the size of instances must be equal to the size of y"

        X_b = np.hstack((np.ones(shape=(len(X_train), 1)), X_train))
        self._theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_train)

        self.intercept_ = self._theta[0]
        self.coefficients_ = self._theta[1:]
        return self

    def predict(self, X_test):
        X_b = np.hstack((np.ones(shape=(len(X_test), 1)), X_test))
        y_predict = X_b.dot(self._theta)
        return y_predict

    def score(self, X_test, y_test):
        y_predict = self.predict(X_test)
        return r2_squared(y_test, y_predict)

    def __repr__(self):
        return "MyPolyLinearRegression()"
