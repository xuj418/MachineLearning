import numpy as np


class MySimpleLinearRegression:

    def __init__(self):
        self.a_ = None
        self.b_ = None

    def fit(self, X_train, y_train):
        assert len(X_train) == len(y_train), \
            "the size of instances must be equal to the size of y"
        assert X_train.ndim == 1, \
            "Simple Linear Regresion can only resolve single feature data"

        num, den = 0.0, 0.0
        num = np.sum([(x_i - np.mean(X_train)) * (y_i - np.mean(y_train))
                      for x_i, y_i in zip(X_train, y_train)])
        den = np.sum([(x_i - np.mean(X_train)) ** 2 for x_i in X_train])
        a = num / den
        b = np.mean(y_train) - a * np.mean(X_train)
        
        self.a_ = a
        self.b_ = b
        return self

    def predict(self, X_test):
        y_predict = self.a_ * X_test + self.b_
        return y_predict

    def __repr__(self):
        return "MySimpleLinearRegression()"


class MyVectorLinearRegression:

    def __init__(self):
        self.a_ = None
        self.b_ = None

    def fit(self, X_train, y_train):
        assert len(X_train) == len(y_train), \
            "the size of instances must be equal to the size of y"
        assert X_train.ndim == 1, \
            "Simple Linear Regresion can only resolve single feature data"

        num = (X_train - np.mean(X_train)).dot(y_train - np.mean(y_train))
        den = (X_train - np.mean(X_train)).dot(X_train - np.mean(X_train))
        a = num / den
        b = np.mean(y_train) - a * np.mean(X_train)

        self.a_ = a
        self.b_ = b
        return self

    def predict(self, X_test):
        y_predict = self.a_ * X_test + self.b_
        return y_predict

    def __repr__(self):
        return "MyVectorLinearRegression()"
