import numpy as np
from .MyMetrics import r2_squared


class MyGradientLinearRegression:

    def __init__(self):
        self.coef_ = None
        self.intercept_ = None
        self._theta = None

    def fit(self, X, y, eta=0.1, epsilon=1e-8, n_iters=1e4):

        def J(theta, X_b, y):
            return np.sum((y - X_b.dot(theta)) ** 2) / len(X_b)

        def dJ(theta, X_b, y):
            ##res = np.empty(X_b.shape[1])
            ##res[0] = np.sum(X_b.dot(theta) - y)
            ##for i in range(1, X_b.shape[1]):
                ##res[i] = (X_b.dot(theta) - y).dot(X_b[:, i])

            res = X_b.T.dot(X_b.dot(theta) - y)
            return res * 2 / len(X_b)

        def gradient_descent(X_b, y, theta, eta, epsilon, n_iters):
            i_iter = 0
            while i_iter < n_iters:
                last_theta = theta
                gradient = dJ(theta, X_b, y)
                theta = theta - eta * gradient
                if abs(J(theta, X_b, y) - J(last_theta, X_b, y)) < epsilon:
                    break
                i_iter = i_iter + 1
            return theta

        X_b = np.hstack((np.ones(shape=(len(X), 1)), X))
        theta = np.zeros(shape=(X_b.shape[1]))
        self._theta = gradient_descent(X_b, y, theta, eta, epsilon, n_iters)
        self.coef_ = self._theta[1:]
        self.intercept_ = self._theta[0]

        return self

    def predict(self, X_test):
        X_b = np.hstack((np.ones(shape=(len(X_test), 1)), X_test))
        return X_b.dot(self._theta)

    def score(self, X_test, y_test):
        y_predict = self.predict(X_test)
        return r2_squared(y_test, y_predict)

    def __repr__(self):
        return "MyGradientLinearRegression()"
