import numpy as np
from .MyMetrics import r2_squared


class MyStoGradientLinearRegression:

    def __init__(self):
        self.coef_ = None
        self.intercept_ = None
        self._theta = None

    def fit(self, X, y, n_iters=5):

        def J(theta, X_b, y):
            return np.sum((y - X_b.dot(theta)) ** 2) / len(X_b)

        def dJ_sgd(theta, X_b_i, y_i):
            res = X_b_i.T.dot(X_b_i.dot(theta) - y_i)
            return res * 2

        def gradient_descent_sgd(X_b, y, initial_theta, n_iters):
            cur_iter = 0
            t0, t1 = 5, 50
            m = len(X_b)
            theta = initial_theta

            def learning_rate(t):
                return t0 / (t + t1)

            for cur_iter in range(n_iters):
                indexs = np.random.permutation(m)
                X_b_new = X_b[indexs]
                y_new = y[indexs]

                for i in range(m):
                    gradient = dJ_sgd(theta, X_b_new[i], y_new[i])
                    theta = theta - learning_rate(cur_iter * m + i) * gradient

            return theta

        X_b = np.hstack((np.ones(shape=(len(X), 1)), X))
        initial_theta = np.random.randn(X_b.shape[1])
        self._theta = gradient_descent_sgd(X_b, y, initial_theta, n_iters)
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
        return "MyStoGradientLinearRegression()"
