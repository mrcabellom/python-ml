import numpy as np

class Perceptron(object):
    """Binary Perceptron"""

    def __init__(self, eta=0.1, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter
        self.weights_ = None
        self.errors_ = None

    def fit(self, x, y):
        """Fit perceptron"""
        self.weights_ = np.zeros(1 + x.shape[1])
        self.errors_ = []
        for _ in range(self.n_iter):
            errors = 0
            for index_x, target in zip(x, y):
                update = self.eta * (target - self.predict(index_x))
                self.weights_[1:] += update * index_x
                self.weights_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, x):
        return np.dot(x, self.weights_[1:]) + self.weights_[0]

    def predict(self, x):
        return np.where(self.net_input(x) >= 0.0, 1, 0)
