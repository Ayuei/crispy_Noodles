import numpy as np
from abc import ABC, abstractmethod

class loss(ABC):
    def __init(self, **kwargs):
        self.kwargs = kwargs

    @abstractmethod
    def loss(self, y_hat, y):
        pass

    @abstractmethod
    def grad(self, y_hat, y):
        pass

class mean_squared_error(loss):
    def loss(self, y_hat, y):
        return np.sum(np.abs(np.power(y_hat-y, 2)))


    def grad(self, y_hat, y):
        return np.abs(y_hat - y)

class cross_entropy_softmax(loss):
    def loss(self, y_hat, y):
        print(y_hat)
        return -np.sum(np.log(y_hat) * y)

    def grad(self, y_hat, y):
        return y_hat - y
