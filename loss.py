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

        num_examples = y_hat.shape[0]

        probs = y_hat

        if y.ndim > 1:
            yGuess = np.argmax(y, axis=1)
        else:
            yGuess = y

        corect_logprobs = -np.log(probs[range(num_examples), yGuess.astype(int)])

        data_loss = np.sum(corect_logprobs)

        return 1. / num_examples * data_loss

    def grad(self, y_hat, y):
        m = y.shape[0]
        yGuess = 0
        if y.ndim > 1:
            yGuess = (np.argmax(y, axis=1))
        else:
            yGuess = y

        grad = np.array(y_hat, copy=True)

        grad[range(m), yGuess.astype(int)] -= 1

        grad = grad / m
        return grad
