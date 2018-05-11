import numpy as np
from abc import ABC, abstractmethod


class loss(ABC):
    def __init(self, **kwargs):
        self.kwargs = kwargs

    @abstractmethod
    def loss(self, y_hat, y):
        pass

    @abstractmethod
    def grad(self, y_hat, y, layer):
        pass


class mean_squared_error(loss):

    def __init__(self, activation):
        self.activation = activation

    def loss(self, y_hat, y):

        num_examples = y_hat.shape[0]

        data_loss = np.sum((np.square(np.subtract(y_hat, y)).mean(axis=0)))

        return 1./num_examples * data_loss

    def grad(self, y_hat, y, layer):

        m = y.shape[0]

        return (np.abs(y_hat - y))/m * self.activation.deriv(layer.output)


class cross_entropy_softmax(loss):

    def __init__(self, activation):
        if activation.__class__.__name__!= 'softmax':
            print("WARNING: Cross Entropy is not being run with SoftMax")
            self.use_softmax = False

        else:
            self.use_softmax = True

        self.activation = activation


    def loss(self, y_hat, y):

        num_examples = y_hat.shape[0]

        probs = y_hat

        if y.ndim > 1:
            yGuess = np.argmax(y, axis=1)
        else:
            yGuess = y

        corect_logprobs = -np.log(probs[range(num_examples), yGuess.astype(int)])

        return np.mean(corect_logprobs)

    def grad(self, y_hat, y, layer):
        m = y.shape[0]

        yGuess = 0

        if y.ndim > 1:
            yGuess = (np.argmax(y, axis=1))
        else:
            yGuess = y

        grad = np.array(y_hat, copy=True)

        grad[range(m), yGuess.astype(int)] -= 1

        grad = grad / m

        if not self.use_softmax:
            grad = grad * self.activation.deriv(layer.output)

        return grad
