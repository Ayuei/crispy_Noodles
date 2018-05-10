from abc import ABC, abstractmethod
from activation import *
import math

class Layer(ABC):
    
    @abstractmethod
    def forward(self, x, **kwargs):
        pass
    
    @abstractmethod
    def backward(self, x):
        pass
    
    @abstractmethod
    def create_weights(self, in_shape):
        pass

    @abstractmethod
    def update(self, *args):
        pass

class Dropout(Layer):

    def __init__(self, *args, **kwargs):
        import random
        if "dropout" in kwargs:
            self.dropout_rate = kwargs["dropout"]
        else:
            self.dropout_rate = 0.2
        self.rand = random.Random()

        self.W = 0
        self.b = 0
        self.grad_W = 0
        self.grad_b = 0

    def forward(self, inp, **kwargs):

        if 'predict' in kwargs and kwargs['predict']:
            return inp

        indices = [i for i in range(inp.shape[1])]
        self.rand.shuffle(indices)
        stop = math.floor(len(indices)*self.dropout_rate)

        for i in range(int(stop)):
            inp[:, indices[i]] = np.zeros(inp.shape[0])

        return inp

    def create_weights(self, in_shape):
        pass

    def backward(self, inp):
        return inp*(1+self.dropout_rate)

    def update(self, *args):
        pass


class Dense(Layer):

    def __init__(self, n=None, in_shape=None, activation="relu"):
        assert(n is not None)
        if activation in globals().keys():
            self.activation = globals()[activation]()
        else:
            raise NotImplementedError()
        self.n = n
        self.loss = None
        self.in_shape = in_shape

    def create_weights(self, in_shape):

        if self.in_shape is None:
            self.in_shape = in_shape

        self.W = np.random.randn(self.in_shape, self.n)*np.sqrt(1/(self.n-1))
        self.b = np.atleast_2d(np.array([np.random.randn()*np.sqrt(1/(self.n-1)) for i in range(self.n)]))

        self.grad_W = np.zeros(self.W.shape)
        self.grad_b = np.zeros(self.b.shape)

    def forward(self, inp, **kwargs):

        self.input = inp

        output = self.activation.activate(np.dot(inp, self.W)+self.b)

        self.output = output

        return output

    def backward(self, delta, last_layer=False):

        if not last_layer:
            delta = delta*self.activation.deriv(self.output)

        self.grad_W = np.atleast_2d(self.input).T.dot(np.atleast_2d(delta))
        self.grad_b = np.dot(np.ones((1, delta.shape[0]), dtype=np.float64), delta)

        return delta.dot(self.W.T)

    def update(self, learning_rate):
        self.W -= learning_rate * self.grad_W
        self.b -= learning_rate * self.grad_b


class Batch_norm(Layer):
    def __init__(self, in_shape, out_shape, epsilon=1e-16):
        self.gamma = np.ones(shape=(in_shape))
        self.beta = np.zeros(shape=(in_shape))
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.epsilon=epsilon
        self.x_hat = None
        self.x_mu = None
        self.inv_var = None
        self.sqrtvar = None
        self.var = None

    def forward(self, X, **kwargs):
        self.mu = np.mean(X, axis=0)
        self.var = np.var(X, axis=0)

        self.X_norm = (X - self.mu) / np.sqrt(self.var + self.epsilon)
        output = self.gamma * self.X_norm + self.beta

        self.X = X

        return output

    def backward(self, delta):
        N, D = self.X.shape

        X_mu = self.X - self.mu
        std_inv = 1. / np.sqrt(self.var + 1e-8)

        dX_norm = delta * self.gamma
        dvar = np.sum(dX_norm * X_mu, axis=0) * -.5 * std_inv ** 3
        dmu = np.sum(dX_norm * -std_inv, axis=0) + dvar * np.mean(-2. * X_mu, axis=0)

        delta = (dX_norm * std_inv) + (dvar * 2 * X_mu / N) + (dmu / N)
        self.dgamma = np.sum(delta * self.X_norm, axis=0)
        self.dbeta = np.sum(delta, axis=0)

        return delta

    def create_weights(self, in_shape):
        pass

    def update(self, learning_rate):
        self.gamma -= learning_rate*self.dgamma
        self.beta -= learning_rate*self.dbeta
