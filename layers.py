from abc import ABC, abstractmethod
from activation import *
import math

class Layer(ABC):
    
    @abstractmethod
    def forward(self, x):
        pass
    
    @abstractmethod
    def backward(self, x):
        pass
    
    @abstractmethod
    def create_weights(self, in_shape):
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

    def forward(self, inp):
        indices = [i for i in range(inp.shape[0])]
        self.rand.shuffle(indices)
        stop = math.floor(len(indices)*self.dropout_rate)

        for i in range(int(stop)):
            inp[i] = 0

        return inp

    def create_weights(self, in_shape):
        pass

    def backward(self, inp):
        return inp*(1+self.dropout_rate)


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

    def forward(self, inp):

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
