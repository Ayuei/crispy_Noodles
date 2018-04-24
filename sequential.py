import numpy as np
from activation import *
from loss import *

class Layer():

    def __init__(self, n=None, in_shape=None, activation="leaky_relu"):
        assert(n is not None)
        if activation in globals().keys():
            self.activation = globals()[activation]()
        else:
            raise NotImplementedError()
        self.n = n
        self.W = np.random.randn(in_shape, n)*np.sqrt(1/(n-1))
        self.b = np.array([np.random.randn()*np.sqrt(1/(n-1)) for i in
                           range(n)])
        self.loss = None
#dropout implementation; add layer that shuffles indices, and removes the first
# 40% or something. Use floor.
        self.grad_W = np.zeros(self.W.shape)
        self.grad_b = np.zeros(self.b.shape)

    def forward(self, inp):
        self.input = inp
        return self.activation.__activate__(np.dot(inp, self.W)+self.b)

    def backward(self, delta):
        self.grad_W = np.atleast_2d(self.input).T.dot(np.atleast_2d(delta.T))
        self.grad_b = delta

        return delta.dot(self.W.T)*self.activation.__deriv__(self.input)

class Sequential():

    def __init__(self, batch_size=1, learning_rate=0.001, epochs=100):
        self.layers = []
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate

    def add_layer(self, *args, **params):

        if len(self.layers) != 0:
            params["in_shape"] = self.layers[-1].n

        self.layers.append(Layer(*args, **params))

    def add(self, func):
        self.layers.append(func)

    def forward_pass(self, inp):
        for layer in self.layers:
            output = layer.forward(inp)
            inp = output
        return output

    def backward(self, delta):
        for layer in reversed(self.layers):
            delta = layer.backward(delta)

    def update(self):
        for layer in self.layers:
            layer.W += self.learning_rate*layer.grad_W
            layer.b += self.learning_rate*layer.grad_b

    def compile(self, loss="cross_entropy", optimiser=None):
        if loss in globals().keys():
            self.loss = globals()[loss]()
        else:
            raise NotImplementedError()


    def fit(self, X, y):
        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64)

        for k in range(self.epochs):
            print('Starting epoch: '+str(k))
            loss = np.zeros(X.shape[0])

            for i in range(X.shape[0]):
                i = np.random.randint(X.shape[0])
                y_hat = self.forward_pass(X[i])

                grd_trth = np.array([0 for j in range(int(y.max()))], dtype=np.float64)

                grd_trth[int(y[i])-1] = 1

                loss_ = self.loss.__loss__(y_hat, grd_trth)
                loss[i] = np.mean(loss_)
                delta = loss[i] * self.layers[-1].activation.__deriv__(y_hat)

                self.backward(delta)

                self.update()
            print("Current loss:"+ str(np.mean(loss)))
    def predict(self, x):
        x = np.array(x)

        return self.forward_pass(x)
