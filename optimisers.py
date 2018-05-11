import numpy as np
from abc import ABC, abstractmethod


class Optimiser(ABC):

    @abstractmethod
    def forward(self, x, **kwargs):
        pass

    @abstractmethod
    def backward(self, x, **kwargs):
        pass

    @abstractmethod
    def update(self, *args):
        pass


class Adam(Optimiser):
    def __init__(self, layer, regularization=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.01):
        self.reg = regularization
        self.beta1 = beta1
        self.beta2 = beta2
        self.t= 0
        self.v_dW = np.zeros((np.shape(layer.W)))
        self.v_db = np.zeros((np.shape(layer.b)))
        self.s_dW = np.zeros((np.shape(layer.W)))
        self.s_db = np.zeros((np.shape(layer.b)))
        self.vcW = None
        self.vcb = None
        self.scW = None
        self.scb = None
        self.layer = layer
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.input = None
        self.output = None

    def backward(self, delta, **kwargs):

        last_layer = kwargs["last_layer"]

        delta = self.layer.backward(delta, last_layer)

        self.output = self.layer.output

        # Increment time-step for Adam
        self.t += 1
        self.layer.grad_W += self.reg * self.layer.W

        self.v_dW = (self.beta1 * self.v_dW) + ((1 - self.beta1) * self.layer.grad_W)
        self.v_db = (self.beta1 * self.v_db) + ((1 - self.beta1) * self.layer.grad_b)

        self.vcW = self.v_dW / (1 - self.beta1 ** self.t)
        self.vcb = self.v_db / (1 - self.beta1 ** self.t)

        self.s_dW = (self.beta2 * self.s_dW) + ((1 - self.beta2) * self.layer.grad_W ** 2)
        self.s_db = (self.beta2 * self.s_db) + ((1 - self.beta2) * self.layer.grad_b ** 2)

        self.scW = self.s_dW / (1 - self.beta2 ** self.t)
        self.scb = self.s_db / (1 - self.beta2 ** self.t)

        return delta

    def forward(self, X, **kwargs):
        self.input = X
        output = self.layer.forward(X)
        self.output = output
        return output

    def update(self, lr, *args, **kwargs):
        self.layer.W = self.layer.W - (lr * (self.vcW / (np.sqrt(self.scW) + self.epsilon))) - \
                       self.layer.W*self.weight_decay*lr

        self.layer.b = self.layer.b - (lr * (self.vcb / (np.sqrt(self.scb) + self.epsilon))) - \
                       self.layer.b*self.weight_decay*lr


class SGDMomentum(Optimiser):
    def __init__(self, layer, gamma=0.9, weight_decay=0.01):
        self.gamma = gamma
        self.layer = layer
        self.velocity = None
        self.velocity_b = None
        self.weight_decay = weight_decay
        self.input = None
        self.output = None

    def forward(self, x, **kwargs):
        self.input = x
        output = self.layer.forward(x)
        self.output = output
        return output

    def backward(self, delta, **kwargs):
        last_layer = kwargs["last_layer"]

        delta = self.layer.backward(delta, last_layer)

        return delta

    def update(self, lr, *args, **kwargs):

        # Momentum
        if self.velocity is None:
            self.velocity = np.zeros_like(self.layer.grad_W)
            self.velocity_b = np.zeros_like(self.layer.grad_b)

        self.velocity = self.gamma * self.velocity + lr * self.layer.grad_W
        self.velocity_b = self.gamma * self.velocity_b + lr * self.layer.grad_b
        self.layer.grad_W -= self.velocity
        self.layer.grad_b -= self.velocity_b

        # Momentum + Weight Decay
        self.layer.W = self.layer.W - (lr*self.layer.grad_W) - self.layer.W*self.weight_decay*lr
        self.layer.b = self.layer.b - (lr*self.layer.grad_b) - self.layer.b*self.weight_decay*lr
