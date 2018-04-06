from Activation import Activation
import numpy as np


class HiddenLayer(object):
    def __init__(self, n_in, n_out, W=None, b=None,
                 activation='relu'):
        self.input=None
        self.activation = Activation(activation).f
        self.activation_deriv = Activation(activation).f_deriv

        self.W = np.random.uniform(
            low = -np.sqrt(6 / (n_in + n_out)),
            high = np.sqrt(6 / (n_in + n_out)),
            size=(n_in, n_out)
        )
        self.b = np.zeros(n_out,)

        self.grad_W = np.zeros(self.W.shape)
        self.grad_b = np.zeros(self.b.shape)

    def forward(self, input):
        lin_output = np.dot(input, self.W) + self.b
        self_output = (
            lin_output if self.activation is None
            else self.activation(input)
        )
        self.input = input
        return self_output

    def backward(self, delta):
        self.grad_W = np.atleast_2d(self.input).T.dot(np.atleast_2d(delta))
        self.grab_b = delta

        delta_ = delta.dot(self.W.T) * self.activation_deriv(self.input)

        return delta_
