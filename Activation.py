import numpy as np
import copy as cp

class Activation(object):

    def _relu(self, a):
        return np.maximum(a, 0)

    def _relu_deriv(self, x):
        x_copy = cp.deepcopy(x)

        x_copy[x_copy<=0] = 0
        x_copy[x_copy>0] = 1

        return x_copy

    def __init__(self, activation='relu'):
        if activation == 'relu':
            self.f = self._relu
            self.f_deriv = self._relu_deriv
