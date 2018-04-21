import numpy as np
import copy as cp

class Activation(object):

    def _relu(self, a):
        return np.maximum(a, 0)

    def _relu_deriv(self, x):

        x[x <= 0] = 0
        x[x > 0] = 1

        return x

    def _softmax(self,x):

        exps = np.exp(x)

        return exps/np.sum(exps)

    def _stable_softmax(self, x):
        exps = np.exp(x - np.max(x))
        return exps / np.sum(exps)

    def _cross_entropy(self, X, y):
        m = y.shape[0]
        p = softmax(X)
        log_likelihood = -np.log(p[range(m), y])
        loss = np.sum(log_likelihood) / m
        return loss

    def _softmax_deriv(self, softmax):
        s = softmax.reshape(-1, 1)
        return np.diagflat(s) - np.dot(s, s.T)

    def __init__(self, activation='relu'):
        if activation == 'relu':
            self.f = self._relu
            self.f_deriv = self._relu_deriv
        if activation == 'softmax':
            self.f = self._stable_softmax
            self.f_deriv = self._softmax_deriv
