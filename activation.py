import numpy as np
import copy
from abc import ABC, abstractmethod

class activation(ABC):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    @abstractmethod
    def activate(self, x):
        pass
    
    @abstractmethod
    def deriv(self, x):
        pass    

class relu(activation):

    def activate(self, x):
        return np.maximum(x, 0)

    def deriv(self, x):

        x[x <= 0] = 0
        x[x > 0] = 1

        return x

class leaky_relu(activation):

    def activate(self, x, alpha=0.1):
        y = copy.deepcopy(x)
        x[x <= 0] = alpha

        return x*y

    def deriv(self, x, alpha=0.1):
        
        x[x <= 0] = alpha
        x[x > 0] = 1
        
        return x

class softmax():

    def activate(self, X):

        exps = np.exp(X-np.max(X))

        return exps/np.sum(exps, axis=1, keepdims=True)

    def deriv(self, X):

        print('Most likely a bug that you\'re differentiating the softmax...')

        X = self.activate(X)

        return np.diagflat(X) - np.dot(X, X.T)

class logistic():

    def activate(self, X):

        return 1.0 / (1.0 + np.exp(-X))

    def deriv(self, X):

        X = self.activate(X)

        return X * (1 - X)

class tanh():

    def activate(self, X):
        return np.tanh(X)

    def deriv(self, X):
        X = self.activate(X)
        return 1.0 - np.power(X, 2)
