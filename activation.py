import numpy as np
import copy

class relu():

    def __activate__(self, x):
        return np.maximum(x,0)

    def __deriv__(self,x):

        x[x <= 0] = 0
        x[x > 0] = 1

        return x

class leaky_relu():

    def __activate__(self, x, alpha=0.01):
        y = copy.deepcopy(x)

        x[x <= 0] = alpha
        x[x > 0] = 1

        return x*y

    def __deriv__(self, x, alpha=0.01):

        x[x <= 0] = alpha
        x[x > 0] = 1

        return x

class softmax():

    def __activate__(self, X):
        exps = np.exp(X - np.max(X))
        return exps / np.sum(exps)

    def __deriv__(self, X):
        return X - np.power(X, 2)
