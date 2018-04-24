import numpy as np

class relu():

    def __activate__(self, x):
        return np.maximum(x,0)

    def __deriv__(self,x):

        x[x <= 0] = 0
        x[x > 0] = 1

        return x


class softmax():

    def __activate__(self, X):
        exps = np.exp(X - np.max(X))
        return exps/np.sum(exps)

    def __deriv__(self, X):
        return softmax - np.power(softmax, 2)