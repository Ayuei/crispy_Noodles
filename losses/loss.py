import numpy as np


class MSE():
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __loss__(self, y_hat, y):
        return np.sum(np.abs(np.power(y_hat-y, 2)))

class CrossEntropy():
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __loss__(self, y_hat, y):
        '''
        params: y_hat (predicted)
                y (ground truth)
        '''
        return np.sum(y_hat*np.log(y)+(1-y)*np.log(1-y_hat))

