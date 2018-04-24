import numpy as np


class mean_squared_error():
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __loss__(self, y_hat, y):
        return np.sum(np.abs(np.power(y_hat-y, 2)))

class cross_entropy():
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __loss__(self, y_hat, y):
        '''
        params: y_hat (predicted)
                y (ground truth)
        '''

        #Avoid log(0)
        y_hat[y_hat == 0.0] = (np.min(y_hat[np.nonzero(y_hat)]))

        return np.log(y_hat) * y

