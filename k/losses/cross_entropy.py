import numpy as np

class CrossEntropy():
    def __init__(self,):
        pass

    def __loss__(self, y_hat, y):
        '''
        params: y_hat (predicted)
                y (ground truth)
        '''
        return np.sum(y_hat*np.log(y)+(1-y)*np.log(1-y_hat))

