import numpy as np

class MSE():
    def __init__(self):
        pass

    def __loss__(self, y_hat, y):
        return np.sum(np.abs(np.power(y_hat-y, 2)))
