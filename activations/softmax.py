from activation import Activation

class softmax(Activation):

    def __deriv__(self, X):
        return softmax - np.power(softmax, 2)

    def __activate__(self, X):
        exps = np.exp(X - np.max(X))
        return exps/np.sum(exps)

softmax()
