import numpy
#import all activations

class layer():

    def __init__(self, n=None, in_shape=None, activation="relu"):
        assert(n is not None and in_shape is not None)
        if activation in globals().keys()
            self.activation = globals()[activation]
        else:
            raise NotImplementedError()

        self.W = np.random.randn(n,n-1, size=(in_shape, n))*np.sqrt(1/(n-1))
        self.b = [np.random.randn(n,n-1)*np.sqrt(1/(n-1)) for i in range(n)]

        self.grad_W = np.zeros(self.W.shape)
        self.grad_b = np.zeros(self.b.shape)

    def forward(self, inp):
        self.input = inp
        return self.activation.__activate__(np.dot(inp, self.W)+self.b)

    def backward(self, inp)



class sequential():

    def __init__(self):
        self.layers = []
        self.
