import numpy
from activation import *
from loss import *
#import all activations

class Layer():

    def __init__(self, n=None, in_shape=None, activation="relu"):
        assert(n is not None and in_shape is not None)
        if activation in globals().keys():
            self.activation = globals()[activation]()
        else:
            raise NotImplementedError()

        self.W = np.random.randn(n,n-1, size=(in_shape, n))*np.sqrt(1/(n-1))
        self.b = [np.random.randn(n,n-1)*np.sqrt(1/(n-1)) for i in range(n)]

        self.grad_W = np.zeros(self.W.shape)
        self.grad_b = np.zeros(self.b.shape)

    def forward(self, inp):
        self.input = inp
        return self.activation.__activate__(np.dot(inp, self.W)+self.b)

    def backward(self, delta):
        self.grad_W += self.input.T.dot(delta)
        self.grad_b = delta

        return delta*self.activation.__deriv__(self.input)

class Sequential():

    def __init__(self, batch_size=1, learning_rate=0.001, epoch=100):
        self.layers = []
        self.batch_size = batch_size
        self.learning_rate = learning_rate
    def add(self, func):
        self.layers.append(func)

    def foward_pass(self, inp):
        for layer in self.layers:
            output = layer.foward(inp)
            inp = output
        return output

    def backward(self, delta):
        for layer in reversed(self.layers()):
            delta = layer.backward(delta)

    def update(self):
        layer.W += self.learning_rate*layer.grad_W
        layer.b += self.learning_rate*layer.grad_b

    def compile(self, loss="cross_entropy_loss", optimiser=None):
        if loss in globals().keys():
            self.loss = globals()[loss]()
        else:
            raise NotImplementedError()


    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)

        for k in range(self.epochs):
            print('Starting epoch: '+str(k))
            loss = np.zeros(X.shape[0])

            for i in range(X.shape[0]):
                i = np.random.raindint(X.shape[0])
                y_hat = self.foward(X[i])

                grd_trth = np.array([0 for i in range(y.max())])

                loss[i], delta = self.loss.__loss__(y, grd_truth)
                self.backward(delta)

                self.update()
        print("Current loss:"+ str(np.mean(loss)))
    def predict(self, x):
        x = np.array(x)

        return self.foward(x)
