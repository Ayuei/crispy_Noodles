import numpy as np
from activation import *
from loss import *
from layers import *


class Sequential():

    def __init__(self, batch_size=50, learning_rate=0.001, epochs=100):
        self.layers = []
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.prev_n = None

    def add(self, obj):
        obj.create_weights(self.prev_n)
        self.layers.append(obj)
        try:
            self.prev_n = obj.n
        except AttributeError:
            pass

    def forward_pass(self, inp):
        for layer in self.layers:
            output = layer.forward(inp)
            inp = output
        return output

    def backward(self, delta):
        for i, layer in enumerate(reversed(self.layers)):
            if i == 0:
                delta = layer.backward(delta, True)
            else:
                delta = layer.backward(delta)

    def update(self):
        for layer in self.layers:
            layer.W += self.learning_rate*layer.grad_W
            layer.b += self.learning_rate*layer.grad_b

    def compile(self, loss="cross_entropy", optimiser=None):
        if loss in globals().keys():
            self.loss = globals()[loss]()
        else:
            raise NotImplementedError()

    def fit(self, X, y):
  
        num_classes = len(np.unique(y))
        
        for k in range(self.epochs):
            print('Starting epoch: '+str(k))
            loss = np.zeros(X.shape[0])
            batches = self.get_batches(X,y)
            for i in range(len(batches)):

                batch_X, batch_Y = batches[i]

                y_hat = self.forward_pass(batch_X)
                
                #CHANGE 3
                loss_ = self.loss.loss(y_hat, batch_Y)
                
                #CHANGE 4
                loss[i] = np.mean(loss_)
                delta = self.loss.grad(y_hat, batch_Y)

                #CHANGE 5
                if i % 1000 == 0:
                    print(y_hat)
                    print('Loss at stoch item %s is %s' % (i, loss[i]))
                
                self.backward(delta)
                self.update()
            
            print('=====================')
            print("Current mean loss:"+ str(np.mean(loss)))
            
            
    def predict(self, x):
        x = np.array(x)

        return self.forward_pass(x)


    def get_batches(self, X, Y):
        m = X.shape[0]f
        mini_batches = []

        permutation = list(np.random.permutation(m))

        shuffled_X = X[permutation]
        shuffled_Y = Y[permutation]

        num_complete_minibatches = math.floor(m / self.batch_size)
        for k in range(0, int(num_complete_minibatches)):
            mini_batch_X = shuffled_X[(k * self.batch_size): (k + 1) * self.batch_size]
            mini_batch_Y = shuffled_Y[(k * self.batch_size): (k + 1) * self.batch_size]
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)

        if m % batchsize != 0:
            mini_batch_X = shuffled_X[-(m % self.batch_size): m]
            mini_batch_Y = shuffled_Y[-(m % self.batch_size): m]
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)

        return mini_batches