from layers import *
from optimisers import *
from loss import *
from activation import *

class Sequential():

    def __init__(self, batch_size=32, learning_rate=0.001, epochs=100, learning_rate_decay=1, weight_decay = 0):
        self.layers = []
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.prev_n = None
        self.learning_rate_decay = learning_rate_decay
        self.weight_decay = weight_decay
        self.optimiser = None
        self.loss = None

    def add(self, obj):
        obj.create_weights(self.prev_n)
        self.layers.append(obj)
        try:
            self.prev_n = obj.n
        except AttributeError:
            pass

    def forward_pass(self, inp, **kwargs):
        for layer in self.layers:
            output = layer.forward(inp, **kwargs)
            inp = output
        return output

    def backward(self, delta):
        for i, layer in enumerate(reversed(self.layers)):
            if i == 0:
                delta = layer.backward(delta, last_layer=True)
            else:
                delta = layer.backward(delta, last_layer=False)

    def update(self):
        for layer in self.layers:
            layer.update(self.learning_rate, self.weight_decay)

    def compile(self, loss="cross_entropy_softmax", optimiser=None):
        if loss in globals().keys():
            self.loss = globals()[loss](self.layers[-1].activation)
        else:
            raise NotImplementedError()

        if optimiser is None:
            # Mini-batch
            return

        if optimiser in globals().keys():
            self.optimiser = globals()[optimiser]
            for i, layer in enumerate(self.layers):
                if type(layer).__name__ == "Dense":
                    self.layers[i] = self.optimiser(layer, weight_decay=self.weight_decay)

    def fit(self, X, y, verbose=True):
        
        for k in range(self.epochs):

            self.learning_rate = self.learning_rate*self.learning_rate_decay

            print('=====================')
            print('EPOCH: '+str(k)+'/'+str(self.epochs))
            loss = np.zeros(X.shape[0])
            correct_instances = 0
            batches = self.get_batches(X,y)
            for i in range(len(batches)):

                batch_X, batch_Y = batches[i]

                y_hat = self.forward_pass(batch_X)
                loss_ = self.loss.loss(y_hat, batch_Y)
                loss[i] = np.mean(loss_)

                if verbose:
                    correct_instances += np.sum(np.argmax(y_hat, axis=1) == np.argmax(batch_Y, axis=1))

                delta = self.loss.grad(y_hat, batch_Y, self.layers[-1])
                
                self.backward(delta)
                self.update()
            
            print('=====================')
            print("Current mean loss:", np.mean(loss))
            if verbose:
                print("Current training acc:", correct_instances/X.shape[0])

    def predict(self, x):
        x = np.array(x)

        return self.forward_pass(x, predict=True)

    def get_batches(self, X, Y):
        m = X.shape[0]
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

        if m % self.batch_size != 0:
            mini_batch_X = shuffled_X[-(m % self.batch_size): m]
            mini_batch_Y = shuffled_Y[-(m % self.batch_size): m]
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)

        return mini_batches