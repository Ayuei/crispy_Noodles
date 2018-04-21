from Activation import Activation
from HiddenLayer import HiddenLayer
import numpy as np

class MLP:
    """
    """
    def __init__(self, layers, activation='relu'):
        """
        :param layers: A list containing the number of units in each layer.
        Should be at least two values
        :param activation: The activation function to be used. Can be
        "logistic" or "tanh"
        """
        ### initialize layers
        self.layers = []
        self.params = []

        self.activation = activation
        for i in range(len(layers) - 1):
            self.layers.append(HiddenLayer(layers[i], layers[i + 1], activation=activation))

    def forward(self, input):
        for layer in self.layers:
            output = layer.forward(input)
            input = output
        return output

    #def softmax(self, y, y_hat):


    def criterion_MSE(self, y, y_hat):
        activation_deriv = Activation(self.activation).f_deriv
        # MSE
        error = y - y_hat
        loss = error ** 2
        # write down the delta in the last layer
        delta = error * activation_deriv(y_hat)
        # return loss and delta

        return np.sum(loss), delta

    def cross_entropy_loss_softmax(self, y, y_hat):
        activation_deriv = Activation(self.activation).f_deriv

        error = -np.sum(y*np.log(y_hat))

        delta = error*activation_deriv(y_hat)

        return error, delta




    def backward(self, delta):
        for layer in reversed(self.layers):
            delta = layer.backward(delta)

    def update(self, lr):
        for layer in self.layers:
            layer.W += lr * layer.grad_W
            layer.b += lr * layer.grad_b

    def fit(self, X, y, learning_rate=0.1, epochs=100):
        """
        Online learning.
        :param X: Input data or features
        :param y: Input targets
        :param learning_rate: parameters defining the speed of learning
        :param epochs: number of times the dataset is presented to the network for learning
        """
        X = np.array(X)
        y = np.array(y)
        to_return = np.zeros(epochs)

        for k in range(epochs):
            loss = np.zeros(X.shape[0])
            for it in range(X.shape[0]):
                i = np.random.randint(X.shape[0])

                # forward pass
                y_hat = self.forward(X[i])

                # backward pass
                #loss[it], delta = self.criterion_MSE(y[i], y_hat)
                grd_trth = np.array([0 for i in range(y.max())])
                grd_trth[y[i]-1] = 1
                loss[it], delta = self.cross_entropy_loss_softmax(grd_trth, y_hat)
                #loss[it], delta = self.criterion_MSE(grd_trth, y_hat)
                self.backward(delta)

                # update
                self.update(learning_rate)
            to_return[k] = np.mean(loss)
        return to_return

    def predict(self, x):
        x = np.array(x)

        return self.forward(x)