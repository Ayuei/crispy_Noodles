from Activation import Activation
from HiddenLayer import HiddenLayer
import numpy as np

class MLP:

    def __init__(self,layers,activation='relu'):
        self.layers = []
        self.params = []

        self.activation = activation

        for i in range(len(layers)-1):
            self.layers.append(HiddenLayer(layers[i], layers[i+1],
                                           activation=activation))

    def forward(self, input):
        for layer in self.layers:
            output=layer.forward(input)
            input=output
        return output

    def criterion_MSE(self, y, y_hat):
        activation_deriv = Activation(self.activation).f_deriv
        error = y - y_hat
        loss = error**2

        delta=error*activataion_deriv(y_hat)

        return loss,delta

    def backward(self,delta):
        for layer in reversed(self.layers):
            delta=layer.backward(delta)

    def update(self, lr):
        for layer in self.layers:
            layer.W += lr * layer.grad_W
            layer.b += lr * layer.grab_b

    def fit(self, X, Y, learning_rate=0.1, epochs=100):
        X = np.array(X)
        Y = np.array(Y)

        to_return = np.zeros(epochs)

        for k in range(epochs):
            loss = np.zeros(X.shape[0])
            for it in range(X.shape[0]):
                i = np.random.randint(X.shape[0])

                y_hat = self.forward(X[i])

                loss[it], delta=self.criterion_MSE(Y[i], y_hat)

                self.backward(delta)

                self.update(learning_rate)

            to_return[k] = np.mean(loss)
        return to_return

    def predict(self,X):
        X = np.array(X)
        output = np.zeros(x.shape[0])
        for i in np.arange(x.shape[0]):
            output[i] = nn.foward(x[i,:])
        return output
