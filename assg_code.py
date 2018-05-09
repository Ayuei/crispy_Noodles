import numpy as np

class Activation(object):
    def __tanh(self, x):
        return np.tanh(x)

    def __tanh_deriv(self, a):
        # a = np.tanh(x)
        return 1.0 - a**2
    def __logistic(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def __logistic_derivative(self, a):
        # a = logistic(x)
        return  a * (1 - a )

    def __relu(self, x):
        return np.maximum(x,0)

    def __relu_derivative(self,x):

        x[x <= 0] = 0
        x[x > 0] = 1

        return x

    def __softmax(self, X):

        exps = np.exp(X - np.max(X))
        #x = np.max(X)
        output = exps / np.sum(exps)

        return output

    def __softmax_derivative(self, X):

        X = self.__softmax(X)
        print(np.diagflat(X) - np.dot(X, X.T))
        exit()
        return np.diagflat(X) - np.dot(X, X.T)


    def __init__(self,activation='tanh'):
        if activation == 'logistic':
            self.f = self.__logistic
            self.f_deriv = self.__logistic_derivative
        elif activation == 'tanh':
            self.f = self.__tanh
            self.f_deriv = self.__tanh_deriv
        elif activation == 'relu':
            self.f = self.__relu
            self.f_deriv = self.__relu_derivative

        elif activation == 'softmax':
            self.f = self.__softmax
            self.f_deriv = self.__softmax_derivative


class HiddenLayer(object):
    def __init__(self, n_in, n_out, W=None, b=None,
                 activation='tanh'):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: string
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.input = None
        self.activation = Activation(activation).f
        self.activation_deriv = Activation(activation).f_deriv
        # end-snippet-1

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.
        self.W = np.random.uniform(
            low=-np.sqrt(6. / (n_in + n_out)),
            high=np.sqrt(6. / (n_in + n_out)),
            size=(n_in, n_out)
        )
        if activation == 'logistic':
            W_values *= 4

        self.b = np.zeros(n_out, )

        self.grad_W = np.zeros(self.W.shape)
        self.grad_b = np.zeros(self.b.shape)

    def forward(self, input):
        '''
        :type input: numpy.array
        :param input: a symbolic tensor of shape (n_in,)
        '''
        lin_output = np.dot(input, self.W) + self.b
        self.output = (
            lin_output if self.activation is None
            else self.activation(lin_output)
        )
        self.input = input
        return self.output

    def backward(self, delta):
        self.grad_W = np.atleast_2d(self.input).T.dot(np.atleast_2d(delta))
        self.grad_b = delta
        # return delta_ for next layer
        delta_ = delta.dot(self.W.T) * self.activation_deriv(self.input)
        return delta_


class MLP:
    """
    """

    def __init__(self, layers, activation='relu', last_activation='softmax'):
        """
        :param layers: A list containing the number of units in each layer.
        Should be at least two values
        :param activation: The activation function to be used. Can be
        "logistic" or "tanh"
        """
        ### initialize layers
        self.layers = []
        self.params = []

        self.activation = last_activation
        for i in range(len(layers) - 2):
            self.layers.append(HiddenLayer(layers[i], layers[i + 1], activation=activation))

        i = len(layers) - 2

        self.layers.append(HiddenLayer(layers[i], layers[i + 1], activation=activation))

    def forward(self, input):
        for layer in self.layers:
            output = layer.forward(input)
            input = output
        return output

    def criterion_MSE(self, y, y_hat):
        activation_deriv = Activation(self.activation).f_deriv
        # MSE
        error = y - y_hat
        print(error)
        loss = error ** 2
        print(loss)
        # write down the delta in the last layer
        delta = error * activation_deriv(y_hat)
        # return loss and delta

        loss=np.sum(loss)
        return loss, delta

    def cross_entropy_loss(self, y, y_hat):
        activation_deriv = Activation(self.activation).f_deriv

        error = y_hat - y
        loss = np.sum(np.log(y_hat) * y)
        print(error)
        print(loss)

        #delta = error * activation_deriv(y_hat)

        return loss, delta

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

                num_classes = len(np.unique(y))
                grd_trth = np.array([0 for j in range(num_classes)], dtype=np.float64)

                grd_trth[int(y[i]) - 1] = 1

                loss[it], delta = self.criterion_MSE(y[i], y_hat)

#                loss[it], delta = self.cross_entropy_loss(grd_trth, y_hat)
                self.backward(delta)

                # update
                self.update(learning_rate)
            to_return[k] = np.mean(loss)
        return to_return

    def predict(self, x):
        x = np.array(x)
        output = np.zeros(x.shape[0])
        for i in np.arange(x.shape[0]):
            output[i] = nn.forward(x[i, :])
        return output