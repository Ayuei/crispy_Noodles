import numpy as np

class Adam:
    def __init__(self, layer, regularization=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.reg = regularization
        self.beta1 = beta1
        self.beta2 = beta2
        self.v_dW = np.zeros((np.shape(layer.W)))
        self.v_db = np.zeros((np.shape(layer.b)))
        self.s_dW = np.zeros((np.shape(layer.W)))
        self.s_db = np.zeros((np.shape(layer.b)))
        self.layer = layer
        self.epsilon = epsilon

    def backward(self, delta, **kwargs):

        t = kwargs["global_time_step"]
        last_layer = kwargs["last_layer"]

        delta = self.layer.backward(delta, last_layer)

        self.layer.grad_W += self.reg * self.layer.W

        self.v_dW = (self.beta1 * self.v_dW) + ((1 - self.beta1) * self.layer.grad_W)
        self.v_db = (self.beta1 * self.v_db) + ((1 - self.beta1) * self.layer.grad_b)

        self.vcW = self.v_dW / (1 - (self.beta1) ** t)
        self.vcb = self.v_db / (1 - (self.beta1) ** t)


        self.s_dW = (self.beta2 * self.s_dW) + ((1 - self.beta2) * self.layer.grad_W ** 2)
        self.s_db = (self.beta2 * self.s_db) + ((1 - self.beta2) * self.layer.grad_b ** 2)


        self.scW = self.s_dW / (1 - (self.beta2) ** t)
        self.scb = self.s_db / (1 - (self.beta2) ** t)

        return delta

    def forward(self, X, **kwargs):
        return self.layer.forward(X)

    def update(self, lr):
        self.layer.W = self.layer.W - (lr * (self.vcW / (np.sqrt(self.scW) + self.epsilon)))
        self.layer.b = self.layer.b - (lr * (self.vcb / (np.sqrt(self.scb) + self.epsilon)))

class SGDMomentum:
    def __init__(self, layer, gamma=0.9):
        self.gamma = gamma
        self.layer = layer
        self.velocity = None
        self.velocity_b = None

    def forward(self, X, **kwargs):
        return self.layer.forward(X)

    def backward(self, delta, **kwargs):
        last_layer = kwargs["last_layer"]

        delta = self.layer.backward(delta, last_layer)

        return delta

    def update(self, lr):
        if self.velocity is None:
            self.velocity = np.zeros_like(self.layer.grad_W)
            self.velocity_b = np.zeros_like(self.layer.grad_b)

        self.velocity = self.gamma * self.velocity + lr * self.layer.grad_W
        self.velocity_b = self.gamma * self.velocity_b + lr * self.layer.grad_b
        self.layer.grad_W -= self.velocity
        self.layer.grad_b -= self.velocity_b

        self.layer.W = self.layer.W - (lr*self.layer.grad_W)
        self.layer.b = self.layer.b - (lr*self.layer.grad_b)
