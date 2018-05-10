import numpy as np

class adam:
    def __init__(self, layer, regularization=0.01, beta1=0.9, beta2=0.999, epsilon=0.01):
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
