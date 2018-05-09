import math
import numpy as np
from softmax import Softmax

class Layer:
    def __init__(self, W, b, activation="TanH"):
        self.W = W
        self.b = b
        self.activation = activation
        self.input = []
        
        self.matMulOutput = [] 
        
        self.linearOutput = []
        
        self.output = []
    
    def activate_forward(self, X):
        if self.activation=="TanH":
            return np.tanh(X)

    def activate_backward(self, X, delta):
        if self.activation=="TanH":
            output = self.activate_forward(X)
            return (1.0 - np.square(output)) * delta
        
    def biasBackwards(self, X, b, delta):
        dX = delta * np.ones_like(X)
        db = np.dot(np.ones((1, delta.shape[0]), dtype=np.float64), delta)
        return db, dX
    
    def weightsBackwards(self, W, X, delta):
        dW = np.dot(np.transpose(X), delta)
        dX = np.dot(delta, np.transpose(W))
        return dW, dX
        
    def forward(self, X):
        self.input = X
        self.matMulOutput = np.dot(X, self.W)
        self.linearOutput = self.matMulOutput + self.b
        self.output = self.activate_forward(self.linearOutput)
        return self.output
        
    def backward(self, i, delta, lr, regularization):
        #print('I:%s | W:%s | Input:%s' % (str(i), np.shape(self.W),np.shape(self.input)))
        dadd = self.activate_backward(self.linearOutput, delta)
        db, deriv_mul = self.biasBackwards(self.matMulOutput, self.b, dadd)
        dW, new_delta = self.weightsBackwards(self.W, self.input, deriv_mul)
        
        dW += regularization * self.W
        self.b += -lr * db
        self.W += -lr * dW
        return new_delta
        
    
        
        
        
        
        
        
        