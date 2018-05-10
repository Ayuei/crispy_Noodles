import math
import copy
import numpy as np
from softmax import Softmax

class Layer:
    def __init__(self, W, b, activation="LeakyRelu", keepProb=0.90):
        self.W = W
        self.b = b
        self.activation = activation
        self.epsilon = 1e-9
        
        self.input = []
        self.matMulOutput = [] 
        self.linearOutput = []
        self.output = []
        
        
        self.v_dW = np.zeros((np.shape(W)))
        self.v_db = np.zeros((np.shape(b)))
        self.s_dW = np.zeros((np.shape(W)))
        self.s_db = np.zeros((np.shape(b)))
        
        
        self.keepProb = keepProb
    
    def activate_forward(self, X):
        if self.activation=="TanH":
            return np.tanh(X)
        elif self.activation=="LeakyRelu":
             dx = copy.deepcopy(X)
             dx[dx==0] = 0.01
             dx[dx<0] *= 0.01
             return dx

    def activate_backward(self, X, delta):
        if self.activation=="TanH":
            output = self.activate_forward(X)
            return (1.0 - np.square(output)) * delta
        elif self.activation=="LeakyRelu":
            dx = np.ones_like(X)
            dx[X <= 0] = 0.01
            return dx * delta
        
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
        
        self.dropMask = np.random.randn(np.shape(self.output)[0], np.shape(self.output)[1])
        self.dropMask = self.dropMask < self.keepProb
        self.output = self.output * self.dropMask
        self.output = self.output / self.keepProb

        return self.output
    
    def backward(self, i, delta, lr, regularization,beta1,beta2,t):
        
        #print('I:%s | W:%s | Input:%s' % (str(i), np.shape(self.W),np.shape(self.input)))
        dadd = self.activate_backward(self.linearOutput, delta)
       
        dadd = np.multiply(dadd, self.dropMask)
        dadd = dadd / self.keepProb 
        
        db, deriv_mul = self.biasBackwards(self.matMulOutput, self.b, dadd)
        dW, new_delta = self.weightsBackwards(self.W, self.input, deriv_mul)
        

        #==========Adam optimizer==============
        dW += regularization * self.W
         
        self.v_dW = (beta1 * self.v_dW) + ((1-beta1) * dW)
        self.v_db = (beta1 * self.v_db) + ((1-beta1) * db)
        
        vcW = self.v_dW / (1-(beta1)**t)
        vcb = self.v_db / (1-(beta1)**t)
        
        self.s_dW = (beta2 * self.s_dW) + ((1-beta2) * dW**2)
        self.s_db = (beta2 * self.s_db) + ((1-beta2) * db**2)
       
        scW = self.s_dW / (1-(beta2)**t)
        scb = self.s_db / (1-(beta2)**t)
                
        self.W = self.W - (lr * (vcW / (np.sqrt(scW) + self.epsilon)))
        self.b = self.b - (lr * (vcb / (np.sqrt(scb) + self.epsilon)))
        #======================================
        
        #dW += regularization * self.W
        #self.b += -lr * db
        #self.W += -lr * dW
        return new_delta
               


##================DROP OUT - WORKING=========================       
#     
#    def forward(self, X):
#        self.input = X
#        self.matMulOutput = np.dot(X, self.W)
#        self.linearOutput = self.matMulOutput + self.b
#        self.output = self.activate_forward(self.linearOutput)
#        
#        self.dropMask = np.random.randn(np.shape(self.output)[0], np.shape(self.output)[1])
#        self.dropMask = self.dropMask < self.keepProb
#        self.output = self.output * self.dropMask
#        self.output = self.output / self.keepProb
#
#        return self.output
#    
#    def backward(self, i, delta, lr, regularization):
#        #print('I:%s | W:%s | Input:%s' % (str(i), np.shape(self.W),np.shape(self.input)))
#        dadd = self.activate_backward(self.linearOutput, delta)
#       
#        dadd = np.multiply(dadd, self.dropMask)
#        dadd = dadd / self.keepProb 
#        
#        db, deriv_mul = self.biasBackwards(self.matMulOutput, self.b, dadd)
#        dW, new_delta = self.weightsBackwards(self.W, self.input, deriv_mul)
#        
#        dW += regularization * self.W
#        self.b += -lr * db
#        self.W += -lr * dW
#        return new_delta
#               
    
    
    
#=========================================
#ORIGINAL NON-DROPOUT VERSION
#    def forward(self, X):
#        self.input = X
#        self.matMulOutput = np.dot(X, self.W)
#        self.linearOutput = self.matMulOutput + self.b
#        self.output = self.activate_forward(self.linearOutput)
#        return self.output     
#
#        
#    def backward(self, i, delta, lr, regularization):
#        #print('I:%s | W:%s | Input:%s' % (str(i), np.shape(self.W),np.shape(self.input)))
#        dadd = self.activate_backward(self.linearOutput, delta)
#                
#        db, deriv_mul = self.biasBackwards(self.matMulOutput, self.b, dadd)
#        dW, new_delta = self.weightsBackwards(self.W, self.input, deriv_mul)
#        
#        dW += regularization * self.W
#        self.b += -lr * db
#        self.W += -lr * dW
#        return new_delta
#        
    
        
        
        
        