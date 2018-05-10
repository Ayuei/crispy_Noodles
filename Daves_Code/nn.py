import numpy as np
from layer import Layer
from softmax import Softmax
from minibatch import MiniBatch

class Model:
    def __init__(self, layers_dim):
        self.layers = []
        self.layerdim = layers_dim
        self.SM = Softmax()
        self.MB = MiniBatch()
        
        for i in range(len(layers_dim)-1):
            #For TanH activation ensure weights are positive
            #wx = np.random.randint(0, 100, size=(layers_dim[i],  layers_dim[i+1])) / 10000
            #bx = np.atleast_2d(np.array([np.random.randint(0, 100) / 1000 for i in range(layers_dim[i+1])]))
            
            #For leaky relu we want both positive and negative weights
            wx = np.random.randn(layers_dim[i], layers_dim[i+1]) / np.sqrt(layers_dim[i])
            bx = np.atleast_2d(np.random.randn(layers_dim[i+1]).reshape(1, layers_dim[i+1]))
            self.layers.append(Layer(wx, bx))
 
    def calculate_loss(self, X, y):
        output = X
        for i in self.layers:
            output = i.forward(output)
        return self.SM.loss(output, y)

    def predict(self, X):
        output = X
        for i in self.layers:
            output = i.forward(output)
        probs = self.SM.predict(output)
        return np.argmax(probs, axis=1)   

    def train(self, X, y, num_passes=1000, epsilon=0.01, reg_lambda=0.01, print_loss=False):
        epochLoss = []
        for epoch in range(num_passes):
            # Forward propagation
            minibatches = self.MB.mini_batches(X, y, 50)
            
            batchLoss = []
            for i, minibatch in enumerate(minibatches):
                
                (minibatch_X, minibatch_Y) = minibatch
                            
                output = minibatch_X
                
                for i in self.layers:
                    output = i.forward(output)
    
                delta = self.SM.diff(output, minibatch_Y)
                loss = self.SM.loss(output, minibatch_Y)
                batchLoss.append(loss)
                
                for i, layer in enumerate(reversed(self.layers)):
                    ix = len(self.layers) - i
                    delta = layer.backward(ix,
                                               delta, 
                                               epsilon, 
                                               reg_lambda)
                    
            epochLoss.append(np.mean(batchLoss))
            
            if print_loss and epoch % 100 == 0:
                print("Loss after iteration %i: %f" %(epoch, self.calculate_loss(X, y)))
            
        return epochLoss
                