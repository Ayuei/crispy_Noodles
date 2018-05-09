import math
import numpy as np

class MiniBatch:
    def mini_batches(self, X, Y, batchsize = 50):
 
        m = X.shape[0]          
        mini_batches = []
                   
        permutation = list(np.random.permutation(m))
        
        shuffled_X = X[permutation]
        shuffled_Y = Y[permutation]
           
        num_complete_minibatches = math.floor(m/batchsize) 
        for k in range(0, num_complete_minibatches):
            mini_batch_X = shuffled_X[(k*batchsize) : (k+1) * batchsize]
            mini_batch_Y = shuffled_Y[(k*batchsize) : (k+1) * batchsize]
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)
        
        if m % batchsize != 0:
            mini_batch_X = shuffled_X[-(m % batchsize) : m]
            mini_batch_Y = shuffled_Y[-(m % batchsize) : m]
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)
            
        return mini_batches