import numpy as np

class Softmax:
    def predict(self, X):
        exp_scores = np.exp(X)
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    def loss(self, X, y):
        num_examples = X.shape[0]
        probs = self.predict(X)
                
        if y.ndim > 1:
            yg  = np.argmax(y, axis=1)
        else:
            yg  = y
            
        corect_logprobs = -np.log(probs[range(num_examples), yg])
        
        loss = np.mean(corect_logprobs) - 1
        
        return loss

    def diff(self, X, y):
        num_examples = X.shape[0]
        probs = self.predict(X)
                
        if y.ndim > 1:
            yg  = np.argmax(y, axis=1)
        else:
            yg  = y
                
        probs[range(num_examples), yg] -= 1
        return probs