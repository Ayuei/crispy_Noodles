import numpy as np

class Softmax:
    def predict(self, X):
        exp_scores = np.exp(X)
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    def loss(self, X, y):
        num_examples = X.shape[0]
        probs = self.predict(X)
                
        if y.ndim > 1:
            yGuess  = np.argmax(y, axis=1)
        else:
            yGuess  = y
            
        corect_logprobs = -np.log(probs[range(num_examples), yGuess])
        
        data_loss = np.sum(corect_logprobs)
        
        return 1./num_examples * data_loss

    def diff(self, X, y):
        num_examples = X.shape[0]
        probs = self.predict(X)
                
        if y.ndim > 1:
            yGuess  = np.argmax(y, axis=1)
        else:
            yGuess  = y
                
        probs[range(num_examples), yGuess] -= 1
        return probs