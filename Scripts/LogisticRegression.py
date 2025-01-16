import numpy as np
import matplotlib.pyplot as plt

class LogisticRegressionGD():
    
    def __init__(self, in_features, out_features):
        self.weights = np.random.normal(0.0, 0.1, size=(in_features, out_features))
        self.bias = np.random.uniform(-0.1, 0.1, size=(out_features,))

    
    def __call__(self, X):
        return np.dot(X, self.weights) + self.bias
    
    def train(self, X, y, lr = 1e-3, epochs = 100, eval_epoch = 10, plot = True):
        losses = list()
        y = np.expand_dims(y, axis=1)
    
        for epoch in range(epochs):
            logits = self(X)
            sigmoid = 1 / 1 + np.exp(logits)

            dsigmoid = np.sum(y * (1 - sigmoid) - (1 - y) * sigmoid)
            dlogits = logits * (1 - logits) * dsigmoid
            dweights = X * dlogits
            dbias = dlogits

            self.weights -= lr * 
            self.bias -= lr * 
            
            if plot:
                if epoch % eval_epoch == 0:
                    loss_epoch = None
                    losses.append(loss_epoch.item())
        
        if plot:
            plt.figure()
            plt.plot(list(range(epochs//eval_epoch)), losses, label="loss")
            plt.show(block=False)