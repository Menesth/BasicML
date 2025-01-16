import numpy as np
import matplotlib.pyplot as plt

class LinearRegressionGD():
    
    def __init__(self, in_features, out_features):
        self.weights = np.random.normal(0.0, 0.1, size=(in_features, out_features))
        self.bias = np.random.uniform(-0.1, 0.1, size=(out_features,))

    
    def __call__(self, X):
        return np.dot(X, self.weights) + self.bias
    
    def train(self, X, y, lr = 1e-3, epochs = 100, eval_epoch = 10, plot = True):
        losses = list()
        y = np.expand_dims(y, axis=1)
        for epoch in range(epochs):
            yhat = self(X)
            error = y - yhat
            weights_grad = -2 * np.dot(X.T, error)
            bias_grad = -2 * np.sum(error, axis=0)

            self.weights -= lr * weights_grad
            self.bias -= lr * bias_grad
            
            if plot:
                if epoch % eval_epoch == 0:
                    loss_epoch = np.sum(error ** 2)
                    losses.append(loss_epoch.item())
        
        if plot:
            plt.figure()
            plt.plot(list(range(epochs//eval_epoch)), losses, label="loss")
            plt.show(block=False)