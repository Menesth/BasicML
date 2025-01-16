import torch
import matplotlib.pyplot as plt

class LinearRegressionGD():
    
    def __init__(self, in_features, out_features):
        self.weights = torch.empty(size=(in_features, out_features)).normal_(0.0, 0.1)
        self.bias = torch.empty(size=(out_features, )).uniform_(-0.1, 0.1)

    
    def __call__(self, X):
        return torch.matmul(X, self.weights) + self.bias
    
    def train(self, X, y, lr = 1e-3, epochs = 100, eval_epoch = 10, plot = True):
        losses = list()
        for epoch in range(epochs):

            WeightGrad = -2 * X.T @ (y - X @ self.weights - self.bias)
            BiasGrad = -2 * (y - X @ self.weights - self.bias)

            self.weights -= lr * WeightGrad
            self.bias -= lr * BiasGrad
            
            if plot:
                if epoch % eval_epoch == 0:
                    loss_epoch = torch.sum((y - self(X)) ** 2)
                    losses.append(loss_epoch.item())
        
            if plot:
                plt.figure()
                plt.plot(range(list(epochs//eval_epoch)), losses, label="loss")
                plt.show()