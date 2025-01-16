import torch
import matplotlib.pyplot as plt

class LogisticRegressionGD():
    
    def __init__(self, in_features, out_features):
        self.weights = torch.empty(size=(in_features, out_features)).normal_(0.0, 0.1)
        self.bias = torch.empty(size=(out_features, )).uniform_(-0.1, 0.1)

    
    def __call__(self, X):
        lin_com = torch.matmul(X, self.weights) + self.bias
        return torch.sigmoid(lin_com)
    
    def train(self, X, y, lr = 1e-3, epochs = 100, eval_epoch = 10, plot = True):
        losses = list()
        for epoch in range(epochs):

            sigmoidGrad = None
            weightsGrad = None * sigmoidGrad
            biasGrad = None * sigmoidGrad

            self.weights -= lr * weightsGrad
            self.bias -= lr * biasGrad
            
            if plot:
                if epoch % eval_epoch == 0:
                    loss_epoch = torch.sum((y - self(X)) ** 2)
                    losses.append(loss_epoch.item())
        
        if plot:
            plt.figure()
            plt.plot(list(range(epochs//eval_epoch)), losses, label="loss")
            plt.show(block=False)