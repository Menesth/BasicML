import numpy as np

class SupportVectorMachine():
    
    def __init__(self):
        self.weight = None
        self.bias = 0
    
    def train(self, X, y, lr=1e-3, l2reg=1e-2, epochs=100):
        N, p = X.shape
        y = np.where(y <= 0.5, -1, 1)
        self.weight = np.zeros(shape=(p, ))

        for _ in range(epochs):
            for i, xi in enumerate(X):
                yi = y[i]
                hyperplan = np.linalg.matmul(xi, self.weight) - self.bias
                if yi * hyperplan >= 1:
                    dweight = 2 * l2reg * self.weight
                    dbias = 0
                else:
                    dweight = 2 * ((l2reg * self.weight) - (yi * xi))
                    dbias = yi
                self.weight -= lr * dweight
                self.bias -= lr * dbias


    def predict(self, X):
        hyperplan = np.linalg.matmul(X, self.weight) - self.bias
        out = np.sign(hyperplan)
        return np.where(out <= 0, 0, 1)