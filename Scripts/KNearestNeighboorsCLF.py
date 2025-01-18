import numpy as np

class KNearestNeighboors():
    
    def __init__(self, k = 3, p = 2):
        self.k = k
        self.p = p

    def train(self, X, y):
        self.X = X
        self.y = y
    
    def _distance(self, x1, x2):
        return np.sum((x1 - x2) ** self.p)

    def _majvote(self, y):
        classes, counts = np.unique(y, return_counts=True)
        return classes[np.argmax(counts)]

    def predict(self, X):
        out = list()
        for x in X:
            idx_incr_distances = np.argsort([self._distance(x, xtr) for xtr in self.X])
            k_nearest_Xtr = idx_incr_distances[:self.k]
            k_nearest_ytr = self.y[k_nearest_Xtr]
            kmost_frequent = self._majvote(k_nearest_ytr)
            out.append(kmost_frequent)
        return out