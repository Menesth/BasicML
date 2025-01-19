import numpy as np
from Scripts.DecisionTreeCLF import DecisionTree

class RandomForest():
    
    def __init__(self, Ntrees=100, max_depth=5, min_sample_split=2, Nfeatures=None, criterion="entropy"):
        self.Ntrees = Ntrees
        self.max_depth = max_depth
        self.min_sample_split = min_sample_split
        self.Nfeatures = Nfeatures
        self.criterion = criterion

    def train(self, X, y):
        self.trees = []
        for _ in range(self.Ntrees):
            tree = DecisionTree(max_depth=self.max_depth, min_sample_split=self.min_sample_split,
                                Nfeatures=self.Nfeatures, criterion=self.criterion)
            Xsample, ysample = self._boostrap(X, y)
            tree.train(Xsample, ysample)
            self.trees.append(tree)

    def _boostrap(self, X, y):
        Nsample = X.shape[0]
        randidx = np.random.choice(Nsample, Nsample, replace=True)
        return X[randidx], y[randidx]


    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        preds = np.transpose(tree_preds)
        return np.array([self._majvote(pred) for pred in preds])
    
    def _majvote(self, y):
        classes, counts = np.unique(y, return_counts=True)
        return classes[np.argmax(counts)]