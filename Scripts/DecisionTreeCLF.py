import numpy as np

class Node():

    def __init__(self, feature=None, threshold=None, left=None, right=None, val=None):
        self.feature=feature
        self.threshold=threshold
        self.left=left
        self.right=right
        self.val=val
    
    def is_leaf_node(self):
        return self.val is not None

class DecisionTree:

    def __init__(self, min_sample_split=2, max_depth=10, Nfeatures=None):
        self.min_sample_split=min_sample_split
        self.max_depth=max_depth
        self.Nfeatures=Nfeatures
        self.root=None

    def train(self, X, y):
        self.Nfeatures = min(X.shape[1], self.Nfeatures) if self.Nfeatures else X.shape[1]
        self.root = self._growtree(X, y)

    def _growtree(self, X, y, depth=0):
        Nsamples, Nfeatures = X.shape
        Nlabels = len(np.unique(y))

        stop = (depth >= self.max_depth) or (Nlabels == 1) or (Nsamples < self.min_sample_split)
        if stop:
            leaf_val = self._majvote(y)
            return Node(val=leaf_val)

        rand_idx = np.random.choice(Nfeatures, self.Nfeatures, replace=False)
        best_threshold, best_feature = self._bestsplit(X, y, rand_idx)
        left_split, right_split = self._split(X[:, best_feature], best_threshold)
        left_child = self._growtree(X[left_split, :], y, depth + 1)
        right_child = self._growtree(X[right_split, :], y, depth + 1)
        return Node(feature=best_feature, threshold=best_threshold, left=left_child, right=right_child)
        
    def _majvote(self, y):
        classes, counts = np.unique(y, return_counts=True)
        return classes[np.argmax(counts)]
    
    def _bestsplit(self, X, y, idx):
        best_info_gain = -1
        split_threshold, split_idx = None, None

        for i in idx:
            X_col = X[:, i]
            thresholds = np.unique(X_col)

            for threshold in thresholds:
                info_gain = self._information_gain(X_col, y, threshold)
                if info_gain > best_info_gain:
                    best_info_gain = info_gain
                    split_idx = i
                    split_threshold = threshold
    
        return split_threshold, split_idx

    def _information_gain(self, X, y, threshold):
        parent_entropy = self._entropy(y)
        left_child, right_child = self._split(X, threshold)
        if (len(left_child) == 0 or len(right_child) == 0):
            return 0

        left_weight, right_weight = len(left_child) / len(y), len(right_child) / len(y)
        left_entropy = left_weight * self._entropy(left_child)
        right_entropy = right_weight * self._entropy(right_child)
        return parent_entropy - (left_entropy + right_entropy)

    def _entropy(self, y):
        probabilites = np.bincount(y) / len(y)
        return -np.sum([p * np.log2(p) for p in probabilites if p > 0])

    def _split(self, X, threshold):
        left_child = np.argwhere(X <= threshold).flatten()
        right_child = np.argwhere(X > threshold).flatten()
        return left_child, right_child

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])
    
    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.val
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)