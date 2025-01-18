import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification

sys.path.append(os.path.abspath("Desktop/BasicML"))
from Scripts.KNearestNeighboorsCLF import KNearestNeighboors

np.random.seed(1337)

SIGMA = 1.0

X, y = make_classification(n_samples=100, n_features=2, n_redundant=0, n_classes=2,
                           n_clusters_per_class=1, class_sep=1, random_state=1337)

model = KNearestNeighboors(in_features=X.shape[-1], out_features=1)
model.train(X, y, lr = 1e-3, epochs = 500, eval_epoch = 10)

yhat = model.predict(X)
plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=y, label="true")
plt.legend()
plt.show(block=False)

plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=yhat, label="pred")
plt.legend()
plt.show()