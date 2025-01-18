import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

sys.path.append(os.path.abspath("Desktop/BasicML"))
from Scripts.SupportVectorMachineCLF import SupportVectorMachine

np.random.seed(1337)

X, y = make_classification(n_samples=200, n_features=2, n_redundant=0, n_classes=2,
                           n_clusters_per_class=1, class_sep=1, random_state=1337)

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=1337)

model = SupportVectorMachine()
model.train(Xtr, ytr)

yhat = model.predict(Xte)

plt.figure()
plt.title("Entire dataset")
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.legend()
plt.show(block=False)

plt.figure()
plt.scatter(Xte[:, 0], Xte[:, 1], c=yte, label="true")
plt.legend()
plt.show(block=False)

plt.figure()
plt.scatter(Xte[:, 0], Xte[:, 1], c=yhat, label="pred")
plt.legend()
plt.show()