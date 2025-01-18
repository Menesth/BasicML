import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

sys.path.append(os.path.abspath("Desktop/BasicML"))
from Scripts.LinearRegression import LinearRegressionGD

np.random.seed(1337)

SIGMA = 5.0

X, y = make_regression(n_samples=100, n_features=1, n_informative=1, noise=SIGMA, bias=2, random_state=1337)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=1337)

model = LinearRegressionGD(in_features=X.shape[-1], out_features=1)
model.train(Xtr, ytr, lr = 1e-3, epochs = 20, eval_epoch = 2)

yhat = model(Xte)

plt.figure()
plt.plot(Xte, yte, label="true", c="b")
plt.plot(Xte, yhat, label="pred", c="r")
plt.xlabel("x")
plt.xlabel("y")
plt.legend()
plt.show()