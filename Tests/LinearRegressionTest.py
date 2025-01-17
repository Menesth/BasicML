import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_regression

sys.path.append(os.path.abspath("Desktop/BasicML"))
from Scripts.LinearRegression import LinearRegressionGD

np.random.seed(1337)

SIGMA = 1.0

X, y = make_regression(n_samples=100, n_features=1, noise=5, random_state=1337)

model = LinearRegressionGD(in_features=X.shape[-1], out_features=1)
model.train(X, y, lr = 1e-3, epochs = 20, eval_epoch = 2)

yhat = model(X)

plt.figure()
plt.plot(X, y, label="true", c="b")
plt.plot(X, yhat, label="pred", c="r")
plt.xlabel("x")
plt.xlabel("y")
plt.legend()
plt.show()