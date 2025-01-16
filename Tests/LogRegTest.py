import sys
import os
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.abspath("Desktop/BasicML"))
from Scripts.LogisticRegression import LogRegressionGD

SIGMA = 1.0

X = np.arange(start=-5.0, stop=5.0, step=0.1).reshape(-1, 1)
y = 2 * X + 1 + np.random.normal(loc=0, scale=SIGMA**2, size=X.shape)

model = LogRegressionGD(in_features=X.shape[-1], out_features=1)
model.train(X, y, lr = 1e-3, epochs = 20, eval_epoch = 2)

yhat = model(X)

plt.figure()
plt.plot(X, y, label="true", c="b")
plt.plot(X, yhat, label="pred", c="r")
plt.xlabel("x")
plt.xlabel("y")
plt.legend()
plt.show()