from Scripts.LinearRegression import LinearRegressionGD
import matplotlib.pyplot as plt
import torch

SIGMA = 1.0

X = torch.arange(start=-5.0, end=5.0, step=0.1)
y = 2 * X + 1 + torch.normal(mean=0, std=SIGMA**2, size=X.shape)

model = LinearRegressionGD(in_features=X.shape[0], out_features=1)
model.train(X, y)


plt.plot(X, y, label="true")
plt.xlabel("x")
plt.xlabel("y")
plt.legend()
plt.show()