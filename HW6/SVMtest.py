import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from MLutils import SVM

np.random.seed(42)
cov_1 = np.array([[6, -3], [-3, 4]])
cov_2 = np.array([[6, -3], [-3, 2]])
data_1 = np.random.multivariate_normal([5, 5], cov_1, size=800)
data_2 = np.random.multivariate_normal([0, 0], cov_2, size=800)
X = np.r_[data_1, data_2]
X = StandardScaler().fit_transform(X)
y = np.r_[np.full(len(data_1), 1), np.full(len(data_1), -1)]
train = pd.DataFrame(np.c_[X, y])
print(f'{X.shape=}')
print(f'{y.shape=}')

plt.plot(X[:800, 0], X[:800, 1], '.', alpha=0.5, color='red')
plt.plot(X[800:, 0], X[800:, 1], '.', alpha=0.5, color='blue')
plt.axis('equal')
plt.grid()
plt.show()

svm = SVM()
svm.fit(train)


def make_meshgrid(X, step=0.02):
    x, y = X[:, 0], X[:, 1]
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, step),
                         np.arange(y_min, y_max, step))
    return xx, yy


def plot_decision_boundary(svm, X, **params):
    xx, yy = make_meshgrid(X)

    # creates matrix of shape (N, 2) samples
    X = np.c_[xx.ravel(), yy.ravel()]

    # convert to dataframe and add a y column
    test = pd.DataFrame(X)
    test['y'] = 0

    z = svm.predict(test)
    z = z.reshape(xx.shape)
    return plt.contourf(xx, yy, z, **params)


plot_decision_boundary(svm, X, cmap=plt.cm.coolwarm, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y,
            cmap=plt.cm.coolwarm, s=20, edgecolors='k')
plt.show()
# %%
