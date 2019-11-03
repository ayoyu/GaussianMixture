from GMM import GaussianMixture
import numpy as np
import matplotlib.pyplot as plt

gmm = GaussianMixture(2, n_components=3, n_init=10)
X = np.load("samples.npz")["data"]

loss = gmm.fit(X)
print(loss)
gamma = gmm.predict_proba(X)
labels = np.argmax(gamma, axis=1)
plt.scatter(X[:, 0], X[:, 1], c=labels, s=30)
plt.axis('equal')
plt.show()


