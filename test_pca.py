from sklearn.decomposition import PCA
from dataset import *
import matplotlib.pyplot as plt

dataset = get_pendigits()

pca = PCA(n_components=2)
pca.fit(dataset.X)
X = pca.transform(dataset.X)

print('covariance:', pca.get_covariance())

x, y = list(zip(*X))

plt.scatter(x, y)
plt.show()