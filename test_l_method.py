from dataset import *
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors
from l_method import agglomerative_l_method
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA

def get_cmap(N):
    '''Returns a function that maps each index in 0, 1, ... N-1 to a distinct
        RGB color.'''
    color_norm = colors.Normalize(vmin=0, vmax=N - 1)
    scalar_map = cmx.ScalarMappable(norm=color_norm, cmap='hsv')

    def map_index_to_rgb_color(index):
        return scalar_map.to_rgba(index)

    def static(index):
        return 'rgb'[index]

    return map_index_to_rgb_color

dataset = get_iris()

pca = PCA(2)
pca.fit(dataset.X)
X = pca.transform(dataset.X)
# X = dataset.X
# X = list(map(lambda x: x[:2], dataset.X))

print('X:', X)

l_method = agglomerative_l_method(X)
clusters_cnt = len(l_method.cluster_centers_)

# agg = AgglomerativeClustering(clusters_cnt).fit(X)
# labels = agg.labels_
labels = l_method.labels_

print('X:', X)

X_by_label = {}
for x, label in zip(X, labels):
    if label not in X_by_label:
        X_by_label[label] = []

    X_by_label[label].append(x)

def plot(X, **kwargs):
    x, y, *_ = list(zip(*X))
    plt.scatter(x, y, **kwargs)

# plot(X, color='grey')

cmap = get_cmap(clusters_cnt + 1)
for label in range(clusters_cnt):
    plot(X_by_label[label], color=cmap(label))

plt.show()