from agglomerative_clustering import AgglomerativeClustering
from l_method import agglomerative_l_method
import numpy as np
from util import get_cmap
import matplotlib.pyplot as plt
from itertools import product


def uniform(count):
    return np.random.uniform(0, 1, (count, 2))

def normal(center, size, count):
    x, y = center
    X = np.random.normal(x, size, count)
    Y = np.random.normal(y, size, count)
    return np.array(list(zip(X, Y)))

def plot(ax, X, **kwargs):
    x1, x2 = list(zip(*X))
    ax.scatter(x1, x2, **kwargs)

distributions = {
    'uniform': uniform(1000),
    'normal': normal((0.5, 0.5), 0.15, 1000),
    'many_normal': np.concatenate((
        normal((0.2, 0.8), 0.05, 100),
        normal((0.8, 0.8), 0.05, 100),
        normal((0.2, 0.2), 0.05, 100),
        normal((0.8, 0.2), 0.05, 100),
        normal((0.35, 0.55), 0.05, 100),
        normal((0.45, 0.45), 0.05, 100),
        normal((0.55, 0.55), 0.05, 100),
        normal((0.65, 0.45), 0.05, 100),
    ))
}

methods = [
    'ward', 'average', 'complete', 'single'
]

def l_method(ax, X, method):
    l_method = agglomerative_l_method(X, method=method)
    suggest_n = len(l_method.cluster_centers_)
    cmap = get_cmap(suggest_n + 1)
    for label in range(suggest_n):
        XX = list(map(lambda xy: xy[0],
                      filter(lambda xy: xy[1] == label,
                             zip(X, l_method.labels_))))
        plot(ax, XX, c=cmap(label), edgecolors='none')

fig, axes = plt.subplots(len(methods), len(distributions))

for (i, method), (j, (dist_name, X)) in product(enumerate(methods),
                                                enumerate(distributions.items())):
    plt.sca(axes[i][j])
    plt.title(dist_name + ' ' + method)
    plt.axis('off')
    l_method(axes[i][j], X, method)

plt.show()