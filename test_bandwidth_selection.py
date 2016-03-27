from bandwidth_selection import BandwidthSelection
from dataset import *
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

dataset = get_pendigits()

# X = dataset.X
pca = PCA(2)
pca.fit(dataset.X)

pca_X = pca.transform(dataset.X)
# print('X:', X)

bandwidths = {
    'gaussian_dist': BandwidthSelection.gaussian_distribution(pca_X),
    'likelihood_10': BandwidthSelection.cv_maximum_likelihood(pca_X),
    'likelihood_100': BandwidthSelection.cv_maximum_likelihood(pca_X, search=np.linspace(1e-4, 1, 100)),
}

# Set up the figure
f, axes = plt.subplots(int(len(bandwidths / 4)) + 1, 4, figsize=(8, 8))
x, y = list(map(np.array, zip(*pca_X)))
bandwidths = sorted([(k, v) for k, v in bandwidths.items()], key=lambda x: x[0])

for ax, (name, bandwidth) in zip(axes.flat, bandwidths):
    ax.set_aspect("equal")

    # Draw the two density plots
    sns.kdeplot(x, y, bw=bandwidth,
                cmap="Reds", shade=True, shade_lowest=False, ax=ax)
    ax.set(xlim=(-1, 1), ylim=(-1, 1))
    plt.sca(ax)
    plt.title(name)

plt.show()