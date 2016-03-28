from dataset import *
from sklearn.decomposition import PCA
from cache import StorageCache
from kmeans_mocking_nested_split import KmeansMockingNestedSplit
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors
from pipe import Pipe
from pipetools import *
from ssltools import *

dataset = get_yeast()

# pca = PCA(2)
# pca.fit(dataset.X)

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

def seed_cache(file):
    file = 'seeding/' + file
    cache = StorageCache(file)
    y_seed = np.array(cache.get())
    return y_seed

def seed_randomly(prob):
    pipe = Pipe() \
        .x(dataset.X) \
        .y(dataset.Y) \
        .y_seed(seeding_random(prob)) \
        .connect(stop())
    return pipe['y_seed']

badness_engine = KmeansMockingNestedSplit(dataset.cluster_cnt * 3, dataset.X)

random_cache = seed_cache('yeast_prob-0.05.json')
random_seed = seed_randomly(1.0)
badness = badness_engine.run(random_cache)
print('badness:', badness)

total_cnt = 0
for group in badness_engine.groups:
    seeding_cnt = group.seeding_cnt()
    total_cnt += group.cnt

    if seeding_cnt == 0:
        continue

    print('total_cnt:', total_cnt)
    print('seeding_cnt:', seeding_cnt, '/', group.cnt)

    seeds = group.seeds()
    seed_x, seed_y = list(map(list, zip(*seeds)))

    model = group.clustering_model
    label_cnt = model.latest_label
    print('label_cnt:', label_cnt)

    cmap = get_cmap(label_cnt + 1)

    group_sizes = [(label, len(model.get_X(label))) for label in range(label_cnt)]
    largest_group, _ = max(group_sizes, key=lambda x: x[1])

    print('largest_group:', largest_group, 'cnt:', len(model.get_X(largest_group)))

    centroid = model.get_centroid(model.get_X_with_idx(largest_group))
    print('centroid:', centroid)

    pca = PCA(2)
    pca.fit(group.X)

    pca_centroid = pca.transform([centroid])[0]
    print('pca_centroid:', pca_centroid)

    c_x = pca_centroid[0]
    c_y = pca_centroid[1]

    def plot(X, **kwargs):
        X = pca.transform(X)
        x1, x2 = list(zip(*X))
        plt.scatter(x1, x2, **kwargs)

    for label in range(label_cnt):
        X = model.get_X(label)
        print('label:', label, 'cnt:', len(X))
        plot(X, c=cmap(label), edgecolors='none')

    plot(seed_x, c='black')
    # plt.xlim(-1 + c_x, 1 + c_x)
    # plt.ylim(-1 + c_y, 1 + c_y)

    plt.show()

