from dataset import *
from pipe import Pipe
from wrapper import *
from badness import *
from ssltools import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from l_method import agglomerative_l_method

dataset = get_pendigits()

def seed_randomly(prob):
    pipe = Pipe() \
        .x(dataset.X) \
        .y(dataset.Y) \
        .y_seed(seeding_random(prob)) \
        .connect(stop())
    return pipe['y_seed']

def seed_some(prob, clusters_cnt):
    pipe = Pipe() \
        .x(dataset.X) \
        .y(dataset.Y) \
        .y_seed(seeding_some(prob, clusters_cnt)) \
        .connect(stop())
    return pipe['y_seed']

def seed_cache(file):
    file = 'seeding/' + file
    cache = StorageCache(file)
    y_seed = np.array(cache.get())
    return y_seed

badness_engine = KmeansMockingNested(dataset.cluster_cnt * 3, dataset.X)

random_cache = seed_cache('pendigits_some-5-prob-0.1.json')
badness = badness_engine.run(random_cache)
print('badness cache:', badness)

total_cnt = 0
all_certain_cnt = 0
all_uncertain_cnt = 0
for group in badness_engine.groups:
    seeding_cnt = group.seeding_cnt()
    total_cnt += group.cnt

    if seeding_cnt == 0:
        continue

    print('total_cnt:', group.cnt)
    print('seeding_cnt:', seeding_cnt, '/', group.cnt)


    centroids = group.clustering_model.cluster_centers_

    count_by_group = Counter(group.clustering_model.labels_)
    seeds = list(filter(lambda xy: xy[1] is not None,
                        zip(group.X, group.y_seed)))

    seed_x, seed_y = list(zip(*seeds))
    seed_groups = group.clustering_model.predict(seed_x)
    certain_cnt = sum(count_by_group[cluster] for cluster in set(seed_groups))
    uncertain_cnt = group.cnt - certain_cnt

    all_certain_cnt += certain_cnt
    all_uncertain_cnt += uncertain_cnt

    print('certain_cnt:', certain_cnt, 'total:', all_certain_cnt)
    print('uncertain_cnt:', uncertain_cnt, 'total:', all_uncertain_cnt)

    print('cluster_cnt:', len(centroids))

    pca = PCA(n_components=3)
    pca.fit(group.X)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    def plot(X, **kwargs):
        X = pca.transform(X)
        x, y, z = list(zip(*X))
        ax.scatter(x, y, z, **kwargs)

    plot(group.X)
    plot(centroids, c='red')
    plt.show()

# badness = badness_engine.run(seed_some(0.1, 1), 1e-20)
# print('badness some 1:', badness)