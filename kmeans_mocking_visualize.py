from dataset import *
from pipe import Pipe
from wrapper import *
from badness import *
from ssltools import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from l_method import agglomerative_l_method

dataset = get_iris()

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


badness_engine = KmeansMockingNested(30, dataset.X)
badness = badness_engine.run(seed_randomly(0.1))
print('badness random:', badness)

for group in badness_engine.groups:
    seeding_cnt = group.seeding_cnt()

    if seeding_cnt == 0:
        continue

    print('seeding_cnt:', seeding_cnt, '/', group.cnt)

    l_method = agglomerative_l_method(group.X)
    centroids = l_method.cluster_centers_
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