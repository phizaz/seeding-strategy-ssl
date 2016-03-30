from dataset import *
from pipe import Pipe
from wrapper import *
from badness import *
from ssltools import *
from sklearn.cluster import KMeans

dataset = get_yeast().rescale()


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

kmeans = KMeans(dataset.cluster_cnt * 3)
kmeans.fit(dataset.X)

model = ClusterMockingNestedRatio(dataset.X, kmeans.labels_)
random_seed = seed_randomly(0.1)
badness = model.run(random_seed)
print('badness random:', badness)
#
# random_cache = seed_cache('yeast_prob-0.06.json')
# badness = model.run(random_cache)
# print('badness cache 0.06:', badness)
#
# random_cache = seed_cache('yeast_prob-0.05.json')
# badness = model.run(random_cache)
# print('badness cache 0.05:', badness)