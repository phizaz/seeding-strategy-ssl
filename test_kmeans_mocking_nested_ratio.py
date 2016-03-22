from dataset import *
from pipe import Pipe
from wrapper import *
from badness import *
from ssltools import *

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

kmeans_mocking_nested_ration = KmeansMockingNestedRatio(dataset.cluster_cnt * 3, dataset.X)
# random_seed = seed_randomly(0.1)
# badness = kmeans_mocking_nested_ration.run(random_seed)
# print('badness random:', badness)

random_cache = seed_cache('pendigits_prob-0.06.json')
badness = kmeans_mocking_nested_ration.run(random_cache)
print('badness cache 0.06:', badness)

random_cache = seed_cache('pendigits_prob-0.05.json')
badness = kmeans_mocking_nested_ration.run(random_cache)
print('badness cache 0.05:', badness)