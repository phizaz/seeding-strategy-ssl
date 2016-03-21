from dataset import *
from pipe import Pipe
from wrapper import *
from badness import *
from ssltools import *

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


badness_engine = KmeansMocking(30, dataset.X)

badness = badness_engine.run(seed_randomly(0.1), 1e-10)
print('badness random:', badness)

# badness = badness_engine.run(seed_some(0.1, 1), 1e-20)
# print('badness some 1:', badness)