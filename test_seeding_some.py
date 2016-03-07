from pipe import Pipe
from wrapper import kmeans
from pipetools import *
from ssltools import *
from dataset import get_iris

dataset = get_iris()

a = Pipe() \
    .x(dataset.X) \
    .y(dataset.Y) \
    .y_seed(seeding_some(0.1, cluster_cnt=3)) \
    .connect(stop())

print(a['y_seed'])