from pipe import Pipe
from wrapper import kmeans
from pipetools import *
from dataset import *

dataset = get_pendigits()

a = Pipe()\
    .x(dataset.X)\
    .pipe(kmeans(dataset.cluster_cnt))\
    .connect(stop())

print(a)