from pipe import Pipe
from wrapper import agglomerative
from pipetools import dump, evaluate
from utils import load_x

file = './datasets/iris/iris.data'

Pipe()\
    .x(load_x(file, delimiter=','))\
    .pipe(agglomerative(n_clusters=3))\
    .pipe(dump('prediction'))