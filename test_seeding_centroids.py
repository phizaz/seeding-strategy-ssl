from pipe import Pipe
from pipetools import *
from ssltools import *

file = './datasets/iris/iris.data'

Pipe()\
    .x(load_x(file))\
    .y(load_y(file))\
    .pipe(seeding_centroids(0.1))