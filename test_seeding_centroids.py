from pipe import Pipe
from ssltools import *
from utils import load_x, load_y

file = './datasets/iris/iris.data'

Pipe()\
    .x(load_x(file))\
    .y(load_y(file))\
    .pipe(seeding_centroids(0.1))