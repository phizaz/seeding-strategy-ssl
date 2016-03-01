from pipe import Pipe
from wrapper import kmeans
from pipetools import *
from dataset import get_iris

iris = get_iris()

a = Pipe()\
    .x(iris.X)\
    .pipe(kmeans(3))\
    .connect(stop())

print(a)