from pipe import Pipe
from wrapper import kmeans
from pipetools import predict, dump, load_x

file = './datasets/iris/iris.data'

a = Pipe()\
    .x(load_x(file, delimiter=','))\
    .pipe(kmeans(3))\
    .pipe(dump('prediction'))


