from pipe import Pipe
from wrapper import knn
from pipetools import predict, dump, load_y, copy, evaluate, echo
from utils import load_x, load_y
from multipipetools import average
from splitter import cross

file = './datasets/iris/iris.data'

a = Pipe() \
    .x(load_x(file)) \
    .y(load_y(file))\
    .split(5, cross()) \
        .pipe(knn(1)) \
        .pipe(copy('x_test', 'x')) \
        .pipe(copy('y_test', 'y')) \
        .pipe(predict()) \
        .pipe(evaluate()) \
    .merge('evaluation', average('evaluation'))\
    .pipe(dump('evaluation'))


