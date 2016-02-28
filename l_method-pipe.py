from pipe import Pipe
from wrapper import recursive_agglomerative_l_method, agglomerative_l_method
from pipetools import dump, evaluate, load_x, start_timer, stop_timer

file = './datasets/pendigits/pendigits.tra'
# file = './datasets/iris/iris.data'

Pipe() \
    .x(load_x(file, delimiter=',')) \
    .connect(start_timer())\
    .pipe(agglomerative_l_method())\
    .connect(stop_timer())\
    .pipe(dump('prediction'))