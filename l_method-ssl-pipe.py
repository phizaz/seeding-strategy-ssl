from multipipetools import average
from pipe import Pipe
from pipetools import *
from ssltools import *
from utils import load_x, load_y
from wrapper import agglomerative_l_method, knn

file = './datasets/iris/iris.data'
# file = './datasets/pendigits/pendigits.tra'
points = load_x(file, delimiter=',')
target = load_y(file, delimiter=',')

def l_method(neighbors):
    def fn(pipe):
        p = pipe \
            .split(5) \
                .pipe(agglomerative_l_method()) \
                .pipe(copy('y', 'y_bak')) \
                .y(seeding_random(0.1)) \
                .y(label_consensus()) \
                .pipe(knn(neighbors)) \
                .pipe(predict()) \
                .pipe(copy('y_bak', 'y')) \
                .pipe(evaluate()) \
            .merge('evaluation', average('evaluation'))
        return p
    return fn

p = Pipe() \
    .x(points) \
    .y(target) \
    .connect(start_timer()) \
    .connect(l_method(neighbors=1)) \
    .connect(stop_timer()) \
    .pipe(dump('evaluation'))
