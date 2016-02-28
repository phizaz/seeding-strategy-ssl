from pipe import Pipe
from wrapper import kmeans, knn
from pipetools import *
from ssltools import *
from splitter import cross

clusters_count = 3
# file = './datasets/iris/iris.data'
file = './datasets/pendigits/pendigits.tra'
points = load_x(file, delimiter=',')
target = load_y(file, delimiter=',')

def kmeans_ssl(clusters, neighbors):
    def fn(pipe):
        p = pipe \
            .split(5) \
                .pipe(kmeans(clusters)) \
                .pipe(predict()) \
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
    .connect(kmeans_ssl(clusters=clusters_count, neighbors=1)) \
    .connect(stop_timer()) \
    .pipe(dump('evaluation'))
