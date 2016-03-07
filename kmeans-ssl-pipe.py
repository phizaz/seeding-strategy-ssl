from multipipetools import average
from pipe import Pipe
from pipetools import *
from ssltools import *
from wrapper import kmeans, knn

clusters_count = 10
# file = './datasets/iris/iris.data'
# file_test = './datasets/iris/iris.data'
file = './datasets/pendigits/pendigits.tra'
file_test = './datasets/pendigits/pendigits.tes'

def kmeans_ssl(clusters, neighbors):
    def fn(pipe):
        p = pipe \
            .split(5) \
                .pipe(kmeans(clusters)) \
                .y(seeding_centroids(0.1)) \
                .y(label_consensus()) \
                .pipe(knn(neighbors)) \
                .pipe(predict()) \
                .pipe(evaluate()) \
            .merge('evaluation', average('evaluation'))
        return p
    return fn

p = Pipe() \
    .x(load_x(file)) \
    .y(load_y(file)) \
    .x_test(load_x(file_test))\
    .y_test(load_y(file_test))\
    .connect(start_timer()) \
    .connect(kmeans_ssl(clusters=clusters_count, neighbors=1)) \
    .connect(stop_timer()) \
    .pipe(dump('evaluation'))
