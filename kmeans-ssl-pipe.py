from pipe import Pipe
from wrapper import kmeans, knn
from pipetools import *
from ssltools import *
from splitter import cross

file = './datasets/iris/iris.data'
points = load_x(file, delimiter=',')
target = load_y(file, delimiter=',')

def kmeans_ssl(clusters, neighbors):
    def fn(pipe):
        p = pipe \
            .split(5) \
                .pipe(kmeans(clusters)) \
                .pipe(predict()) \
                .pipe(copy('y', 'y_bak')) \
                .y(random_select_y(0.1)) \
                .y(label_consensus()) \
                .pipe(knn(neighbors)) \
                .pipe(predict()) \
                .pipe(copy('y_bak', 'y')) \
                .pipe(evaluate()) \
            .merge('evaluation', average('evaluation'))
        return p
    return fn

start_time = time.time()

p = Pipe() \
    .x(points) \
    .connect(start_timer())\
    .y(target) \
    .connect(stop_timer())\
    .split(5, cross()) \
        .connect(kmeans_ssl(clusters=3, neighbors=1)) \
    .merge('evaluation', average('evaluation')) \
    .pipe(dump('evaluation'))

end_time = time.time()

print('time elapsed:', (end_time - start_time))