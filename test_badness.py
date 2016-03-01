from pipe import Pipe
from wrapper import kmeans, knn, badness
from pipetools import *
from ssltools import *
from concurrent.futures import ProcessPoolExecutor

clusters_count = 10
# file = './datasets/iris/iris.data'
# file_test = './datasets/iris/iris.data'
file = './datasets/pendigits/pendigits.tra'
file_test = './datasets/pendigits/pendigits.tes'

def kmeans_ssl(clusters, neighbors):
    def fn(pipe):
        p = pipe \
            .pipe(kmeans(clusters)) \
            .y(label_consensus()) \
            .pipe(knn(neighbors)) \
            .pipe(predict()) \
            .pipe(evaluate())
        return p
    return fn

p = Pipe() \
    .x(load_x(file)) \
    .y(load_y(file)) \
    .x_test(load_x(file_test)) \
    .y_test(load_y(file_test)) \
    .y(seeding_random(0.1))

def ssl(pipe):
    return pipe\
        .connect(kmeans_ssl(clusters_count, 1))\
        .connect(stop())
def bad(pipe):
    return pipe\
        .pipe(badness())\
        .connect(stop())


with ProcessPoolExecutor() as executor:
    ssl_pipe = executor.submit(ssl, p).result()
    bad_pipe = executor.submit(bad, p).result()

ssl_acc = ssl_pipe['evaluation']
bad_metric = bad_pipe['badness']

print('ssl_acc:', ssl_acc, ssl_acc[0] / ssl_acc[1])
print('badness:', bad_metric)