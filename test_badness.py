import json

from dataset import *
from multipipetools import group
from ssltools import *
from wrapper import *

# data = get_iris()
data = get_pendigits()
clusters_count = data.cluster_cnt

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
    .x(data.X) \
    .y(data.Y) \
    .x_test(data.X_test) \
    .y_test(data.Y_test) \
    .pipe(badness_agglomeratvie_l_method(prepare=True))\
    .split(3)\
        .y_seed(seeding_random(0.1)) \
        .pipe(badness_agglomeratvie_l_method()) \
        .connect(kmeans_ssl(clusters_count, data.K_for_KNN))\
    .merge('result', group('evaluation', 'badness'))\
    .connect(stop())

result = p['result']

print('result:', result)

with open('results/badness_agglomerative_l_method-' + data.name + '.json', 'w') as file:
    json.dump(result, file)