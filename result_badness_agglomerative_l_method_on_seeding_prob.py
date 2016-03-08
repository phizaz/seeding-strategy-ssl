import traceback
from concurrent.futures import ProcessPoolExecutor

import sys

from dataset import *
from multipipetools import total, group, cross
from ssltools import *
from wrapper import *
import numpy as np

datasets = [
    get_iris(),
    # get_yeast(),
    # get_letter(),
    # get_pendigits(),
    # get_satimage(),
    # get_banknote(),
    # get_eeg(),
    # get_magic(),
    # get_spam()
]

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

def seeder(probs):
    def map_fn(inst, idx, total):
        seeding_fn = seeding_random(probs[idx])
        y_seed = seeding_fn(inst)
        # print('pipe no:', idx, 'prob:', probs[idx])
        # print('y_seed:', y_seed)
        return inst\
            .set('y_seed', y_seed)\
            .set('name', 'prob-' + str(probs[idx]))

    return map_fn

def normal(data, probs):
    cluster_cnt = data.cluster_cnt * 3

    return Pipe() \
        .x(data.X) \
        .y(data.Y) \
        .x_test(data.X_test) \
        .y_test(data.Y_test) \
        .pipe(badness_agglomeratvie_l_method(prepare=True)) \
        .pipe(badness_kde(data.bandwidth, prepare=True)) \
        .split(len(probs), seeder(probs))\
            .pipe(badness_agglomeratvie_l_method()) \
            .pipe(badness_kde()) \
            .connect(kmeans_ssl(cluster_cnt, data.K_for_KNN)) \
        .merge('result', group('evaluation', 'badness', 'badness_kde', 'name'))\
        .connect(stop())

def cv(data, probs):
    cluster_cnt = data.cluster_cnt * 3

    return Pipe() \
        .x(data.X) \
        .y(data.Y) \
        .pipe(badness_agglomeratvie_l_method(prepare=True)) \
        .pipe(badness_kde(data.bandwidth, prepare=True))\
        .split(len(probs), seeder(probs))\
            .pipe(badness_agglomeratvie_l_method()) \
            .pipe(badness_kde())\
            .split(10, cross('y_seed')) \
                .connect(kmeans_ssl(cluster_cnt, data.K_for_KNN)) \
            .merge('evaluation', total('evaluation')) \
        .merge('result', group('evaluation', 'badness', 'badness_kde', 'name'))\
        .connect(stop())

def run_and_save(dataset):
    print('dataset:', dataset.name)
    print('has_testdata:', dataset.has_testdata())

    if dataset.has_testdata():
        fn = normal
    else:
        fn = cv

    result = fn(dataset, np.linspace(0.01, 0.2, 20))
    # result = fn(dataset, [0.2])

    with open('results/badness_agglomerative_l_method_on_seeding_prob-' + dataset.name + '.json', 'w') as file:
        json.dump(result['result'], file)

with ProcessPoolExecutor() as executor:
    for dataset in datasets:
        executor.submit(run_and_save, dataset)
