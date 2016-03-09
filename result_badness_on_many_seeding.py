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

def create_seeding_fns(dataset):

    seeding_fns = []
    seeding_names = []

    probs = np.linspace(0.01, 0.1, 10)

    # seeding with probability
    for prob in probs:
        seeding_fns.append(seeding_random(prob))
        seeding_names.append('prob-' + str(prob))

    # seeding centroids
    for prob in probs:
        seeding_fns.append(seeding_centroids(prob))
        seeding_names.append('centroid-prob-' + str(prob))

    # seeding some clusters
    half_cluster_cnt = int(dataset.cluster_cnt / 2)
    for cluster_cnt in range(half_cluster_cnt, dataset.cluster_cnt + 1):
        prob = 0.1
        seeding_fns.append(seeding_some(prob, cluster_cnt))
        seeding_names.append('some-' + str(cluster_cnt) + '-prob-' + str(prob))

    return seeding_fns, seeding_names

def seeder(seeding_fns, seeding_names):
    def map_fn(inst, idx, total):
        seeding_fn = seeding_fns[idx]
        y_seed = seeding_fn(inst)
        # print('pipe no:', idx, 'prob:', probs[idx])
        # print('y_seed:', y_seed)
        return inst \
            .set('y_seed', y_seed) \
            .set('name', seeding_names[idx])

    return map_fn

def run_and_save(dataset):
    print('dataset:', dataset.name)
    print('has_testdata:', dataset.has_testdata())

    # just set as defacto standard kmeans
    cluster_cnt = dataset.cluster_cnt * 3

    seeding_fns, seeding_names = create_seeding_fns(dataset)
    print('seeding_names:', seeding_names)

    def normal():
        return Pipe() \
            .x(dataset.X) \
            .y(dataset.Y) \
            .x_test(dataset.X_test) \
            .y_test(dataset.Y_test) \
            .pipe(badness_naive(prepare=True)) \
            .pipe(badness_agglomeratvie_l_method(prepare=True)) \
            .pipe(badness_denclue(bandwidth=dataset.bandwidth, prepare=True)) \
            .split(len(seeding_fns), seeder(seeding_fns, seeding_names)) \
                .pipe(badness_naive()) \
                .pipe(badness_agglomeratvie_l_method()) \
                .pipe(badness_denclue()) \
                .connect(kmeans_ssl(cluster_cnt, dataset.K_for_KNN)) \
            .merge('result', group('evaluation', 'badness', 'badness_denclue', 'badness_naive', 'name')) \
            .connect(stop())

    def cv():
        return Pipe() \
            .x(dataset.X) \
            .y(dataset.Y) \
            .pipe(badness_naive(prepare=True)) \
            .pipe(badness_agglomeratvie_l_method(prepare=True)) \
            .pipe(badness_denclue(bandwidth=dataset.bandwidth, prepare=True)) \
            .split(len(seeding_fns), seeder(seeding_fns, seeding_names)) \
                .pipe(badness_naive()) \
                .pipe(badness_agglomeratvie_l_method()) \
                .pipe(badness_denclue()) \
                .split(10, cross('y_seed')) \
                    .connect(kmeans_ssl(cluster_cnt, dataset.K_for_KNN)) \
                .merge('evaluation', total('evaluation')) \
            .merge('result', group('evaluation', 'badness', 'badness_denclue', 'badness_naive', 'name')) \
            .connect(stop())

    if dataset.has_testdata():
        fn = normal
    else:
        fn = cv

    result = fn()
    # result = fn(dataset, [0.2])

    with open('results/badness_on_many_seeding-' + dataset.name + '.json', 'w') as file:
        json.dump(result['result'], file)

with ProcessPoolExecutor() as executor:
    for dataset in datasets:
        executor.submit(run_and_save, dataset)
