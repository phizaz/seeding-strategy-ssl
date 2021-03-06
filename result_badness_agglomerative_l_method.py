from concurrent.futures import ProcessPoolExecutor

from dataset import *
from multipipetools import total, group, cross
from ssltools import *
from wrapper import *

datasets = [
    # get_iris(),
    # get_yeast(),
    # get_letter(),
    get_pendigits(),
    # get_satimage(),
    # get_banknote(),
    # get_eeg(),
    # get_magic(),
    # get_spam()
]

random_cnt = 25
seeding_prob = 0.1

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

def normal(data, prob):
    cluster_cnt = data.cluster_cnt * 3

    return Pipe() \
        .x(data.X) \
        .y(data.Y) \
        .x_test(data.X_test) \
        .y_test(data.Y_test) \
        .pipe(badness_agglomeratvie_l_method(prepare=True)) \
        .split(random_cnt) \
            .y_seed(seeding_random(prob)) \
            .pipe(badness_agglomeratvie_l_method()) \
            .connect(kmeans_ssl(cluster_cnt, data.K_for_KNN)) \
        .merge('result', group('evaluation', 'badness')) \
        .connect(stop())

def cv(data, prob):
    cluster_cnt = data.cluster_cnt * 3

    return Pipe() \
        .x(data.X) \
        .y(data.Y) \
        .pipe(badness_agglomeratvie_l_method(prepare=True)) \
        .split(random_cnt) \
            .y_seed(seeding_random(prob)) \
            .pipe(badness_agglomeratvie_l_method()) \
            .split(10, cross('y_seed')) \
                .connect(kmeans_ssl(cluster_cnt, data.K_for_KNN)) \
            .merge('evaluation', total('evaluation')) \
        .merge('result', group('evaluation', 'badness')) \
        .connect(stop())


def run_and_save(dataset, prob):
    if dataset.has_testdata():
        fn = normal
    else:
        fn = cv

    result = fn(dataset, prob)
    with open('results/badness_agglomerative_l_method_prob-' + seeding_prob + '-' + dataset.name + '.json', 'w') as file:
        json.dump(result['result'], file)

with ProcessPoolExecutor() as executor:
    for dataset in datasets:
        executor.submit(run_and_save, dataset, seeding_prob)
