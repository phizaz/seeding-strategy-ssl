from concurrent.futures import ProcessPoolExecutor

from dataset import *
from multipipetools import *
from ssltools import *
from wrapper import *
import numpy as np
from cache import StorageCache

datasets = [
    # get_iris(),
    # get_pendigits(),
    get_yeast(),
    # get_satimage(),
    # get_banknote(),
    # get_letter(), # large dataset
    # get_magic(), # super slow for kde hill climbing
    # get_eeg(), # is not suitable for SSL
    # get_spam(), # prone to imbalanced problem
]

def kmeans_ssl(clusters, neighbors, field):
    def fn(pipe):
        p = pipe \
            .pipe(kmeans(clusters)) \
            .y(label_consensus()) \
            .pipe(kmeans_predict()) \
            .pipe(evaluate()) \
            .pipe(copy('evaluation', field))
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
    # problematic !! don't know why ...
    #for prob in probs:
    #    seeding_fns.append(seeding_centroids(prob))
    #    seeding_names.append('centroid-prob-' + str(prob))

    # seeding some clusters
    half_cluster_cnt = int(dataset.cluster_cnt / 2)
    for cluster_cnt in range(half_cluster_cnt, dataset.cluster_cnt + 1):
        prob = 0.1
        seeding_fns.append(seeding_some(prob, cluster_cnt))
        seeding_names.append('some-' + str(cluster_cnt) + '-prob-' + str(prob))

    return seeding_fns, seeding_names

def seeder(seeding_fns, seeding_names, name):
    def map_fn(inst, idx, total):
        # now using caching technique to have the consistent result for each runtime
        file = 'seeding/' + name + '_' + seeding_names[idx] + '.json'
        cache = StorageCache(file)

        if not cache.isnew():
            y_seed = np.array(cache.get())
        else:
            seeding_fn = seeding_fns[idx]
            y_seed = seeding_fn(inst)

            # save to the cache
            cache.update(array_to_list(y_seed))
            cache.save()

        #print('pipe no:', idx)
        # print('y_seed:', y_seed)
        return inst \
            .set('y_seed', y_seed) \
            .set('name', seeding_names[idx])

    return map_fn

def create_kmeans_sigmoid():
    params = []
    names = []
    space = np.logspace(-1, -20, 20)

    for c in space:
        params.append(c)
        names.append('sig-' + str(c))

    return params, names

def kmeans_sigmoid(params, names):
    def map_fn(inst, idx, total):
        return inst.set('kmeans_sigmoid', params[idx])

    return map_fn

def run_and_save(dataset):
    print('dataset:', dataset.name)
    print('has_testdata:', dataset.has_testdata())

    seeding_fns, seeding_names = create_seeding_fns(dataset)
    print('seeding_names:', seeding_names)

    kmeans_sigmoids, kmeans_sigmoid_names = create_kmeans_sigmoid()
    print('kmeans_sigmoids:', kmeans_sigmoids)

    def normal():
        return Pipe() \
            .x(dataset.X) \
            .y(dataset.Y) \
            .x_test(dataset.X_test) \
            .y_test(dataset.Y_test) \
            .pipe(badness_naive(prepare=True)) \
            .pipe(let('kmeans_clusters_cnt', dataset.cluster_cnt * 3)) \
            .pipe(badness_kmeans_mocking(prepare=True)) \
            .split(len(seeding_fns), seeder(seeding_fns, seeding_names, name=dataset.name)) \
                .pipe(badness_naive()) \
                .split(len(kmeans_sigmoids), kmeans_sigmoid(kmeans_sigmoids, kmeans_sigmoid_names)) \
                    .pipe(badness_kmeans_mocking()) \
                .merge(['badness_kmeans_mocking',
                        'kmeans_sigmoid'],
                       flat_group('badness_kmeans_mocking'),
                       flat_group('kmeans_sigmoid')) \
                .connect(kmeans_ssl(dataset.cluster_cnt, dataset.K_for_KNN, 'evaluation_kmeans_1')) \
                .connect(kmeans_ssl(dataset.cluster_cnt * 3, dataset.K_for_KNN, 'evaluation_kmeans_3')) \
            .merge('result', group('evaluation_kmeans_1',
                                   'evaluation_kmeans_3',
                                   'badness_kmeans_mocking',
                                   'badness_naive',
                                   'kmeans_sigmoid',
                                   'name')) \
            .connect(stop())

    def cv():
        return Pipe() \
            .x(dataset.X) \
            .y(dataset.Y) \
            .pipe(badness_naive(prepare=True)) \
            .pipe(let('kmeans_clusters_cnt', dataset.cluster_cnt * 3)) \
            .pipe(badness_kmeans_mocking(prepare=True)) \
            .split(len(seeding_fns), seeder(seeding_fns, seeding_names, name=dataset.name)) \
                .pipe(badness_naive()) \
                .split(len(kmeans_sigmoids), kmeans_sigmoid(kmeans_sigmoids, kmeans_sigmoid_names)) \
                    .pipe(badness_kmeans_mocking()) \
                .merge(['badness_kmeans_mocking',
                        'kmeans_sigmoid'],
                       flat_group('badness_kmeans_mocking'),
                       flat_group('kmeans_sigmoid')) \
                .split(10, cross('y_seed')) \
                    .connect(kmeans_ssl(dataset.cluster_cnt, dataset.K_for_KNN, 'evaluation_kmeans_1')) \
                    .connect(kmeans_ssl(dataset.cluster_cnt * 3, dataset.K_for_KNN, 'evaluation_kmeans_3')) \
                .merge(['evaluation_kmeans_1', 'evaluation_kmeans_3'], total('evaluation_kmeans_1'), total('evaluation_kmeans_3')) \
            .merge('result', group('evaluation_kmeans_1',
                                   'evaluation_kmeans_3',
                                   'badness_kmeans_mocking',
                                   'badness_naive',
                                   'kmeans_sigmoid',
                                   'name')) \
            .connect(stop())

    if dataset.has_testdata():
        fn = normal
    else:
        fn = cv

    result = fn()
    # result = fn(dataset, [0.2])

    with open('results/badness_on_many_seeding-' + dataset.name + '.json', 'w') as file:
        json.dump(result['result'], file)

# with ProcessPoolExecutor() as executor:
#     for dataset in datasets:
#         executor.submit(run_and_save, dataset)

for dataset in datasets:
    run_and_save(dataset)
