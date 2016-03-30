from concurrent.futures import ProcessPoolExecutor

from dataset import *
from multipipetools import *
from ssltools import *
from wrapper import *
import numpy as np
from cache import StorageCache

'''
generate results for
cluster mocking on many linkages
'''

datasets = [
    get_iris_with_test().rescale(),
    get_pendigits().rescale(),
    get_yeast_with_test().rescale(),
    get_satimage().rescale(),
    get_banknote_with_test().rescale(),
    get_spam_with_test().rescale(),
    get_drd_with_test().rescale(),
    get_imagesegment().rescale(),
    get_pageblock_with_test().rescale(),
    get_statlogsegment_with_test().rescale(),
    get_winequality_with_test('white').rescale(),
    get_winequality_with_test('red').rescale(),

    # get_magic_with_test().rescale(),
    # get_letter_with_test().rescale(),
    # get_eeg_with_test().rescale(),
    # get_auslan_with_test().rescale(),
]

datasets.sort(key=lambda dataset: len(dataset.X))
print('datasets:', ', '.join(list(map(lambda dataset: dataset.name, datasets))))

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

def equal_X(X, X_ori):
    def equal(a, b):
        for aa, bb in zip(a, b):
            if abs(aa - bb) > 1e-8:
                return False
        return True

    for i, (x, x_ori) in enumerate(zip(X, X_ori)):
        if not equal(x, x_ori):
            print(i, x, x_ori)
            raise Exception('not equal input')

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

        return inst \
            .set('y_seed', y_seed) \
            .set('name', seeding_names[idx])

    return map_fn

def run_and_save(dataset):
    print('has_testdata:', dataset.has_testdata())

    seeding_fns, seeding_names = create_seeding_fns(dataset)
    print('seeding_names:', seeding_names)

    def kmeans_ssl(clusters, neighbors, name):
        def fn(pipe):
            p = pipe \
                .pipe(kmeans(clusters)) \
                .pipe(copy('prediction', 'cluster_labels')) \
                .y(label_consensus()) \
                .pipe(label_correctness('y', 'y_ori')) \
                .pipe(knn(neighbors)) \
                .pipe(predict()) \
                .pipe(evaluate()) \
                .pipe(copy('evaluation', 'evaluation_' + name)) \
                .pipe(copy('label_correctness', 'label_correctness_' + name))
            return p

        return fn

    def normal():
        return Pipe() \
            .x(dataset.X) \
            .y(dataset.Y) \
            .pipe(copy('y', 'y_ori')) \
            .x_test(dataset.X_test) \
            .y_test(dataset.Y_test) \
            .split(len(seeding_fns), seeder(seeding_fns, seeding_names, name=dataset.name)) \
            .connect(kmeans_ssl(dataset.cluster_cnt * 3, dataset.K_for_KNN, 'kmeans_3')) \
            .pipe(goodness_cluster_mocking()) \
            .pipe(goodness_cluster_mocking_nested_ratio()) \
            .pipe(goodness_cluster_mocking_nested_split(method='ward')) \
            .merge('result', group('evaluation_kmeans_3',
                                   'label_correctness_kmeans_3',
                                   'goodness_cluster_mocking',
                                   'goodness_cluster_mocking_nested_ratio',
                                   'goodness_cluster_mocking_nested_split_ward',
                                   'name')) \
            .connect(stop())

    if dataset.has_testdata():
        fn = normal
    else:
        raise Exception('no cv')

    result = fn()

    with open('results/goodness_cluster_mocking-' + dataset.name + '.json', 'w') as file:
        json.dump(result['result'], file)

# with ProcessPoolExecutor() as executor:
#     for dataset in datasets:
#         executor.submit(run_and_save, dataset)

for i, dataset in enumerate(datasets):
    print('dataset no:', i + 1, '/', len(datasets))
    print('dataset:', dataset.name)
    print('yet to be done:', ', '.join(list(map(lambda d: d.name, datasets[i+1:]))))
    run_and_save(dataset)
