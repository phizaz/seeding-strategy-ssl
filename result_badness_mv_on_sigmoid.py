from concurrent.futures import ProcessPoolExecutor

from dataset import *
from multipipetools import *
from ssltools import *
from wrapper import *
import numpy as np
from cache import StorageCache

'''
generate results for majority voronoid (best possible clustering) on different sigmoids
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
]

datasets.sort(key=lambda dataset: len(dataset.X))
print('datasets:', ', '.join(list(map(lambda dataset: dataset.name, datasets))))

def kmeans_ssl(clusters, neighbors, field):
    def fn(pipe):
        p = pipe \
            .pipe(kmeans(clusters)) \
            .y(label_consensus()) \
            .pipe(knn(neighbors)) \
            .pipe(predict()) \
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

def create_voronoid_sigmoid():
    params = []
    names = []
    space = np.logspace(-1, -20, 7, base=math.exp(1))

    for c in space:
        params.append(c)
        names.append('sig-' + str(c))

    return params, names

def voronoid_sigmoid(params, names):
    def map_fn(inst, idx, total):
        return inst.set('voronoid_sigmoid', params[idx])

    return map_fn

def run_and_save(dataset):
    print('dataset:', dataset.name)
    print('has_testdata:', dataset.has_testdata())

    seeding_fns, seeding_names = create_seeding_fns(dataset)
    print('seeding_names:', seeding_names)

    voronoid_sigmoids, voronoid_sigmoid_names = create_voronoid_sigmoid()
    print('voronoid_sigmoids:', voronoid_sigmoids)

    def normal():
        return Pipe() \
            .x(dataset.X) \
            .y(dataset.Y) \
            .x_test(dataset.X_test) \
            .y_test(dataset.Y_test) \
            .pipe(badness_naive(prepare=True)) \
            .pipe(badness_majority_voronoid(prepare=True)) \
            .split(len(seeding_fns), seeder(seeding_fns, seeding_names, name=dataset.name)) \
                .pipe(badness_naive()) \
                .split(len(voronoid_sigmoids), voronoid_sigmoid(voronoid_sigmoids, voronoid_sigmoid_names)) \
                    .pipe(badness_majority_voronoid()) \
                .merge(['badness_majority_voronoid',
                        'voronoid_sigmoid'],
                       flat_group('badness_majority_voronoid'),
                       flat_group('voronoid_sigmoid')) \
                .connect(kmeans_ssl(dataset.cluster_cnt * 3, dataset.K_for_KNN, 'evaluation_kmeans_3')) \
            .merge('result', group('evaluation_kmeans_3',
                                   'badness_majority_voronoid',
                                   'badness_naive',
                                   'voronoid_sigmoid',
                                   'name')) \
            .connect(stop())

    if dataset.has_testdata():
        fn = normal
    else:
        raise Exception('no cv')

    result = fn()

    with open('results/badness_mv_on_sigmoid-' + dataset.name + '.json', 'w') as file:
        json.dump(result['result'], file)

# with ProcessPoolExecutor() as executor:
#     for dataset in datasets:
#         executor.submit(run_and_save, dataset)

for dataset in datasets:
    run_and_save(dataset)
