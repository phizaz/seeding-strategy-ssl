from concurrent.futures import ProcessPoolExecutor

from dataset import *
from multipipetools import *
from ssltools import *
from wrapper import *
import numpy as np
from cache import StorageCache

'''
generate results for
- agglomerative l-method hierarchical clustering
- kde + denclue (hill climbing)
- naive
'''

# rot is not for use anymore because it is wrong !!
datasets = [
    get_iris_with_test(bandwidth='rot').rescale(),
    get_pendigits(bandwidth='rot').rescale(),
    get_yeast_with_test(bandwidth='rot').rescale(),
    get_satimage(bandwidth='rot').rescale(),
    get_banknote_with_test(bandwidth='rot').rescale(),
    get_spam_with_test(bandwidth='rot').rescale(),
    get_drd_with_test(bandwidth='rot').rescale(),
    get_imagesegment(bandwidth='rot').rescale(),
    get_pageblock_with_test(bandwidth='rot').rescale(),
    get_statlogsegment_with_test(bandwidth='rot').rescale(),
    get_winequality_with_test('white', bandwidth='rot').rescale(),
    get_winequality_with_test('red', bandwidth='rot').rescale(),
]

datasets_cv_ml = [
    get_iris_with_test(bandwidth='cv_ml'),
    get_pendigits(bandwidth='cv_ml'),
    get_yeast_with_test(bandwidth='cv_ml'),
    get_satimage(bandwidth='cv_ml'),
    get_banknote_with_test(bandwidth='cv_ml'),
    get_spam_with_test(bandwidth='cv_ml'),
    get_drd_with_test(bandwidth='cv_ml'),
    get_imagesegment(bandwidth='cv_ml'),
    get_pageblock_with_test(bandwidth='cv_ml'),
    get_statlogsegment_with_test(bandwidth='cv_ml'),
    get_winequality_with_test('white', bandwidth='cv_ml'),
    get_winequality_with_test('red', bandwidth='cv_ml'),
]

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

def run_and_save(dataset, bandwidth_rot, bandwidth_cv_ml):
    print('bandwidth:', 'rot:', bandwidth_rot, 'cv_ml:', bandwidth_cv_ml)

    print('dataset:', dataset.name)
    print('has_testdata:', dataset.has_testdata())

    seeding_fns, seeding_names = create_seeding_fns(dataset)
    print('seeding_names:', seeding_names)

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

    def normal():
        return Pipe() \
            .x(dataset.X) \
            .y(dataset.Y) \
            .x_test(dataset.X_test) \
            .y_test(dataset.Y_test) \
            .pipe(badness_denclue(bandwidth=bandwidth_rot, prepare=True, name=dataset.name, id='rot')) \
            .pipe(badness_denclue(bandwidth=bandwidth_cv_ml, prepare=True, name=dataset.name, id='cv_ml')) \
            .split(len(seeding_fns), seeder(seeding_fns, seeding_names, name=dataset.name)) \
                .pipe(badness_denclue(id='rot')(mode='weighted')) \
                .pipe(badness_denclue(id='cv_ml')(mode='weighted')) \
                .connect(kmeans_ssl(dataset.cluster_cnt * 3, dataset.K_for_KNN, 'evaluation_kmeans_3')) \
            .merge('result', group('evaluation_kmeans_3',
                                   'badness_denclue_weighted_rot',
                                   'badness_denclue_weighted_cv_ml',
                                   'name')) \
            .connect(stop())

    if dataset.has_testdata():
        fn = normal
    else:
        raise Exception('no cv anymore!')

    result = fn()

    with open('results/badness_denclue_bandwidth-' + dataset.name + '.json', 'w') as file:
        json.dump(result['result'], file)


for dataset, dataset_cv_ml in zip(datasets, datasets_cv_ml):
    bandwidth_rot = dataset.bandwidth
    bandwidth_cv_ml = dataset_cv_ml.bandwidth
    run_and_save(dataset, bandwidth_rot, bandwidth_cv_ml)
