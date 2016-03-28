from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.neighbors import KNeighborsClassifier, KernelDensity
from sklearn.linear_model import LinearRegression
import numpy as np
import l_method
from badness import *
from pipe import Pipe
from pipetools import *
from util import *
from kde import *
from cache import StorageCache

def kmeans(n_clusters=8, n_init=10):
    def fn(inst):
        x = requires('x', inst)

        #print('kmeans x:', x)

        kmeans = KMeans(n_clusters=n_clusters, n_init=n_init)
        kmeans.fit(x)

        #print('kmeans model:', kmeans)

        return inst.set('model', kmeans)\
            .set('prediction', kmeans.labels_)\
            .set('centroids', kmeans.cluster_centers_)

    return fn

def kmeans_predict():
    def fn(inst):
        x, y, x_test, prediction = requires(['x', 'y', 'x_test', 'prediction'], inst)
        kmeans = requires('model', inst)
        groups = set(prediction)

        group_to_label = dict(zip(groups, [None for i in range(len(groups))]))

        for y, group in zip(y, prediction):
            if group_to_label[group] is not None:
                continue

            group_to_label[group] = y

        groups = kmeans.predict(x_test)
        result = list(map(lambda group: group_to_label[group], groups))

        return inst.set('prediction', result)

    return fn

def knn(*args, **margs):
    def fn(inst):
        x, y = requires(['x', 'y'], inst)

        #print('len x:', len(x))
        #print('len y:', len(y))

        knn = KNeighborsClassifier(*args, **margs)
        knn.fit(x, y)

        #print('knn model:', knn)

        return inst.set('model', knn)

    return fn

def linear_regression(*args, **margs):
    def fn(inst):
        if not 'x' in inst:
            raise Exception('no x')
        if not 'y' in inst:
            raise Exception('no y')

        x = inst['x']
        y = inst['y']

        regression = LinearRegression(*args, **margs)
        regression.fit(x, y)
        #
        # print('coef:', regression.coef_)
        # print('intercept:', regression.intercept_)

        return inst.set('model', regression)

    return fn

def agglomerative(*args, **margs):
    def fn(inst):
        if not 'x' in inst:
            raise Exception('no x')

        x = inst['x']

        agg = AgglomerativeClustering(*args, **margs)
        agg.fit(x)

        return inst.set('model', agg).set('prediction', agg.labels_)

    return fn

def agglomerative_l_method():
    def fn(inst):
        if not 'x' in inst:
            raise Exception('no x')

        x = inst['x']
        model = l_method.agglomerative_l_method(x)
        return inst.set('prediction', model.labels_)\
            .set('centroids', model.cluster_centers_)

    return fn

def recursive_agglomerative_l_method():
    def fn(inst):
        if not 'x' in inst:
            raise Exception('no x')

        x = inst['x']
        clusters = l_method.recursive_agglomerative_l_method(x)
        return inst.set('prediction', clusters)

    return fn

def kernel_density_estimation(*args, **margs):
    def fn(inst):
        if not 'x' in inst:
            raise Exception('no x')

        x = inst['x']

        kde = KernelDensity(*args, **margs)
        kde.fit(x)

        log_pdf = kde.score_samples(x)
        pdf = np.exp(log_pdf)

        return inst.set('model', kde).set('pdf', pdf)

    return fn

def badness_denclue(bandwidth=None, prepare=False, name=None):
    # with given name, it will cache

    def prepare_fn(inst):
        if not bandwidth:
            raise Exception('no bandwidth given!')

        # get good centroids
        x = requires('x', inst)

        if name is not None:
            full_name = 'centroids_' + name + '_denclue_bandwidth_' + str(bandwidth)
            file = 'storage/' + full_name + '.json'
            cache = StorageCache(file)

        if 'cache' in locals() and not cache.isnew():
            # load good centroids from storage and convert to np array
            centroids = np.array(cache.get())
        else:
            if len(x) < 200:
                sample_size = len(x)
            else:
                # 200 < sample_size * 0.2 < 10000
                sample_size = max(min(10000, int(len(x) * 0.2)), 200)

            # get the 'good' centroids
            centroids = denclue(x, bandwidth, sample_size)
            if 'cache' in locals():
                # update cache, save to the storage
                cache.update(array_to_list(centroids))
                cache.save()

        return inst\
            .set('denclue_centroids', centroids)\
            .set('denclue_bandwidth', bandwidth)

    def run(mode='normal'):

        def normal(inst):
            x,\
            y_seed,\
            good_centroids\
                = requires(['x',
                            'y_seed',
                            'denclue_centroids'],
                           inst)

            # build seeding list
            seeding = list(map(lambda x: x[0],
                               filter(lambda a: a[1] is not None,
                                      zip(x, y_seed))))

            if len(seeding) == len(x):
                raise Exception('you probably seed with 100%')

            badness = {
                'md': md_nearest_from_centroids(seeding, good_centroids),
            }

            # print('badness_denclue:', badness)

            return inst.set('badness_denclue', badness)

        def weighted(inst):
            x, \
            y_seed, \
            centroids, \
                = requires(['x',
                            'y_seed',
                            'denclue_centroids'],
                           inst)

            # build seeding list
            seeding = list(map(lambda x: x[0],
                               filter(lambda a: a[1] is not None,
                                      zip(x, y_seed))))

            if len(seeding) == len(x):
                raise Exception('you probably seed with 100%')

            weights = get_centroid_weights(x, centroids)

            badness = {
                'md': md_weighted_nearest_from_centroids(seeding, centroids, weights),
                'voronoid_filling': voronoid_filling(seeding, centroids, weights)
            }

            return inst.set('badness_denclue_weighted', badness)

        modes = {
            'normal': normal,
            'weighted': weighted,
        }

        return modes[mode]


    if prepare:
        return prepare_fn
    else:
        return run

def badness_agglomeratvie_l_method(prepare=False, name=None):
    # with a given name it will cache
    def prepare_fn(inst):
        # get good centroids
        x, y = requires(['x', 'y'], inst)

        if name is not None:
            full_name = 'centroids_' + name + '_l_method'
            file = 'storage/' + full_name + '.json'
            cache = StorageCache(file)

        if 'cache' in locals() and not cache.isnew():
            centroids = np.array(cache.get())
        else:
            # get the 'good' centroids
            result = Pipe() \
                .x(x) \
                .y(y) \
                .pipe(agglomerative_l_method()) \
                .connect(stop())

            if not 'centroids' in result:
                raise Exception('no centroids in pipe')

            centroids = result['centroids']

            if 'cache' in locals():
                # update the cache and save to the storage
                cache.update(array_to_list(centroids))
                cache.save()

        return inst.set('l_method_centroids', centroids)

    def run(mode = 'normal'):

        def normal(inst):
            x, y_seed, centroids = requires(['x', 'y_seed', 'l_method_centroids'], inst)

            # build seeding list
            seeding = list(map(lambda x: x[0],
                               filter(lambda a: a[1] is not None,
                                      zip(x, y_seed))))

            if len(seeding) == len(x):
                raise Exception('you probably seed with 100%')

            badness = {
                'md': md_nearest_from_centroids(seeding, centroids),
            }

            #print('badness-normal:', badness)

            return inst.set('badness_l_method', badness)

        def weighted(inst):
            x, y_seed, centroids = requires(['x', 'y_seed', 'l_method_centroids'], inst)

            # build seeding list
            seeding = list(map(lambda x: x[0],
                               filter(lambda a: a[1] is not None,
                                      zip(x, y_seed))))

            if len(seeding) == len(x):
                raise Exception('you probably seed with 100%')

            weights = get_centroid_weights(x, centroids)

            badness = {
                'md': md_weighted_nearest_from_centroids(seeding, centroids, weights),
                'voronoid_filling': voronoid_filling(seeding, centroids, weights),
            }

            return inst.set('badness_l_method_weighted', badness)

        modes = {
            'normal': normal,
            'weighted': weighted
        }

        return modes[mode]

    if prepare:
        return prepare_fn
    else:
        return run

def badness_hierarchical_voronoid_filling(prepare=False):

    def prepare_fn(inst):
        x = requires('x', inst)

        badness_engine = HierarchicalVoronoidFilling(x)

        return inst.set('hierarchical_voronoid_filling_engine',
                        badness_engine)

    def fn(inst):
        x, y_seed, c, sigmoid = requires(['x',
                                          'y_seed',
                                          'voronoid_c',
                                          'voronoid_sigmoid'],
                                         inst)
        badness_engine = requires('hierarchical_voronoid_filling_engine',
                                  inst)

        seeding = list(map(lambda xy: xy[0],
                           filter(lambda xy: xy[1] is not None,
                                  zip(x, y_seed))))

        return inst.set('badness_hierarchical_voronoid_filling',
                        badness_engine.run(seeding, c, sigmoid))

    modes = {
        True: prepare_fn,
        False: fn
    }

    return modes[prepare]

def badness_majority_voronoid(prepare=False):

    def prepare_fn(inst):
        x = requires('x', inst)
        badness_engine = MajorityVoronoid(x)
        return inst.set('majority_voronoid_engine',
                        badness_engine)

    def fn(inst):
        x, y_seed = requires(['x', 'y_seed'], inst)
        badness_engine = requires('majority_voronoid_engine',
                                  inst)
        return inst.set('badness_majority_voronoid',
                        badness_engine.run(y_seed))

    modes = {
        True: prepare_fn,
        False: fn
    }

    return modes[prepare]

def badness_kmeans_mocking(prepare=False):

    def prepare_fn(inst):
        x, clusters_cnt = requires(['x', 'kmeans_clusters_cnt'], inst)
        badness_engine = KmeansMocking(clusters_cnt, x)
        return inst.set('kmeans_mocking_engine', badness_engine)

    def fn(inst):
        y_seed, sigmoid = requires(['y_seed', 'kmeans_sigmoid'], inst)
        badness_engine = requires('kmeans_mocking_engine', inst)
        return inst.set('badness_kmeans_mocking',
                        badness_engine.run(y_seed, sigmoid))

    modes = {
        True: prepare_fn,
        False: fn
    }

    return modes[prepare]

def badness_kmeans_mocking_nested(prepare=False):

    def prepare_fn(inst):
        x, clusters_cnt = requires(['x', 'kmeans_clusters_cnt'], inst)
        badness_engine = KmeansMockingNested(clusters_cnt, x)
        return inst.set('kmeans_mocking_nested_engine', badness_engine)

    def fn(inst):
        y_seed = requires('y_seed', inst)
        badness_engine = requires('kmeans_mocking_nested_engine', inst)
        return inst.set('badness_kmeans_mocking_nested',
                        badness_engine.run(y_seed))

    modes = {
        True: prepare_fn,
        False: fn
    }

    return modes[prepare]

def badness_kmeans_mocking_nested_ratio(prepare=False):

    def prepare_fn(inst):
        x, clusters_cnt = requires(['x', 'kmeans_clusters_cnt'], inst)
        badness_engine = KmeansMockingNestedRatio(clusters_cnt, x)
        return inst.set('kmeans_mocking_nested_ratio_engine', badness_engine)

    def fn(inst):
        y_seed = requires('y_seed', inst)
        badness_engine = requires('kmeans_mocking_nested_ratio_engine', inst)
        return inst.set('badness_kmeans_mocking_nested_ratio',
                        badness_engine.run(y_seed))

    modes = {
        True: prepare_fn,
        False: fn
    }

    return modes[prepare]

def badness_kmeans_mocking_nested_split(prepare=False, method='ward'):

    def prepare_fn(inst):
        x, clusters_cnt = requires(['x', 'kmeans_clusters_cnt'], inst)
        badness_engine = KmeansMockingNestedSplit(clusters_cnt, x, method)
        return inst.set('kmeans_mocking_nested_split_engine_' + method, badness_engine)

    def fn(inst):
        y_seed = requires('y_seed', inst)
        badness_engine = requires('kmeans_mocking_nested_split_engine_' + method, inst)
        return inst.set('badness_kmeans_mocking_nested_split_' + method,
                        badness_engine.run(y_seed))

    modes = {
        True: prepare_fn,
        False: fn
    }

    return modes[prepare]

def goodness_cluster_mocking_nested_split(method='ward'):

    def fn(inst):
        x, labels = requires(['x', 'cluster_labels'], inst)
        y_seed = requires('y_seed', inst)
        badness_engine = ClusterMockingNestedSplit(x, labels, method)
        return inst.set('goodness_cluster_mocking_nested_split_' + method,
                        badness_engine.run(y_seed))

    return fn

def badness_naive(prepare=False):
    def prepare_fn(inst):
        # get good centroids
        x = requires('x', inst)
        return inst.set('good_centroids_naive', x)

    def fn(inst):
        x, y_seed, good_centroids = requires(['x', 'y_seed', 'good_centroids_naive'], inst)

        # build seeding list
        seeding = list(map(lambda x: x[0],
                           filter(lambda a: a[1] is not None,
                                  zip(x, y_seed))))

        if len(seeding) == len(x):
            raise Exception('you probably seed with 100%')

        badness = {
            'md': md_nearest_from_centroids(seeding, good_centroids),
        }

        #print('badness-naive:', badness)

        return inst.set('badness_naive', badness)

    if prepare:
        return prepare_fn
    else:
        return fn

def good_K_for_KNN():
    # use Dataset class instead, it will do this automatically for you see dataset.py
    def fn(inst):
        x, y, x_test, y_test = \
            requires(['x', 'y', 'x_test', 'y_test'], inst)

        goodK = good_K_for_KNN_with_testdata(x, y, x_test, y_test)
        return inst.set('goodK', goodK)
    
    return fn