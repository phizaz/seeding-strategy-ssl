from collections import Counter
from random import shuffle, randint
from pyrsistent import pvector
from sklearn.cluster import KMeans
from numpy import inner
import itertools

def label_consensus():
    def fn(inst):
        if not 'prediction' in inst:
            raise Exception('no prediction')
        if not 'y' in inst:
            raise Exception('no y')

        prediction = inst['prediction']
        y = inst['y']
        # print('y:', y)
        # print('prediction:', prediction)
        group_labels = [None for each in range(max(prediction) + 1)]
        for i, g in enumerate(prediction):
            label = y[i]
            if label:
                if not group_labels[g]:
                    group_labels[g] = Counter()
                group_labels[g][label] += 1
        # print('group_labels:', group_labels)
        majority = list(map(lambda x: x.most_common(1)[0] if x else None, group_labels))
        # print('majority:', majority)
        new_y = [None for i in range(len(y))]
        for i, g in enumerate(prediction):
            # majority comes in (label, freq) or None
            maj = majority[g]
            if maj:
                # if there is a majority
                # take only the first part
                new_y[i] = maj[0]
        # randomly fill the rest (None)
        for i, each in enumerate(new_y):
            if not each:
                # randomly select one label from another
                # if unfortunate we select None again
                # this's why we put it inside while loop
                while True:
                    r = randint(0, len(new_y) - 1)
                    v = new_y[r]
                    if v:
                        new_y[i] = v
                        break
        return pvector(new_y)

    return fn

def seeding_random(prob):
    def fn(inst):
        if not 'y' in inst:
            raise Exception('no y')

        y = inst['y']
        seq = [i for i in range(len(y))]
        shuffle(seq)
        select_cnt = int(len(y) * prob)
        selected_ids = seq[: select_cnt]
        new_y = [None for i in range(len(y))]
        for id in selected_ids:
            new_y[id] = y[id]
        return pvector(new_y)

    return fn

def seeding_centroids(prob):

    def fn(inst):
        if not 'y' in inst:
            raise Exception('no y')
        if not 'x' in inst:
            raise Exception('no x')

        x = inst['x']
        y = inst['y']
        seeding_cnt = int(len(y) * prob)
        kmeans = KMeans(n_clusters=seeding_cnt)
        kmeans.fit(x)
        clusters = kmeans.labels_
        # print('clusters:', clusters)
        centroids = kmeans.cluster_centers_
        # print('centroids:', centroids)

        # group points in the same cluster into the same array, with index
        map_cluster_idx = [[] for i in range(seeding_cnt)]
        for idx, cluster in enumerate(clusters):
            map_cluster_idx[cluster].append(idx)

        # print('map_cluster_idx:', map_cluster_idx)

        new_y = [None for i in range(len(y))]
        # find the closest point to each centroid
        for cluster_no, cluster in enumerate(map_cluster_idx):
            min_sqdist = float('inf')
            min_idx = None
            a = centroids[cluster_no]
            for idx in cluster:
                b = x[idx]
                ab = a - b
                sqdist = inner(ab, ab)
                if sqdist < min_sqdist:
                    min_sqdist = sqdist
                    min_idx = idx
            new_y[min_idx] = y[idx]

        # print('new_y:', new_y)
        return pvector(new_y)

    return fn

def seeding_equally(prob):
    # this will work on natural clusters
    def fn(inst):
        if not 'x' in inst:
            raise Exception('no x')
        if not 'y' in inst:
            raise Exception('no y')

        x = inst['x']
        y = inst['y']

        seeding_cnt = int(len(y) * prob)
        counter = Counter(y)
        clusters_name = counter.keys()
        clusters_cnt = len(Counter(y))
        seeding_per_cluster = int(seeding_cnt / clusters_cnt)

        # group points in the same cluster into the same array, with index
        map_cluster_idx = {}
        for name in clusters_name:
            map_cluster_idx[name] = []
        for idx, cluster in enumerate(y):
            map_cluster_idx[cluster].append(idx)

        # select seending from each cluster
        new_y = [None for i in range(len(y))]
        cluster_no = 0
        seeding_selected = 0
        for cluster, points in map_cluster_idx.items():
            shuffle(points)

            if cluster_no == clusters_cnt - 1:
                # last cluster select the rest
                selecting_cnt = seeding_cnt - seeding_selected
            else:
                selecting_cnt = seeding_per_cluster

            seeding_selected += selecting_cnt

            for selected in itertools.islice(points, selecting_cnt):
                # print('selected:', selected)
                # print('selected:', y[selected])
                new_y[selected] = y[selected]

        # print('new_y:', new_y)

        return pvector(new_y)

    return fn