from fastcluster import linkage
from disjoint import DisjointSet
import numpy as np

def get_centroids(X, belong_to):
    clusters_cnt = max(belong_to) + 1
    acc_centroids_by_group = [np.zeros(X[0].shape) for i in range(clusters_cnt)]
    cluster_member_cnt = [0 for i in range(clusters_cnt)]
    for i, x in enumerate(X):
        belongs = belong_to[i]
        cluster_member_cnt[belongs] += 1
        acc_centroids_by_group[belongs] += x

    centroids = [acc_centroid / member_cnt
                 for acc_centroid, member_cnt
                 in zip(acc_centroids_by_group, cluster_member_cnt)]

    return centroids, cluster_member_cnt

def agglomerative_clutering(X, max_merge_dist):
    merge_hist = linkage(X, method='ward', metric='euclidean', preserve_input=True)

    disjoint = DisjointSet(len(X))
    for a, b, merge_dist, _ in merge_hist:
        if merge_dist > max_merge_dist:
            break

        a, b = int(a), int(b)
        disjoint.join(a, b)

    belong_to = [disjoint.parent(i) for i in range(len(X))]

    # rename the cluster name to be 0 -> cluster_count - 1
    cluster_map = {}
    cluster_name = 0
    belong_to_renamed = []
    for each in belong_to:
        if not each in cluster_map:
            cluster_map[each] = cluster_name
            cluster_name += 1
        belong_to_renamed.append(cluster_map[each])

    # print('belong_to_renamed:', belong_to_renamed)

    centroids, cluster_member_cnt = get_centroids(X, belong_to_renamed)
    print('centroids:', centroids)
    print('cluster_member_cnt:', cluster_member_cnt)

    return centroids, cluster_member_cnt
