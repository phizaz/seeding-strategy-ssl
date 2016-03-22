from agglomerative_clustering import AgglomerativeClusteringMaxMergeDist
from dataset import *

dataset = get_iris()

print('dataset size:', len(dataset.X))

agg = AgglomerativeClusteringMaxMergeDist()
centroids, cluster_member_cnt = agg.fit(dataset.X, 0.2)

print('grouped size:', len(centroids))
