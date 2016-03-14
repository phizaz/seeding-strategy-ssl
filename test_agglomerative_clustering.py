from agglomerative_clustering import agglomerative_clutering
from dataset import *

dataset = get_iris()

print('dataset size:', len(dataset.X))

centroids, cluster_member_cnt = agglomerative_clutering(dataset.X, 0.2)

print('grouped size:', len(centroids))
