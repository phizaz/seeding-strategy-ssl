from util import *
from dataset import *
from kde import *

dataset = get_iris()
centroids = denclue(dataset.X, dataset.bandwidth, len(dataset.X))
weights = get_centroid_weights(dataset.X, centroids)

assert len(centroids) == len(weights)
print('weights:', weights)