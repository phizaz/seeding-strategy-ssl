from kde import *
from dataset import *

dataset = get_yeast()

centroids = denclue(dataset.X, dataset.bandwidth, sample_size=len(dataset.X))

# print('cnt:', len(centroids))
# print('centroids:', centroids)
