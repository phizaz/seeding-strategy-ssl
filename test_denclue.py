from kde import *
from dataset import *

dataset = get_iris()

centroids = denclue(dataset.X, dataset.bandwidth, sample_size=int(len(dataset.X) / 2))

print('cnt:', len(centroids))
print('centroids:', centroids)